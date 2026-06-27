"""Triton-based fused-cell implementation of PRLSTM.

Replaces the per-step elementwise op chain (sigmoid x 4, mul, add, tanh) with a
single Triton kernel, while continuing to use cuBLAS (via torch.matmul) for the
W_ih and W_hh matmuls. Wrapped in a torch.autograd.Function with a hand-written
backward so PyTorch sees one op per forward (no traced graph through T steps).

Activation choices in the cell match the canonical PRLSTM modifications:
    1. g_t uses sigmoid (not tanh)
    2. W_hh contribution scaled by per-gate scalar a = [a_i, a_f, a_g, a_o]
       (scaling is applied to W_hh, b_hh once, outside the loop)

Gate ordering in the [4H] concat: [i, f, g, o]. Same as torch.nn.LSTM.
"""

import torch
import triton
import triton.language as tl


# =====================================================================
# Forward cell kernel
# =====================================================================
@triton.jit
def _fused_cell_fwd_kernel(
    gates_ptr,        # [B, 4H]  — already (W_ih @ x_t + b_ih) + a*(W_hh @ h_{t-1} + b_hh)
    c_prev_ptr,       # [B, H]
    h_out_ptr,        # [B, H]
    c_out_ptr,        # [B, H]
    saved_ptr,        # [B, 5H] — saves i, f, g, o, tanh_c per (batch, h)
    B, H,
    stride_gates_b, stride_gates_h,
    stride_state_b,  stride_state_h,
    stride_saved_b,  stride_saved_h,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # Load the 4 gate pre-activations for this (batch, h-tile).
    # gates layout: [B, 4H] where the H gate columns are concatenated [i, f, g, o].
    base = pid_b * stride_gates_b
    pre_i = tl.load(gates_ptr + base + (0 * H + h_offs) * stride_gates_h, mask=h_mask, other=0.0)
    pre_f = tl.load(gates_ptr + base + (1 * H + h_offs) * stride_gates_h, mask=h_mask, other=0.0)
    pre_g = tl.load(gates_ptr + base + (2 * H + h_offs) * stride_gates_h, mask=h_mask, other=0.0)
    pre_o = tl.load(gates_ptr + base + (3 * H + h_offs) * stride_gates_h, mask=h_mask, other=0.0)

    # All four gates use sigmoid (mod #1 vs nn.LSTM which has tanh on g).
    i = tl.sigmoid(pre_i.to(tl.float32))
    f = tl.sigmoid(pre_f.to(tl.float32))
    g = tl.sigmoid(pre_g.to(tl.float32))
    o = tl.sigmoid(pre_o.to(tl.float32))

    c_prev = tl.load(c_prev_ptr + pid_b * stride_state_b + h_offs * stride_state_h,
                     mask=h_mask, other=0.0).to(tl.float32)
    c_new = f * c_prev + i * g
    tanh_c = (tl.exp(2.0 * c_new) - 1.0) / (tl.exp(2.0 * c_new) + 1.0)  # stable tanh via exp
    h_new = o * tanh_c

    # Cast back to the output dtype on store (Triton infers from ptr dtype).
    tl.store(h_out_ptr + pid_b * stride_state_b + h_offs * stride_state_h, h_new, mask=h_mask)
    tl.store(c_out_ptr + pid_b * stride_state_b + h_offs * stride_state_h, c_new, mask=h_mask)

    # Save activations for backward: layout [B, 5H] = [i, f, g, o, tanh_c]
    s_base = pid_b * stride_saved_b
    tl.store(saved_ptr + s_base + (0 * H + h_offs) * stride_saved_h, i,      mask=h_mask)
    tl.store(saved_ptr + s_base + (1 * H + h_offs) * stride_saved_h, f,      mask=h_mask)
    tl.store(saved_ptr + s_base + (2 * H + h_offs) * stride_saved_h, g,      mask=h_mask)
    tl.store(saved_ptr + s_base + (3 * H + h_offs) * stride_saved_h, o,      mask=h_mask)
    tl.store(saved_ptr + s_base + (4 * H + h_offs) * stride_saved_h, tanh_c, mask=h_mask)


# =====================================================================
# Backward cell kernel
# =====================================================================
@triton.jit
def _fused_cell_bwd_kernel(
    dh_ptr, dc_ptr,                # incoming gradients [B, H] each
    c_prev_ptr,                    # [B, H]
    saved_ptr,                     # [B, 5H] = [i, f, g, o, tanh_c]
    dgates_ptr,                    # [B, 4H]  — output: gradient w.r.t. pre-activations
    dc_prev_ptr,                   # [B, H]   — output: gradient w.r.t. c_{t-1}
    B, H,
    stride_state_b, stride_state_h,
    stride_saved_b, stride_saved_h,
    stride_gates_b, stride_gates_h,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    s_base = pid_b * stride_saved_b
    i = tl.load(saved_ptr + s_base + (0 * H + h_offs) * stride_saved_h, mask=h_mask, other=0.0).to(tl.float32)
    f = tl.load(saved_ptr + s_base + (1 * H + h_offs) * stride_saved_h, mask=h_mask, other=0.0).to(tl.float32)
    g = tl.load(saved_ptr + s_base + (2 * H + h_offs) * stride_saved_h, mask=h_mask, other=0.0).to(tl.float32)
    o = tl.load(saved_ptr + s_base + (3 * H + h_offs) * stride_saved_h, mask=h_mask, other=0.0).to(tl.float32)
    tanh_c = tl.load(saved_ptr + s_base + (4 * H + h_offs) * stride_saved_h, mask=h_mask, other=0.0).to(tl.float32)

    c_prev = tl.load(c_prev_ptr + pid_b * stride_state_b + h_offs * stride_state_h,
                     mask=h_mask, other=0.0).to(tl.float32)
    dh = tl.load(dh_ptr + pid_b * stride_state_b + h_offs * stride_state_h,
                 mask=h_mask, other=0.0).to(tl.float32)
    dc_upstream = tl.load(dc_ptr + pid_b * stride_state_b + h_offs * stride_state_h,
                          mask=h_mask, other=0.0).to(tl.float32)

    # Backprop through cell:
    #   h = o * tanh(c)       => do = dh * tanh_c
    #                            dc_via_h = dh * o * (1 - tanh_c^2)
    #   c = f * c_prev + i * g
    do_act = dh * tanh_c
    dc_total = dc_upstream + dh * o * (1.0 - tanh_c * tanh_c)

    df_act = dc_total * c_prev
    dc_prev = dc_total * f
    di_act = dc_total * g
    dg_act = dc_total * i

    # Backprop through sigmoid: d_pre = d_act * act * (1 - act)
    di_pre = di_act * i * (1.0 - i)
    df_pre = df_act * f * (1.0 - f)
    dg_pre = dg_act * g * (1.0 - g)
    do_pre = do_act * o * (1.0 - o)

    # Write d_gates [B, 4H] in [i, f, g, o] order
    g_base = pid_b * stride_gates_b
    tl.store(dgates_ptr + g_base + (0 * H + h_offs) * stride_gates_h, di_pre, mask=h_mask)
    tl.store(dgates_ptr + g_base + (1 * H + h_offs) * stride_gates_h, df_pre, mask=h_mask)
    tl.store(dgates_ptr + g_base + (2 * H + h_offs) * stride_gates_h, dg_pre, mask=h_mask)
    tl.store(dgates_ptr + g_base + (3 * H + h_offs) * stride_gates_h, do_pre, mask=h_mask)
    tl.store(dc_prev_ptr + pid_b * stride_state_b + h_offs * stride_state_h, dc_prev, mask=h_mask)


# =====================================================================
# Python wrappers around the kernels
# =====================================================================
BLOCK_H = 128


def _launch_fwd(gates, c_prev, h_out, c_out, saved):
    B, H4 = gates.shape
    assert H4 % 4 == 0
    H = H4 // 4
    grid = (B, triton.cdiv(H, BLOCK_H))
    _fused_cell_fwd_kernel[grid](
        gates, c_prev, h_out, c_out, saved,
        B, H,
        gates.stride(0), gates.stride(1),
        c_prev.stride(0), c_prev.stride(1),
        saved.stride(0), saved.stride(1),
        BLOCK_H=BLOCK_H,
    )


def _launch_bwd(dh, dc, c_prev, saved, dgates, dc_prev):
    B, H = dh.shape
    grid = (B, triton.cdiv(H, BLOCK_H))
    _fused_cell_bwd_kernel[grid](
        dh, dc, c_prev, saved, dgates, dc_prev,
        B, H,
        dh.stride(0), dh.stride(1),
        saved.stride(0), saved.stride(1),
        dgates.stride(0), dgates.stride(1),
        BLOCK_H=BLOCK_H,
    )


# =====================================================================
# CUDA Graph capture of the forward time-loop
# =====================================================================
# Capturing the per-step matmul + cell-kernel sequence as a CUDA Graph
# replaces (3 * T) per-step Python/CUDA launches with a single graph replay.
# Biggest impact at small sizes where launch overhead dominates.
#
# Cache key: (T, B, D, H, dtype, device). For each new shape we allocate
# static buffers once, warm up, capture, then reuse across calls.

USE_FWD_GRAPH = True       # global on/off switch (off = same code path as before)
_FWD_GRAPH_CACHE = {}

# Optional persistent CUDA kernel (one launch for the entire T-step forward,
# state in shared memory across all timesteps). Implementation in
# _persistent.py — works correctly but in its naive form (no tensor cores,
# uncoalesced W_hh reads) is *slower* than the Triton + CUDA Graph path at the
# sizes we tested. Disabled by default. Kept as a reference for future work
# that adds tensor-core MMA + cooperative tile loads.
USE_PERSISTENT_FWD = False
_PERSISTENT_AVAILABLE = None  # lazy-init


def _persistent_available():
    global _PERSISTENT_AVAILABLE
    if _PERSISTENT_AVAILABLE is None:
        try:
            from ._persistent import persistent_forward  # noqa
            _PERSISTENT_AVAILABLE = True
        except Exception:
            _PERSISTENT_AVAILABLE = False
    return _PERSISTENT_AVAILABLE


def _build_or_get_fwd_graph(T, B, D, H, dtype, device):
    key = (T, B, D, H, dtype, str(device))
    cached = _FWD_GRAPH_CACHE.get(key)
    if cached is not None:
        return cached

    static = {
        'W_hh_scaled':  torch.empty(4 * H, H,        device=device, dtype=dtype),
        'b_hh_scaled':  torch.empty(4 * H,           device=device, dtype=dtype),
        'gates_ih_all': torch.empty(T, B, 4 * H,     device=device, dtype=dtype),
        'h_save':       torch.empty(T + 1, B, H,     device=device, dtype=dtype),
        'c_save':       torch.empty(T + 1, B, H,     device=device, dtype=dtype),
        'saved_acts':   torch.empty(T, B, 5 * H,     device=device, dtype=dtype),
        'gates_buf':    torch.empty(B, 4 * H,        device=device, dtype=dtype),
    }
    # Realistic noise so any kernel autotuning during warmup gets reasonable signal.
    for v in static.values():
        v.normal_(0, 0.01)
    static['h_save'][0].zero_()
    static['c_save'][0].zero_()

    W_hh_scaled_T = static['W_hh_scaled'].t()

    def run_loop():
        for t in range(T):
            torch.mm(static['h_save'][t], W_hh_scaled_T, out=static['gates_buf'])
            static['gates_buf'].add_(static['b_hh_scaled'])
            static['gates_buf'].add_(static['gates_ih_all'][t])
            _launch_fwd(static['gates_buf'], static['c_save'][t],
                        static['h_save'][t + 1], static['c_save'][t + 1],
                        static['saved_acts'][t])

    # Warmup on a side stream (required by torch.cuda.graph)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run_loop()
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_loop()

    static['graph'] = graph
    _FWD_GRAPH_CACHE[key] = static
    return static


# =====================================================================
# CUDA Graph capture of the backward time-loop
# =====================================================================
# The reverse-time loop has the same per-step structure as forward:
# elementwise add of dout, fused-bwd kernel, one mm for the recurrent dh.
# Captured into a graph and replayed. The batched-GEMM portion of backward
# (dW_hh, dW_ih, dx, etc.) stays outside the graph — those are already single
# kernel launches.
#
# Cache key uses the same (T, B, D, H, dtype, device) tuple as forward.

USE_BWD_GRAPH = True
_BWD_GRAPH_CACHE = {}


def _build_or_get_bwd_graph(T, B, D, H, dtype, device):
    key = (T, B, D, H, dtype, str(device))
    cached = _BWD_GRAPH_CACHE.get(key)
    if cached is not None:
        return cached

    static = {
        'dout':        torch.empty(T, B, H,        device=device, dtype=dtype),
        'c_save':      torch.empty(T + 1, B, H,    device=device, dtype=dtype),
        'saved_acts':  torch.empty(T, B, 5 * H,    device=device, dtype=dtype),
        'W_hh_scaled': torch.empty(4 * H, H,       device=device, dtype=dtype),
        'dgates_all':  torch.empty(T, B, 4 * H,    device=device, dtype=dtype),
        'dh_a':        torch.empty(B, H,           device=device, dtype=dtype),
        'dh_b':        torch.empty(B, H,           device=device, dtype=dtype),
        'dc_a':        torch.empty(B, H,           device=device, dtype=dtype),
        'dc_b':        torch.empty(B, H,           device=device, dtype=dtype),
    }
    for v in static.values():
        v.normal_(0, 0.01)

    def run_loop():
        # Local ping-pong refs. Graph capture records the alternating memory
        # accesses across the T iterations explicitly.
        dh_cur, dh_nxt = static['dh_a'], static['dh_b']
        dc_cur, dc_nxt = static['dc_a'], static['dc_b']
        for t in range(T - 1, -1, -1):
            dh_cur.add_(static['dout'][t])
            _launch_bwd(dh_cur, dc_cur, static['c_save'][t], static['saved_acts'][t],
                        static['dgates_all'][t], dc_nxt)
            torch.mm(static['dgates_all'][t], static['W_hh_scaled'], out=dh_nxt)
            dh_cur, dh_nxt = dh_nxt, dh_cur
            dc_cur, dc_nxt = dc_nxt, dc_cur

    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            # Reset dh_a / dc_a as if they were the upstream gradients.
            static['dh_a'].normal_(0, 0.01)
            static['dc_a'].normal_(0, 0.01)
            run_loop()
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_loop()

    static['graph'] = graph
    _BWD_GRAPH_CACHE[key] = static
    return static


# =====================================================================
# Autograd function: full forward + backward, hand-written
# =====================================================================
class PRLSTMTritonFn(torch.autograd.Function):
    """Custom autograd. forward returns (output [T,B,H], h_n [B,H], c_n [B,H])."""

    @staticmethod
    def forward(ctx, x, h0, c0, W_ih, W_hh, b_ih, b_hh, a):
        T, B, D = x.shape
        H = h0.shape[-1]
        dtype = x.dtype
        device = x.device

        # Pre-scale W_hh and b_hh by the per-gate a (a_expanded broadcast over [4H]).
        a_expanded = a.repeat_interleave(H)                    # [4H]
        W_hh_scaled = a_expanded.unsqueeze(1) * W_hh           # [4H, H]
        b_hh_scaled = a_expanded * b_hh                        # [4H]

        # Batched input GEMM: gates_ih_all[t, b] = W_ih @ x[t, b] + b_ih.
        # One cuBLAS call over all T*B, not T separate calls.
        gates_ih_all = torch.nn.functional.linear(x, W_ih, b_ih)   # [T, B, 4H]

        use_persistent = (
            USE_PERSISTENT_FWD
            and device.type == "cuda"
            and dtype == torch.float32
            and 4 * H <= 1024
            and _persistent_available()
        )

        if use_persistent:
            # Hand-written persistent CUDA kernel (one launch for the
            # entire T-step forward, state lives in shared memory).
            from ._persistent import persistent_forward
            h_save, c_save, saved_acts = persistent_forward(
                gates_ih_all.contiguous(),
                W_hh_scaled.contiguous(),
                b_hh_scaled.contiguous(),
                h0.contiguous(),
                c0.contiguous(),
            )
        elif USE_FWD_GRAPH and device.type == "cuda":
            # CUDA Graph path: get cached graph + static buffers, copy inputs in,
            # replay, copy outputs out for backward.
            static = _build_or_get_fwd_graph(T, B, D, H, dtype, device)
            static['W_hh_scaled'].copy_(W_hh_scaled)
            static['b_hh_scaled'].copy_(b_hh_scaled)
            static['gates_ih_all'].copy_(gates_ih_all)
            static['h_save'][0].copy_(h0)
            static['c_save'][0].copy_(c0)
            static['graph'].replay()
            # Copy outputs out — static buffers get clobbered by next call.
            h_save     = static['h_save'].clone()
            c_save     = static['c_save'].clone()
            saved_acts = static['saved_acts'].clone()
        else:
            # Eager path (no CUDA graph).
            h_save = torch.empty(T + 1, B, H, device=device, dtype=dtype)
            c_save = torch.empty(T + 1, B, H, device=device, dtype=dtype)
            saved_acts = torch.empty(T, B, 5 * H, device=device, dtype=dtype)
            gates_buf = torch.empty(B, 4 * H, device=device, dtype=dtype)
            h_save[0].copy_(h0)
            c_save[0].copy_(c0)
            W_hh_scaled_T = W_hh_scaled.t()
            for t in range(T):
                torch.mm(h_save[t], W_hh_scaled_T, out=gates_buf)
                gates_buf.add_(b_hh_scaled)
                gates_buf.add_(gates_ih_all[t])
                _launch_fwd(gates_buf, c_save[t], h_save[t + 1], c_save[t + 1], saved_acts[t])

        output = h_save[1:]      # view, [T, B, H]
        h_n = h_save[T]          # view
        c_n = c_save[T]          # view

        ctx.save_for_backward(x, h_save, c_save, saved_acts,
                              W_ih, W_hh, b_ih, b_hh, a,
                              W_hh_scaled, b_hh_scaled, a_expanded)
        ctx.T, ctx.B, ctx.D, ctx.H = T, B, D, H
        return output, h_n, c_n

    @staticmethod
    def backward(ctx, dout, dh_n, dc_n):
        (x, h_save, c_save, saved_acts,
         W_ih, W_hh, b_ih, b_hh, a,
         W_hh_scaled, b_hh_scaled, a_expanded) = ctx.saved_tensors
        T, B, D, H = ctx.T, ctx.B, ctx.D, ctx.H
        device = x.device
        dtype = x.dtype

        if USE_BWD_GRAPH and device.type == "cuda":
            # CUDA Graph backward: cached graph + static buffers.
            static = _build_or_get_bwd_graph(T, B, D, H, dtype, device)
            static['dout'].copy_(dout)
            static['c_save'].copy_(c_save)
            static['saved_acts'].copy_(saved_acts)
            static['W_hh_scaled'].copy_(W_hh_scaled)
            static['dh_a'].copy_(dh_n.contiguous())
            static['dc_a'].copy_(dc_n.contiguous())
            static['graph'].replay()
            # The next backward replay will clobber dgates_all; we consume it
            # immediately below in the batched GEMMs so cloning isn't required.
            dgates_all = static['dgates_all']
        else:
            # Eager path.
            dgates_all = torch.empty(T, B, 4 * H, device=device, dtype=dtype)
            dh_a = dh_n.contiguous().clone()
            dh_b = torch.empty(B, H, device=device, dtype=dtype)
            dc_a = dc_n.contiguous().clone()
            dc_b = torch.empty(B, H, device=device, dtype=dtype)
            dh_cur, dh_nxt = dh_a, dh_b
            dc_cur, dc_nxt = dc_a, dc_b
            for t in range(T - 1, -1, -1):
                dh_cur.add_(dout[t])
                _launch_bwd(dh_cur, dc_cur, c_save[t], saved_acts[t],
                            dgates_all[t], dc_nxt)
                torch.mm(dgates_all[t], W_hh_scaled, out=dh_nxt)
                dh_cur, dh_nxt = dh_nxt, dh_cur
                dc_cur, dc_nxt = dc_nxt, dc_cur

        # Batched GEMMs for the parameter and input gradients.
        #   dgates_all flat: [T*B, 4H]
        #   h_save[:-1]   :  [T, B, H]  -> [T*B, H]
        #   x             :  [T, B, D]  -> [T*B, D]
        dgates_flat = dgates_all.view(T * B, 4 * H)
        h_prev_flat = h_save[:-1].reshape(T * B, H)
        x_flat      = x.view(T * B, D)

        dW_hh_scaled = dgates_flat.t() @ h_prev_flat   # [4H, H]
        dW_ih        = dgates_flat.t() @ x_flat        # [4H, D]
        db_sum       = dgates_flat.sum(dim=0)          # [4H]  — same for b_ih and b_hh_scaled
        db_hh_scaled = db_sum
        db_ih        = db_sum
        dx_all       = (dgates_flat @ W_ih).view(T, B, D)

        # Backprop the a-scaling: W_hh_scaled = a_expanded[:,None] * W_hh,
        #                         b_hh_scaled = a_expanded * b_hh
        dW_hh = a_expanded.unsqueeze(1) * dW_hh_scaled
        db_hh = a_expanded * db_hh_scaled
        da_expanded = (W_hh * dW_hh_scaled).sum(dim=1) + b_hh * db_hh_scaled  # [4H]
        da = da_expanded.view(4, H).sum(dim=1)                                 # [4]

        return dx_all, None, None, dW_ih, dW_hh, db_ih, db_hh, da


def prlstm_triton(x, h0, c0, W_ih, W_hh, b_ih, b_hh, a):
    """Functional wrapper. Returns (output, h_n, c_n)."""
    return PRLSTMTritonFn.apply(x, h0, c0, W_ih, W_hh, b_ih, b_hh, a)


# =====================================================================
# Parallel-mode forward (Heinsen log-domain scan)
# =====================================================================
# The recurrence c_t = f_t * c_{t-1} + b_t is a diagonal linear recurrence in
# c with positive coefficients (since f and b = i*g are products of sigmoids,
# both in (0,1)). That structure permits an O(log T) parallel prefix scan via
# log-domain accumulation [Heinsen 2023].
#
# Parallel mode *drops the W_hh h_{t-1} + b_hh recurrent contribution*; the
# gate pre-activations come from x only. This makes the gates independent of
# h_{t-1}, which is what unlocks the parallel scan.
#
# Consequence: parallel mode is *exactly equivalent* to recurrent mode when
# a = 0 (since recurrent then also has zero W_hh contribution). As a grows
# away from zero, parallel and recurrent diverge — parallel becomes an
# approximation. This is the head-start training trick: cheap parallel mode
# while a is small, switch to recurrent mode when needed.


def _log_parallel_scan(f, c0, b, eps: float = 1e-6):
    """Heinsen log-domain prefix scan for c_t = f_t * c_{t-1} + b_t.

    All inputs are [B, T, H] (with T along dim=1) and must be positive.
    Returns c [B, T, H].
    """
    log_f  = torch.log(torch.clamp(f,  min=eps))
    log_b  = torch.log(torch.clamp(b,  min=eps))
    log_c0 = torch.log(torch.clamp(c0, min=eps))

    a_star = torch.cumsum(log_f, dim=1)                      # log(prod_{s<=t} f_s)
    log_b_minus_a_star = log_b - a_star
    cats = torch.cat([log_c0, log_b_minus_a_star], dim=1)    # [B, T+1, H]
    tail_lcse = torch.logcumsumexp(cats, dim=1)[:, 1:]       # [B, T, H]
    return torch.exp(a_star + tail_lcse)                     # [B, T, H]


def prlstm_triton_parallel(x, h0, c0, W_ih, W_hh, b_ih, b_hh, a, eps: float = 1e-6):
    """Parallel-mode forward via Heinsen scan. Autograd is traced through
    PyTorch ops — no custom backward.

    NOTE: W_hh, b_hh, a, h0 are accepted for API compatibility but are NOT
    used. Parallel mode drops the W_hh contribution. At a=0 this matches the
    recurrent mode exactly; as a grows it becomes an approximation.

    Inputs (same shapes as recurrent path):
        x:    [T, B, D]
        c0:   [B, H]
        W_ih: [4H, D],  b_ih: [4H]
    Returns:
        output [T, B, H], h_n [B, H], c_n [B, H]
    """
    T, B, D = x.shape
    H = c0.shape[-1]

    # Single batched cuBLAS GEMM for input contribution
    gates = torch.nn.functional.linear(x, W_ih, b_ih)        # [T, B, 4H]
    i_pre, f_pre, g_pre, o_pre = gates.chunk(4, dim=-1)

    i_act = torch.sigmoid(i_pre)
    f_act = torch.sigmoid(f_pre)
    g_act = torch.sigmoid(g_pre)                             # sigmoid g (mod #1)
    o_act = torch.sigmoid(o_pre)
    b_act = i_act * g_act                                    # in (0,1), positive

    # Permute (T, B, H) -> (B, T, H) for the scan (which sums along dim=1)
    f_btH = f_act.transpose(0, 1).contiguous()
    b_btH = b_act.transpose(0, 1).contiguous()
    c0_btH = c0.unsqueeze(1)                                 # [B, 1, H]

    c_btH = _log_parallel_scan(f_btH, c0_btH, b_btH, eps=eps)   # [B, T, H]
    c_seq = c_btH.transpose(0, 1).contiguous()                  # [T, B, H]

    output = o_act * torch.tanh(c_seq)                       # [T, B, H]
    return output, output[-1].contiguous(), c_seq[-1].contiguous()

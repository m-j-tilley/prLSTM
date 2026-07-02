"""Portable recurrent-mode HeadStartLSTM forward, pure-PyTorch ops only.

Runs on any backend (CUDA, MPS, CPU). The structure mirrors the Triton
backend but uses ordinary PyTorch ops everywhere:

  * Single batched input GEMM `linear(x, W_ih, b_ih)` for all T steps
  * Pre-scales W_hh, b_hh by per-gate `a` once outside the loop
  * Custom torch.autograd.Function with a hand-written backward — no
    autograd graph traced through the T-step loop, and all parameter
    gradients (dW_hh, dW_ih, dx, db) are computed via three batched GEMMs
    after the reverse-time loop, not T separate matmuls inside autograd.

The cell-step elementwise chain (4 sigmoids, mul/add for c, tanh, mul for h)
is factored into `_cell_forward_step` / `_cell_backward_step` so it can be
wrapped with `torch.compile` to fuse the ops into a single device kernel per
step. That fusion is opt-in via `HeadStartLSTM(..., compile_cell=True)`.
"""

import torch


# ---------------------------------------------------------------------------
# Per-step cell ops. Factored out so torch.compile can fuse them.
# Forward returns (h_new, c_new, saved) where `saved` packs (i, f, g, o,
# tanh_c) for the backward.
# ---------------------------------------------------------------------------
def _cell_forward_step(gates, c_prev):
    i_, f_, g_, o_ = gates.chunk(4, dim=1)
    i_ = torch.sigmoid(i_)
    f_ = torch.sigmoid(f_)
    g_ = torch.sigmoid(g_)
    o_ = torch.sigmoid(o_)
    c_new = f_ * c_prev + i_ * g_
    tanh_c = torch.tanh(c_new)
    h_new = o_ * tanh_c
    return h_new, c_new, i_, f_, g_, o_, tanh_c


def _cell_backward_step(dh, dc_upstream, c_prev, i_, f_, g_, o_, tanh_c):
    """Returns (dgates, dc_prev) for one cell step."""
    do_act = dh * tanh_c
    dc_total = dc_upstream + dh * o_ * (1.0 - tanh_c * tanh_c)
    df_act = dc_total * c_prev
    dc_prev = dc_total * f_
    di_act = dc_total * g_
    dg_act = dc_total * i_
    # sigmoid: d_pre = d_act * act * (1 - act)
    di_pre = di_act * i_ * (1.0 - i_)
    df_pre = df_act * f_ * (1.0 - f_)
    dg_pre = dg_act * g_ * (1.0 - g_)
    do_pre = do_act * o_ * (1.0 - o_)
    dgates = torch.cat([di_pre, df_pre, dg_pre, do_pre], dim=1)
    return dgates, dc_prev


# Lazily-compiled variants. We swap the module-level binding rather than
# wrapping inside the autograd Function so torch.compile gets a stable
# callable to cache against.
_cell_fwd_fn = _cell_forward_step
_cell_bwd_fn = _cell_backward_step
_compile_attempted = False


def _maybe_compile_cell():
    global _cell_fwd_fn, _cell_bwd_fn, _compile_attempted
    if _compile_attempted:
        return
    _compile_attempted = True
    try:
        _cell_fwd_fn = torch.compile(_cell_forward_step, dynamic=False)
        _cell_bwd_fn = torch.compile(_cell_backward_step, dynamic=False)
    except Exception:
        # If compile is unavailable / errors at trace time, keep eager.
        _cell_fwd_fn = _cell_forward_step
        _cell_bwd_fn = _cell_backward_step


class HeadStartLSTMTorchFn(torch.autograd.Function):
    """Custom autograd for the recurrent HeadStartLSTM forward (pure PyTorch).

    Forward returns (output [T, B, H], h_n [B, H], c_n [B, H]). Backward
    walks the reverse-time loop manually and finalises with batched GEMMs
    for the parameter gradients.
    """

    @staticmethod
    def forward(ctx, x, h0, c0, W_ih, W_hh, b_ih, b_hh, a):
        T, B, _ = x.shape
        H = h0.shape[-1]

        a_expanded  = a.repeat_interleave(H)                # [4H]
        W_hh_scaled = a_expanded.unsqueeze(1) * W_hh        # [4H, H]
        b_hh_scaled = a_expanded * b_hh                     # [4H]
        W_hh_T      = W_hh_scaled.t().contiguous()          # [H, 4H]

        gates_ih_all = torch.nn.functional.linear(x, W_ih, b_ih)  # [T, B, 4H]

        # Stacks to keep saved activations contiguous in time.
        h_save     = x.new_zeros(T + 1, B, H)
        c_save     = x.new_zeros(T + 1, B, H)
        i_save     = x.new_zeros(T, B, H)
        f_save     = x.new_zeros(T, B, H)
        g_save     = x.new_zeros(T, B, H)
        o_save     = x.new_zeros(T, B, H)
        tanh_c_save = x.new_zeros(T, B, H)

        h_save[0].copy_(h0)
        c_save[0].copy_(c0)

        h, c = h0, c0
        for t in range(T):
            gates = gates_ih_all[t] + h @ W_hh_T + b_hh_scaled
            h, c, i_, f_, g_, o_, tanh_c = _cell_fwd_fn(gates, c)
            h_save[t + 1] = h
            c_save[t + 1] = c
            i_save[t] = i_
            f_save[t] = f_
            g_save[t] = g_
            o_save[t] = o_
            tanh_c_save[t] = tanh_c

        output = h_save[1:]
        h_n = h_save[T]
        c_n = c_save[T]

        ctx.save_for_backward(
            x, h_save, c_save, i_save, f_save, g_save, o_save, tanh_c_save,
            W_ih, W_hh, b_ih, b_hh, a, W_hh_scaled, a_expanded,
        )
        ctx.T, ctx.B, ctx.H = T, B, H
        return output, h_n, c_n

    @staticmethod
    def backward(ctx, dout, dh_n, dc_n):
        (x, h_save, c_save, i_save, f_save, g_save, o_save, tanh_c_save,
         W_ih, W_hh, b_ih, b_hh, a, W_hh_scaled, a_expanded) = ctx.saved_tensors
        T, B, H = ctx.T, ctx.B, ctx.H
        D = x.shape[-1]

        dgates_all = x.new_zeros(T, B, 4 * H)
        dh_cur = dh_n.contiguous().clone()
        dc_cur = dc_n.contiguous().clone()
        for t in range(T - 1, -1, -1):
            dh_cur = dh_cur + dout[t]
            dgates, dc_cur = _cell_bwd_fn(
                dh_cur, dc_cur, c_save[t],
                i_save[t], f_save[t], g_save[t], o_save[t], tanh_c_save[t],
            )
            dgates_all[t] = dgates
            dh_cur = dgates @ W_hh_scaled    # [B, H]

        # Batched parameter GEMMs.
        dgates_flat = dgates_all.view(T * B, 4 * H)
        h_prev_flat = h_save[:-1].reshape(T * B, H)
        x_flat      = x.view(T * B, D)

        dW_hh_scaled = dgates_flat.t() @ h_prev_flat   # [4H, H]
        dW_ih        = dgates_flat.t() @ x_flat        # [4H, D]
        db_sum       = dgates_flat.sum(dim=0)          # [4H]
        db_hh_scaled = db_sum
        db_ih        = db_sum
        dx_all       = (dgates_flat @ W_ih).view(T, B, D)

        # Unpack the a-scaling.
        dW_hh = a_expanded.unsqueeze(1) * dW_hh_scaled
        db_hh = a_expanded * db_hh_scaled
        da_expanded = (W_hh * dW_hh_scaled).sum(dim=1) + b_hh * db_hh_scaled
        da = da_expanded.view(4, H).sum(dim=1)

        return dx_all, None, None, dW_ih, dW_hh, db_ih, db_hh, da


def headstartlstm_torch(x, h0, c0, W_ih, W_hh, b_ih, b_hh, a, compile_cell: bool = False):
    if compile_cell:
        _maybe_compile_cell()
    return HeadStartLSTMTorchFn.apply(x, h0, c0, W_ih, W_hh, b_ih, b_hh, a)

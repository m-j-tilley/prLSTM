"""Metal (Apple Silicon) backend for HeadStartLSTM.

Loads `_csrc/lstm_metal.mm` lazily via torch.utils.cpp_extension on first
use. The .mm file compiles two fused Metal compute kernels — per-step cell
forward and per-step cell backward — that replace the ~10 elementwise
PyTorch ops per step with one Metal launch each. The matmul work
(recurrent W_hh, input W_ih, parameter-gradient GEMMs) stays in PyTorch
because MPSGraph's tiled matmul beats anything we'd write in pure Metal at
typical hidden sizes.

Public API:
    available()        — bool; True if the extension built/loaded.
    headstartlstm_metal(...)  — autograd-wrapped forward+backward.

Constraints inherited from the kernels:
    * fp32 only (Metal doesn't expose fp64).
    * Device must be MPS.

The caller (HeadStartLSTM.forward) checks these and falls back to the torch
backend when they don't hold.
"""

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ext = None
_load_failed_reason = None


def _try_load():
    """Lazy build/load. Returns the extension module or None on failure.

    Only meaningful on macOS — `__init__.py` guards by `sys.platform`.
    """
    global _ext, _load_failed_reason
    if _ext is not None:
        return _ext
    if _load_failed_reason is not None:
        return None
    try:
        from torch.utils.cpp_extension import load
        _ext = load(
            name="headstartlstm_metal",
            sources=[str(_HERE / "_csrc" / "lstm_metal.mm")],
            extra_cflags=[
                "-O3", "-std=c++17",
                "-Wno-unused-variable", "-Wno-deprecated-declarations",
            ],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            verbose=False,
        )
    except Exception as e:
        _load_failed_reason = f"{type(e).__name__}: {e}"
        return None
    return _ext


def available() -> bool:
    """True if Metal backend can be used on this machine."""
    if sys.platform != "darwin":
        return False
    if not torch.backends.mps.is_available():
        return False
    return _try_load() is not None


class HeadStartLSTMMetalFn(torch.autograd.Function):
    """Custom autograd. forward returns (output [T, B, H], h_n, c_n).

    Per-step structure (same as the Triton/CUDA path, scaled to MPS):
      forward:  mm_result = h @ W_hh_scaled.T   (1 MPS matmul launch)
                h, c, saved = cell_fwd(...)     (1 Metal launch)
      backward: dgates, dc = cell_bwd(...)      (1 Metal launch)
                dh_prev = dgates @ W_hh_scaled  (1 MPS matmul launch)
    Two launches per step in each direction — down from ~10 for the eager
    PyTorch path, and matched against `nn.LSTM`'s 1 fused MPSGraph launch
    for the entire T-step recurrence.
    """

    @staticmethod
    def forward(ctx, x, h0, c0, W_ih, W_hh, b_ih, b_hh, a):
        ext = _try_load()
        if ext is None:
            raise RuntimeError(f"headstartlstm metal extension failed to load: {_load_failed_reason}")

        T, B, D = x.shape
        H = h0.shape[-1]

        a_expanded  = a.repeat_interleave(H)                              # [4H]
        W_hh_scaled = (a_expanded.unsqueeze(1) * W_hh).contiguous()       # [4H, H]
        b_hh_scaled = (a_expanded * b_hh).contiguous()                    # [4H]
        W_hh_T      = W_hh_scaled.t().contiguous()                        # [H, 4H]

        # Single batched input GEMM (one MPSGraph matmul over T*B rows).
        gates_ih_all = torch.nn.functional.linear(x, W_ih, b_ih).contiguous()  # [T, B, 4H]

        # Pre-allocate stacks; the per-step kernel writes into them in-place.
        opts = {"device": x.device, "dtype": x.dtype}
        h_save     = torch.empty((T + 1, B, H),     **opts)
        c_save     = torch.empty((T + 1, B, H),     **opts)
        saved_acts = torch.empty((T,     B, 5 * H), **opts)
        h_save[0].copy_(h0)
        c_save[0].copy_(c0)

        h, c = h_save[0], c_save[0]
        for t in range(T):
            mm_res = h @ W_hh_T                          # [B, 4H]  (MPS matmul)
            h_new, c_new, saved = ext.headstartlstm_cell_fwd(
                mm_res.contiguous(),
                gates_ih_all[t].contiguous(),
                b_hh_scaled, c.contiguous(),
            )
            h_save[t + 1] = h_new
            c_save[t + 1] = c_new
            saved_acts[t] = saved
            h, c = h_new, c_new

        output = h_save[1:]
        h_n = h_save[T]
        c_n = c_save[T]

        ctx.save_for_backward(
            x, h_save, c_save, saved_acts,
            W_ih, W_hh, b_ih, b_hh, a, W_hh_scaled, a_expanded,
        )
        ctx.T, ctx.B, ctx.H = T, B, H
        return output, h_n, c_n

    @staticmethod
    def backward(ctx, dout, dh_n, dc_n):
        ext = _try_load()
        (x, h_save, c_save, saved_acts,
         W_ih, W_hh, b_ih, b_hh, a, W_hh_scaled, a_expanded) = ctx.saved_tensors
        T, B, H = ctx.T, ctx.B, ctx.H
        D = x.shape[-1]

        opts = {"device": x.device, "dtype": x.dtype}
        dgates_all = torch.empty((T, B, 4 * H), **opts)

        dh_cur = dh_n.contiguous().clone()
        dc_cur = dc_n.contiguous().clone()
        for t in range(T - 1, -1, -1):
            dh_cur = dh_cur + dout[t]
            dgates, dc_cur = ext.headstartlstm_cell_bwd(
                dh_cur.contiguous(), dc_cur.contiguous(),
                c_save[t].contiguous(), saved_acts[t].contiguous(),
            )
            dgates_all[t] = dgates
            dh_cur = dgates @ W_hh_scaled        # [B, H]

        dh0 = dh_cur
        dc0 = dc_cur

        # Batched parameter GEMMs (each one MPSGraph matmul launch).
        dgates_flat = dgates_all.view(T * B, 4 * H)
        h_prev_flat = h_save[:-1].reshape(T * B, H)
        x_flat      = x.view(T * B, D)

        dW_hh_scaled = dgates_flat.t() @ h_prev_flat
        dW_ih        = dgates_flat.t() @ x_flat
        db_sum       = dgates_flat.sum(dim=0)
        db_hh_scaled = db_sum
        db_ih        = db_sum
        dx_all       = (dgates_flat @ W_ih).view(T, B, D)

        dW_hh = a_expanded.unsqueeze(1) * dW_hh_scaled
        db_hh = a_expanded * db_hh_scaled
        da_expanded = (W_hh * dW_hh_scaled).sum(dim=1) + b_hh * db_hh_scaled
        da = da_expanded.view(4, H).sum(dim=1)

        return dx_all, dh0, dc0, dW_ih, dW_hh, db_ih, db_hh, da


def headstartlstm_metal(x, h0, c0, W_ih, W_hh, b_ih, b_hh, a):
    return HeadStartLSTMMetalFn.apply(x, h0, c0, W_ih, W_hh, b_ih, b_hh, a)

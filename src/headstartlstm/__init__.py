"""HeadStartLSTM: an LSTM whose recurrence can be trained in parallel first, then annealed in.

HeadStartLSTM is a single-layer, unidirectional LSTM modified two ways from
``torch.nn.LSTM``:

  1. ``g_t`` uses ``sigmoid`` (not ``tanh``).
  2. The ``W_hh h_{t-1} + b_hh`` recurrent contribution is multiplied by a
     learnable per-gate scalar ``a = [a_i, a_f, a_g, a_o]`` (init ``a_init``).

These two modifications turn the cell into a recurrence with positive
coefficients that admits an *exact parallel scan* (Heinsen log-domain) when
``a = 0``, and degrades smoothly into a standard nonlinear LSTM as ``a`` grows.

Backends
--------
``backend="auto"`` (default) — picks per-device:
    * CUDA  -> ``"triton"`` (fused Triton cell + cuBLAS GEMMs + CUDA-Graph)
    * MPS   -> ``"metal"`` (fused Metal compute shaders for the per-step
      cell forward/backward; MPSGraph handles the matmuls — two MPS launches
      per step). Falls back to ``"torch"`` if the Metal extension can't build
      (e.g. Xcode Command Line Tools missing).
    * CPU / anywhere else -> ``"torch"`` (portable pure-PyTorch loop with
      a single batched input GEMM up-front)
``backend="cpp"``    — ATen-ops C++ baseline, JIT-compiled on first use.
``backend="torch"``  — portable pure-PyTorch path; works on CUDA/MPS/CPU.
``backend="triton"`` — CUDA-only fast path; raises if CUDA / triton missing.
``backend="metal"``  — Apple-Silicon fast path; raises if MPS unavailable
                       or the .mm extension fails to build.

``parallel=True`` enables the Heinsen log-domain parallel scan forward
(O(log T) instead of O(T)). It drops the ``a · W_hh h`` term, so it is
*mathematically identical to recurrent forward at a = 0* and an
approximation as a grows. Pure PyTorch — runs on any device.

Quick start::

    import torch
    from headstartlstm import HeadStartLSTM

    m = HeadStartLSTM(input_size=128, hidden_size=256).to("mps")
    x = torch.randn(64, 8, 128, device="mps")     # [T, B, D]
    output, (h_n, c_n) = m(x)

    m.set_mode(parallel=True)
    output_par, (h_n_par, c_n_par) = m(x)         # ~equiv at a=0, fast scan
"""

import math
from pathlib import Path

import torch
import torch.nn as nn

__version__ = "0.1.0"

_HERE = Path(__file__).resolve().parent

# The C++ baseline is JIT-compiled on first use (needs ninja + a C++ compiler).
# Lazy so simply importing headstartlstm doesn't require a toolchain.
_cpp_ext = None


def _load_cpp_ext():
    global _cpp_ext
    if _cpp_ext is None:
        from torch.utils.cpp_extension import load
        _cpp_ext = load(
            name="headstartlstm_cpp",
            sources=[str(_HERE / "_csrc" / "lstm_cell.cpp")],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _cpp_ext


def _resolve_backend(backend: str, device_type: str, hidden_size: int) -> str:
    """Resolve "auto" to a concrete backend based on the device + hidden size."""
    if backend != "auto":
        return backend
    if device_type == "cuda":
        try:
            import triton  # noqa: F401
            return "triton"
        except ImportError:
            return "torch"
    if device_type == "mps":
        # Lazy probe — never imports _metal on non-Darwin systems.
        import sys
        if sys.platform == "darwin":
            try:
                from ._metal import available as _metal_available
                if _metal_available():
                    return "metal"
            except Exception:
                pass
    return "torch"


class HeadStartLSTM(nn.Module):
    """An LSTM with a parallel-scan mode that warm-starts BPTT training.

    Parameters
    ----------
    input_size : int
        Number of expected features in the input ``x``.
    hidden_size : int
        Number of features in the hidden state ``h``.
    batch_first : bool, default False
        If True the input is ``[B, T, D]`` instead of ``[T, B, D]``.
    a_init : float, default 1e-4
        Initial value for the learnable per-gate scaling ``a``.
    backend : {"auto", "torch", "cpp", "triton", "metal"}, default "auto"
        Compute backend for the recurrent forward. "auto" picks "triton" on
        CUDA, "metal" on Apple Silicon (MPS), and "torch" everywhere else
        (falling back to "torch" when the faster backend is unavailable).
    parallel : bool, default False
        If True, use the Heinsen log-domain parallel scan forward (drops the
        ``a · W_hh h`` recurrent term). Equivalent to ``parallel=False`` at
        ``a = 0``; an approximation as ``a`` grows. Pure-PyTorch, runs on any
        device.
    compile_cell : bool, default False
        If True (and ``backend`` resolves to ``"torch"``), wrap the per-step
        cell ops with ``torch.compile`` so the elementwise chain fuses into
        a single device kernel per step. Only affects the ``"torch"`` backend
        (no effect on triton/metal/cpp); useful when forcing ``backend="torch"``
        on MPS to close most of the gap to ``torch.nn.LSTM``.
    """

    _VALID_BACKENDS = ("auto", "torch", "cpp", "triton", "metal")

    def __init__(self, input_size: int, hidden_size: int,
                 batch_first: bool = False, a_init: float = 1e-4,
                 backend: str = "auto", parallel: bool = False,
                 compile_cell: bool = False):
        super().__init__()
        assert backend in self._VALID_BACKENDS, (
            f"backend must be one of {self._VALID_BACKENDS}, got {backend!r}"
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.a_init = a_init
        self.backend = backend
        self.parallel = parallel
        self.compile_cell = compile_cell

        # Match nn.LSTM's parameter names and layout (gate order [i, f, g, o]).
        self.weight_ih_l0 = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih_l0   = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh_l0   = nn.Parameter(torch.zeros(4 * hidden_size))
        self.a            = nn.Parameter(torch.full((4,), float(a_init)))

        self.reset_parameters()

    def set_mode(self, parallel: bool):
        """Switch between recurrent (default) and parallel-scan mode at runtime."""
        self.parallel = parallel

    def reset_parameters(self):
        # nn.LSTM init: uniform(-1/sqrt(H), 1/sqrt(H)) on every weight & bias.
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in (self.weight_ih_l0, self.weight_hh_l0,
                  self.bias_ih_l0,   self.bias_hh_l0):
            nn.init.uniform_(p, -stdv, stdv)
        with torch.no_grad():
            self.a.fill_(self.a_init)

    def forward(self, x: torch.Tensor, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1).contiguous()
        T, B, _ = x.shape

        if hx is None:
            h0 = x.new_zeros(B, self.hidden_size)
            c0 = x.new_zeros(B, self.hidden_size)
        else:
            h0, c0 = hx
            if h0.dim() == 3:
                h0 = h0.squeeze(0)
            if c0.dim() == 3:
                c0 = c0.squeeze(0)

        if self.parallel:
            from ._parallel import headstartlstm_parallel
            output, h_n, c_n = headstartlstm_parallel(
                x, h0, c0,
                self.weight_ih_l0, self.weight_hh_l0,
                self.bias_ih_l0,   self.bias_hh_l0,
                self.a,
            )
        else:
            backend = _resolve_backend(self.backend, x.device.type, self.hidden_size)
            if backend == "triton":
                from ._triton import headstartlstm_triton
                output, h_n, c_n = headstartlstm_triton(
                    x, h0, c0,
                    self.weight_ih_l0, self.weight_hh_l0,
                    self.bias_ih_l0,   self.bias_hh_l0,
                    self.a,
                )
            elif backend == "metal":
                from ._metal import headstartlstm_metal
                output, h_n, c_n = headstartlstm_metal(
                    x, h0, c0,
                    self.weight_ih_l0, self.weight_hh_l0,
                    self.bias_ih_l0,   self.bias_hh_l0,
                    self.a,
                )
            elif backend == "torch":
                from ._torch import headstartlstm_torch
                output, h_n, c_n = headstartlstm_torch(
                    x, h0, c0,
                    self.weight_ih_l0, self.weight_hh_l0,
                    self.bias_ih_l0,   self.bias_hh_l0,
                    self.a,
                    compile_cell=self.compile_cell,
                )
            elif backend == "cpp":
                ext = _load_cpp_ext()
                output, h_n, c_n = ext.headstartlstm_forward(
                    x, h0, c0,
                    self.weight_ih_l0, self.weight_hh_l0,
                    self.bias_ih_l0,   self.bias_hh_l0,
                    self.a,
                )
            else:
                raise RuntimeError(f"unknown backend {backend!r}")

        if self.batch_first:
            output = output.transpose(0, 1).contiguous()

        return output, (h_n.unsqueeze(0), c_n.unsqueeze(0))


__all__ = ["HeadStartLSTM", "__version__"]

# PRLSTM — Parallelisable-Recurrent LSTM

Single-layer LSTM with three modifications to `torch.nn.LSTM`:

1. `g_t` uses sigmoid instead of tanh (needed for 3).
2. The recurrent contribution `W_hh h + b_hh` is scaled by a learnable per-gate scalar `a` (init `1e-4`).
3. Parallel-scan implementation using method from https://arxiv.org/abs/2311.06281 by F. Heinsen

At init the parallel and recurrent versions are ~ the same.

With the motivation that you can start off training in parallel, add back the nonlinearity through the prior state dependence, then train with BPTT. (TBD if that's at all useful).


## Install

```bash
git clone https://github.com/m-j-tilley/prLSTM
cd prLSTM
pip install -e .
```

Requires Python ≥ 3.10, PyTorch ≥ 2.1. Runs on CUDA, Apple Silicon (MPS),
and CPU. Platform-specific backends (Triton, Metal, C++) compile lazily
on first use, so `pip install` works on any platform without needing the
other platform's toolchain.

## Backends

`backend="auto"` (default) picks per-device:

  * **CUDA** → `triton` — fused Triton cell + cuBLAS GEMMs + CUDA-Graph
    capture. Hand-written backward. Matches or beats cuDNN at scale.
  * **MPS (Apple Silicon)** → `metal` — fused Metal compute shaders for
    the per-step cell forward and backward; MPSGraph handles the
    matmuls. Hand-written custom autograd. Matches `torch.nn.LSTM` at
    production training shapes.
  * **CPU / anywhere else** → `torch` — portable pure-PyTorch loop with
    a single batched input GEMM up-front and a custom autograd Function.

`parallel=True` enables the Heinsen log-domain parallel scan (pure
PyTorch — works on any device). On Apple Silicon it's the fastest
option: O(log T) instead of O(T) — see "Performance" below.

## Usage

```python
import torch
from prlstm import PRLSTM

# Works on CUDA, MPS, CPU — the default "auto" backend picks the right path.
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
m = PRLSTM(input_size=128, hidden_size=256).to(device)
x = torch.randn(64, 8, 128, device=device)    # [T, B, D]
out, (h_n, c_n) = m(x)

m.set_mode(parallel=True)                      # Heinsen-scan forward
out_par, _ = m(x)
```

## Performance

Apple M4 Pro, PyTorch 2.10, MPS backend, fp32, training step (forward
+ backward), median of 15 iters:

| Shape (T, B, D, H)    | nn.LSTM | PRLSTM (metal) | PRLSTM (parallel) |
|-----------------------|---------|----------------|-------------------|
| 32,  16,   64,  128   | 0.99 ms | 1.61 ms (1.63×) | 0.65 ms (0.66×)  |
| 64,  32,  128,  256   | 3.35 ms | 3.25 ms (0.97×) | 2.26 ms (0.68×)  |
| 128, 32,  256,  256   | 6.81 ms | 6.91 ms (1.01×) | 4.45 ms (0.65×)  |
| 256, 32,  256,  512   | 28.1 ms | 26.8 ms (0.96×) | —                |
| 64,  32,  512, 1024   | 20.5 ms | 20.4 ms (0.99×) | —                |
| 64,  32, 1024, 2048   | 79.1 ms | 77.5 ms (0.98×) | —                |

The Metal backend matches `nn.LSTM` at all production training scales
(H ≥ 256). At very small hidden sizes the per-step kernel-launch
overhead still shows — we issue 2 MPS launches per step (recurrent
matmul + fused cell) where `nn.LSTM` does the whole T-step recurrence
in one fused MPSGraph launch. The parallel-scan mode beats `nn.LSTM`
at every shape (O(log T) vs O(T)).

## Tests

```bash
python3 tests/test_cpp.py        # reference parity on CPU + MPS
python3 tests/test_parallel.py   # parallel-vs-recurrent on whichever GPU is present
python3 tests/test_metal.py      # Metal backend parity (macOS + MPS)
python3 tests/test_triton.py     # Triton backend parity (CUDA only)
```

## Related work

prLSTM follows recent work on making LSTMs trainable in parallel by removing the
gates' dependence on the previous hidden state, which turns the state update into
an associative first-order recurrence that a parallel prefix scan can evaluate in
O(log T):

- **Were RNNs All We Needed?** — Feng et al., 2024
  ([arXiv:2410.01201](https://arxiv.org/abs/2410.01201)). Introduces *minLSTM* /
  *minGRU*, minimal RNNs whose gates drop the hidden-state dependence so the
  recurrence `h_t = f_t ⊙ h_{t-1} + i_t ⊙ ĥ_t` can be run as a parallel scan.
  prLSTM's `parallel=True` mode (dropping the recurrent term at `a = 0`) is the
  same construction.
- **xLSTM: Extended Long Short-Term Memory** — Beck et al., 2024
  ([arXiv:2405.04517](https://arxiv.org/abs/2405.04517)). Scales LSTMs with
  exponential gating; its *mLSTM* cell is fully parallelisable by making the gates
  depend only on the input — a matrix-memory / chunkwise-parallel route to the
  same goal.
- **Heinsen, 2023** ([arXiv:2311.06281](https://arxiv.org/abs/2311.06281)) — the
  log-domain associative scan used to implement the `parallel=True` forward.

## License

MIT.

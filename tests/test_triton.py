"""Cross-check the Triton backend against the C++ backend.

Both backends should produce numerically identical results (the cell math is the
same; only the kernel implementation differs). fp32 should match to ~1e-5,
fp64 — only if Triton supports fp64 well, which it sometimes doesn't, so we
focus on fp32 and bf16.

Tests:
  1. forward       - output and (h_n, c_n) match the C++ path
  2. backward      - gradients on every parameter and on x match
  3. training_step - after one SGD step, updated parameters match
"""

import sys
import torch


from prlstm import PRLSTM


def tol(dtype):
    return {torch.float32: 1e-4, torch.bfloat16: 1e-2}[dtype]


def _clone_with_same_params(src, backend):
    """Make a fresh PRLSTM with `backend` and copy src's parameters into it."""
    m = PRLSTM(src.input_size, src.hidden_size, backend=backend).to(
        next(src.parameters()).device, next(src.parameters()).dtype
    )
    with torch.no_grad():
        for (n_dst, p_dst), (n_src, p_src) in zip(m.named_parameters(),
                                                  src.named_parameters()):
            assert n_dst == n_src, (n_dst, n_src)
            p_dst.copy_(p_src)
    return m


def test_forward(device, dtype, T=10, B=4, D=16, H=24):
    print(f"  forward   T={T} B={B} D={D} H={H}")
    torch.manual_seed(0)
    m_cpp = PRLSTM(D, H, backend="cpp").to(device, dtype)
    m_tri = _clone_with_same_params(m_cpp, "triton")

    x = torch.randn(T, B, D, device=device, dtype=dtype)
    out_cpp, (h_cpp, c_cpp) = m_cpp(x)
    out_tri, (h_tri, c_tri) = m_tri(x)

    do = (out_cpp - out_tri).abs().max().item()
    dh = (h_cpp - h_tri).abs().max().item()
    dc = (c_cpp - c_tri).abs().max().item()
    t = tol(dtype)
    print(f"    out diff: {do:.3e}  h_n diff: {dh:.3e}  c_n diff: {dc:.3e}  (tol {t:.0e})")
    assert max(do, dh, dc) < t, f"forward mismatch (dtype={dtype})"


def test_backward(device, dtype, T=8, B=3, D=8, H=12):
    print(f"  backward  T={T} B={B} D={D} H={H}")
    torch.manual_seed(0)
    m_cpp = PRLSTM(D, H, backend="cpp").to(device, dtype)
    m_tri = _clone_with_same_params(m_cpp, "triton")

    x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)

    out_cpp, _ = m_cpp(x)
    out_cpp.sum().backward()
    out_tri, _ = m_tri(x2)
    out_tri.sum().backward()

    t = tol(dtype)
    for (n, p_cpp), (_, p_tri) in zip(m_cpp.named_parameters(),
                                      m_tri.named_parameters()):
        d = (p_cpp.grad - p_tri.grad).abs().max().item()
        print(f"    grad {n:18s} max diff: {d:.3e}  (tol {t:.0e})")
        assert d < t, f"grad mismatch on {n}: {d}"
    dxd = (x.grad - x2.grad).abs().max().item()
    print(f"    grad {'x':18s} max diff: {dxd:.3e}  (tol {t:.0e})")
    assert dxd < t


def test_training_step(device, dtype, T=12, B=4, D=8, H=12, lr=1e-2):
    print(f"  training_step  lr={lr}")
    torch.manual_seed(0)
    m_cpp = PRLSTM(D, H, backend="cpp").to(device, dtype)
    m_tri = _clone_with_same_params(m_cpp, "triton")

    x = torch.randn(T, B, D, device=device, dtype=dtype)
    target = torch.randn(T, B, H, device=device, dtype=dtype)

    for m in (m_cpp, m_tri):
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        opt.zero_grad()
        out, _ = m(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()

    t = tol(dtype)
    for (n, p_cpp), (_, p_tri) in zip(m_cpp.named_parameters(),
                                      m_tri.named_parameters()):
        d = (p_cpp - p_tri).abs().max().item()
        print(f"    updated {n:18s} max diff: {d:.3e}  (tol {t:.0e})")
        assert d < t, f"updated param mismatch on {n}: {d}"


def main():
    if not torch.cuda.is_available():
        print("Triton requires CUDA"); sys.exit(1)
    for dtype in [torch.float32]:
        print(f"\n=== device=cuda  dtype={dtype} ===")
        test_forward("cuda", dtype)
        test_backward("cuda", dtype)
        test_training_step("cuda", dtype)
    print("\nAll Triton-vs-cpp checks passed.")


if __name__ == "__main__":
    main()

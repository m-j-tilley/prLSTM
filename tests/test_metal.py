"""Verification tests for the Metal fused-cell backend.

Same shape as test_triton.py — cross-checks the Metal backend against the
portable torch backend (which itself is verified bit-equivalent to the
reference cell by test_cpp.py). Only meaningful on macOS with MPS.
"""

import sys
import torch

from headstartlstm import HeadStartLSTM


def tol(dtype):
    return {torch.float32: 1e-4}[dtype]


def _clone_with_same_params(src, backend):
    m = HeadStartLSTM(src.input_size, src.hidden_size, backend=backend).to(
        next(src.parameters()).device, next(src.parameters()).dtype
    )
    with torch.no_grad():
        for (n_dst, p_dst), (n_src, p_src) in zip(m.named_parameters(),
                                                  src.named_parameters()):
            assert n_dst == n_src, (n_dst, n_src)
            p_dst.copy_(p_src)
    return m


def test_forward(T=10, B=4, D=16, H=24, dtype=torch.float32):
    print(f"  forward   T={T} B={B} D={D} H={H}")
    torch.manual_seed(0)
    m_ref = HeadStartLSTM(D, H, backend="torch").to("mps", dtype)
    m_met = _clone_with_same_params(m_ref, "metal")

    x = torch.randn(T, B, D, device="mps", dtype=dtype)
    out_ref, (h_ref, c_ref) = m_ref(x)
    out_met, (h_met, c_met) = m_met(x)

    do = (out_ref - out_met).abs().max().item()
    dh = (h_ref - h_met).abs().max().item()
    dc = (c_ref - c_met).abs().max().item()
    t = tol(dtype)
    print(f"    out diff: {do:.3e}  h_n diff: {dh:.3e}  c_n diff: {dc:.3e}  (tol {t:.0e})")
    assert max(do, dh, dc) < t


def test_backward(T=8, B=3, D=8, H=12, dtype=torch.float32):
    print(f"  backward  T={T} B={B} D={D} H={H}")
    torch.manual_seed(0)
    m_ref = HeadStartLSTM(D, H, backend="torch").to("mps", dtype)
    m_met = _clone_with_same_params(m_ref, "metal")

    x  = torch.randn(T, B, D, device="mps", dtype=dtype, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)

    out_ref, _ = m_ref(x);  out_ref.sum().backward()
    out_met, _ = m_met(x2); out_met.sum().backward()

    t = tol(dtype)
    for (n, p_ref), (_, p_met) in zip(m_ref.named_parameters(),
                                      m_met.named_parameters()):
        d = (p_ref.grad - p_met.grad).abs().max().item()
        print(f"    grad {n:18s} max diff: {d:.3e}  (tol {t:.0e})")
        assert d < t
    dxd = (x.grad - x2.grad).abs().max().item()
    print(f"    grad {'x':18s} max diff: {dxd:.3e}  (tol {t:.0e})")
    assert dxd < t


def test_training_step(T=12, B=4, D=8, H=12, lr=1e-2, dtype=torch.float32):
    print(f"  training_step  lr={lr}")
    torch.manual_seed(0)
    m_ref = HeadStartLSTM(D, H, backend="torch").to("mps", dtype)
    m_met = _clone_with_same_params(m_ref, "metal")

    x = torch.randn(T, B, D, device="mps", dtype=dtype)
    target = torch.randn(T, B, H, device="mps", dtype=dtype)

    for m in (m_ref, m_met):
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        opt.zero_grad()
        out, _ = m(x)
        ((out - target) ** 2).mean().backward()
        opt.step()

    t = tol(dtype)
    for (n, p_ref), (_, p_met) in zip(m_ref.named_parameters(),
                                      m_met.named_parameters()):
        d = (p_ref - p_met).abs().max().item()
        print(f"    updated {n:18s} max diff: {d:.3e}  (tol {t:.0e})")
        assert d < t


def main():
    if sys.platform != "darwin" or not torch.backends.mps.is_available():
        print("Metal backend requires macOS with MPS"); sys.exit(0)
    from headstartlstm._metal import available
    if not available():
        print("Metal extension failed to load — skipping"); sys.exit(0)
    print("\n=== Metal vs torch backend ===")
    test_forward()
    test_backward()
    test_training_step()
    print("\nAll Metal-vs-torch checks passed.")


if __name__ == "__main__":
    main()

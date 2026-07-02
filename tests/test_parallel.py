"""Parity + divergence test for parallel-mode HeadStartLSTM.

Mathematical claim:
  At a = 0, the parallel-mode forward (Heinsen log-domain scan, dropping the
  W_hh contribution) is *exactly* equivalent to the recurrent-mode forward
  (because the recurrent W_hh term is multiplied by a, which is zero). As a
  grows away from 0, the two diverge — parallel becomes an approximation.

Tests:
  1. parity_at_zero_a: parallel ≡ recurrent when a == 0 (to numerical tol)
  2. divergence_at_nonzero_a: parallel and recurrent give different outputs
     when a != 0; the gap should grow with |a|
  3. backward_at_zero_a: gradients also match at a = 0
"""

import sys
import torch


from headstartlstm import HeadStartLSTM


def tol(dtype):
    return {torch.float32: 1e-4, torch.bfloat16: 1e-2}[dtype]


def _default_backend():
    """Pick a recurrent backend that works on the active device.
    triton is CUDA-only; torch works everywhere (CUDA, MPS, CPU)."""
    return "triton" if torch.cuda.is_available() else "torch"


def _make_models(D, H, dtype, device, a_value):
    """Build two HeadStartLSTMs sharing the same parameters, one parallel, one recurrent."""
    torch.manual_seed(0)
    backend = _default_backend()
    m_rec = HeadStartLSTM(D, H, backend=backend, parallel=False).to(device, dtype)
    m_par = HeadStartLSTM(D, H, backend=backend, parallel=True ).to(device, dtype)
    with torch.no_grad():
        for p_par, p_rec in zip(m_par.parameters(), m_rec.parameters()):
            p_par.copy_(p_rec)
        m_rec.a.fill_(a_value)
        m_par.a.fill_(a_value)
    return m_rec, m_par


def test_parity_at_zero_a(device, dtype=torch.float32,
                          T=12, B=3, D=16, H=24):
    print(f"  parity_at_zero_a   T={T} B={B} D={D} H={H}  dtype={dtype}")
    m_rec, m_par = _make_models(D, H, dtype, device, a_value=0.0)
    x = torch.randn(T, B, D, device=device, dtype=dtype)

    out_rec, (h_rec, c_rec) = m_rec(x)
    out_par, (h_par, c_par) = m_par(x)

    do = (out_rec - out_par).abs().max().item()
    dh = (h_rec   - h_par).abs().max().item()
    dc = (c_rec   - c_par).abs().max().item()
    print(f"    out diff {do:.3e}  h_n diff {dh:.3e}  c_n diff {dc:.3e}  (tol {tol(dtype):.0e})")
    assert max(do, dh, dc) < tol(dtype), f"parallel != recurrent at a=0"


def test_divergence_at_nonzero_a(device, dtype=torch.float32,
                                  T=12, B=3, D=16, H=24):
    print(f"  divergence_with_a  T={T} B={B} D={D} H={H}  dtype={dtype}")
    x = torch.randn(T, B, D, device=device, dtype=dtype)

    # Sweep a values, measure parallel-vs-recurrent gap. Should grow with |a|.
    diffs = []
    for a_val in [0.0, 1e-4, 1e-2, 1e-1, 1.0]:
        m_rec, m_par = _make_models(D, H, dtype, device, a_value=a_val)
        out_rec, _ = m_rec(x)
        out_par, _ = m_par(x)
        d = (out_rec - out_par).abs().max().item()
        diffs.append((a_val, d))
        print(f"    a={a_val:>7}  parallel-vs-recurrent max diff = {d:.4e}")

    # Sanity: diff at a=0 is tiny; diff at a=1.0 is substantial; monotone-ish increase.
    assert diffs[0][1]  < tol(dtype), "expected tiny diff at a=0"
    assert diffs[-1][1] > diffs[0][1] * 100, "expected major diff at a=1.0"
    print("    OK: gap grows with |a|, as expected for the head-start trick")


def test_backward_parity_at_zero(device, dtype=torch.float32,
                                 T=10, B=3, D=8, H=12):
    print(f"  backward_parity_a0  T={T} B={B} D={D} H={H}  dtype={dtype}")
    m_rec, m_par = _make_models(D, H, dtype, device, a_value=0.0)

    x_rec = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)
    x_par = x_rec.detach().clone().requires_grad_(True)

    out_rec, _ = m_rec(x_rec)
    out_rec.sum().backward()
    out_par, _ = m_par(x_par)
    out_par.sum().backward()

    # Parity claim: input-path gradients (W_ih, b_ih, x) match exactly at a=0,
    # since the parallel computation for these is identical to recurrent at a=0.
    # Recurrent-path gradients (W_hh, b_hh, a) DO NOT match — that's expected:
    #   - parallel mode drops the W_hh contribution entirely, so it never
    #     computes those grads (None)
    #   - recurrent mode's chain rule still produces non-zero d_a / d_W_hh /
    #     d_b_hh at a=0 (the derivative w.r.t. a of `a * (W_hh @ h + b_hh)`
    #     evaluated at a=0 is just `W_hh @ h + b_hh`, generally non-zero).
    # This asymmetry is exactly the head-start training trick: parallel mode
    # does cheap input-path updates while leaving a / W_hh / b_hh alone, then
    # you switch to recurrent to start training them.
    INPUT_PATH = {"weight_ih_l0", "bias_ih_l0"}
    t = tol(dtype)
    for (n_r, p_r), (n_p, p_p) in zip(m_rec.named_parameters(),
                                      m_par.named_parameters()):
        if n_r in INPUT_PATH:
            d = (p_r.grad - p_p.grad).abs().max().item()
            print(f"    grad {n_r:18s} max diff {d:.3e}  (tol {t:.0e})")
            assert d < t, f"grad mismatch on input-path param {n_r}: {d}"
        else:
            par_status = "None" if p_p.grad is None else f"{p_p.grad.abs().max().item():.3e}"
            rec_status = "None" if p_r.grad is None else f"{p_r.grad.abs().max().item():.3e}"
            print(f"    grad {n_r:18s} parallel={par_status}  recurrent={rec_status}  (no parity expected)")
    dxd = (x_rec.grad - x_par.grad).abs().max().item()
    print(f"    grad {'x':18s} max diff {dxd:.3e}  (tol {t:.0e})")
    assert dxd < t, f"x grad mismatch at a=0: {dxd}"


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = _pick_device()
    print(f"=== parallel-vs-recurrent (device={device}) ===")
    test_parity_at_zero_a(device)
    test_divergence_at_nonzero_a(device)
    test_backward_parity_at_zero(device)
    print("\nAll parallel-mode tests passed.")


if __name__ == "__main__":
    main()

"""Verification tests for headstartlstm.HeadStartLSTM.

Tests:
  1. forward       -- output matches a pure-Python reference of the same cell
  2. backward      -- gradients w.r.t. every parameter (and x) match
  3. training_step -- after one SGD step, updated parameter values match

The reference is a step-by-step Python implementation of the modified cell
(sigmoid g_t, per-gate a scaling on the W_hh h_{t-1} + b_hh contribution).
By construction it expresses the same math; the kernel should match it
bit-for-bit at fp64 and to ~1e-5 at fp32.

Run:  python3 tests/test_cpp.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


from headstartlstm import HeadStartLSTM


def reference_forward(x, h0, c0, W_ih, W_hh, b_ih, b_hh, a):
    """Pure-Python reference of the modified LSTM cell, step by step.
    Gate order [i, f, g, o]. g uses sigmoid. hh contribution scaled per-gate by a."""
    T, B, _ = x.shape
    H = h0.shape[-1]
    h, c = h0, c0
    outputs = []

    a_expanded = a.repeat_interleave(H)
    W_hh_scaled = a_expanded.unsqueeze(1) * W_hh
    b_hh_scaled = a_expanded * b_hh

    for t in range(T):
        gates = F.linear(x[t], W_ih, b_ih) + F.linear(h, W_hh_scaled, b_hh_scaled)
        i_, f_, g_, o_ = gates.chunk(4, dim=1)
        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_)
        g_ = torch.sigmoid(g_)
        o_ = torch.sigmoid(o_)
        c = f_ * c + i_ * g_
        h = o_ * torch.tanh(c)
        outputs.append(h)

    return torch.stack(outputs, dim=0), h, c


def tol_for(dtype):
    return {torch.float32: 1e-5, torch.float64: 1e-10}[dtype]


def test_forward(device, dtype, T=10, B=4, D=16, H=24):
    print(f"  forward   T={T} B={B} D={D} H={H}")
    torch.manual_seed(0)
    m = HeadStartLSTM(D, H).to(device, dtype)
    x = torch.randn(T, B, D, device=device, dtype=dtype)

    out_k, _ = m(x)
    with torch.no_grad():
        out_r, _, _ = reference_forward(
            x, x.new_zeros(B, H), x.new_zeros(B, H),
            m.weight_ih_l0, m.weight_hh_l0,
            m.bias_ih_l0,   m.bias_hh_l0,
            m.a,
        )
    d = (out_k - out_r).abs().max().item()
    tol = tol_for(dtype)
    print(f"    max output diff: {d:.3e}  (tol {tol:.0e})")
    assert d < tol, f"forward mismatch: {d}"


def test_backward(device, dtype, T=8, B=3, D=8, H=12):
    print(f"  backward  T={T} B={B} D={D} H={H}")
    torch.manual_seed(0)
    m = HeadStartLSTM(D, H).to(device, dtype)
    x = torch.randn(T, B, D, device=device, dtype=dtype, requires_grad=True)

    # --- kernel ---
    out_k, _ = m(x)
    out_k.sum().backward()
    g_k = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
    g_k["x"] = x.grad.clone()

    # reset
    x.grad = None
    for p in m.parameters():
        p.grad = None

    # --- reference ---
    out_r, _, _ = reference_forward(
        x, x.new_zeros(B, H), x.new_zeros(B, H),
        m.weight_ih_l0, m.weight_hh_l0,
        m.bias_ih_l0,   m.bias_hh_l0,
        m.a,
    )
    out_r.sum().backward()
    g_r = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
    g_r["x"] = x.grad.clone()

    tol = tol_for(dtype)
    for n in g_k:
        d = (g_k[n] - g_r[n]).abs().max().item()
        print(f"    grad {n:18s} max diff: {d:.3e}  (tol {tol:.0e})")
        assert d < tol, f"grad mismatch on {n}: {d}"


def test_training_step(device, dtype, T=12, B=4, D=8, H=12, lr=1e-2):
    print(f"  training_step  lr={lr}")
    torch.manual_seed(0)

    m_k = HeadStartLSTM(D, H).to(device, dtype)
    ref_params = {n: p.detach().clone().requires_grad_(True)
                  for n, p in m_k.named_parameters()}

    x      = torch.randn(T, B, D, device=device, dtype=dtype)
    target = torch.randn(T, B, H, device=device, dtype=dtype)

    # --- kernel SGD step ---
    opt = torch.optim.SGD(m_k.parameters(), lr=lr)
    opt.zero_grad()
    out_k, _ = m_k(x)
    loss_k = ((out_k - target) ** 2).mean()
    loss_k.backward()
    opt.step()

    # --- reference SGD step ---
    out_r, _, _ = reference_forward(
        x, x.new_zeros(B, H), x.new_zeros(B, H),
        ref_params["weight_ih_l0"], ref_params["weight_hh_l0"],
        ref_params["bias_ih_l0"],   ref_params["bias_hh_l0"],
        ref_params["a"],
    )
    loss_r = ((out_r - target) ** 2).mean()
    loss_r.backward()
    with torch.no_grad():
        for p in ref_params.values():
            p.sub_(lr * p.grad)

    tol = tol_for(dtype)
    for n, p in m_k.named_parameters():
        d = (p - ref_params[n]).abs().max().item()
        print(f"    updated {n:18s} max diff: {d:.3e}  (tol {tol:.0e})")
        assert d < tol, f"updated param mismatch on {n}: {d}"
    print(f"    losses:  kernel={loss_k.item():.6f}  ref={loss_r.item():.6f}")


def main():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    dtypes = [torch.float32, torch.float64]

    for device in devices:
        for dtype in dtypes:
            # MPS doesn't support fp64; skip silently.
            if device == "mps" and dtype == torch.float64:
                continue
            print(f"\n=== device={device}  dtype={dtype} ===")
            test_forward(device, dtype)
            test_backward(device, dtype)
            test_training_step(device, dtype)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()

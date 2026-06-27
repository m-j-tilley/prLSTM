"""Parallel-mode PRLSTM forward via Heinsen log-domain scan.

Pure-PyTorch ops only — runs on any backend (CUDA, MPS, CPU). Autograd is
traced through PyTorch, so no custom backward is required.

The scan implements c_t = f_t * c_{t-1} + b_t in O(log T) parallel steps by
working in the log domain. This is possible only because the modified cell
has all-sigmoid gates (positive coefficients) AND parallel mode drops the
W_hh h_{t-1} + b_hh recurrent contribution. The result is therefore:

  * exactly equivalent to the recurrent forward at a = 0
  * an approximation as a grows (the dropped term scales with a)

This is the head-start training trick — train cheap-and-parallel while a is
small, then switch to the recurrent forward to anneal a away from zero.
"""

import torch


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


def prlstm_parallel(x, h0, c0, W_ih, W_hh, b_ih, b_hh, a, eps: float = 1e-6):
    """Parallel-mode forward via Heinsen scan.

    NOTE: W_hh, b_hh, a, h0 are accepted for API compatibility but are NOT
    used. Parallel mode drops the W_hh contribution. At a=0 this matches the
    recurrent forward exactly; as a grows it becomes an approximation.

    Inputs (same shapes as recurrent path):
        x:    [T, B, D]
        c0:   [B, H]
        W_ih: [4H, D],  b_ih: [4H]
    Returns:
        output [T, B, H], h_n [B, H], c_n [B, H]
    """
    T, B, D = x.shape
    H = c0.shape[-1]

    # One batched matmul for the input contribution across all T steps.
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

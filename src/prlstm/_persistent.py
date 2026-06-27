"""Persistent-kernel CUDA implementation of PRLSTM forward, JIT-compiled via
cupy.RawKernel (NVRTC under the hood — no nvcc required).

Design:
  - One thread block per batch element.
  - `4*H` threads per block; each thread computes one gate pre-activation per
    step (so this layout works for H <= 256, given the 1024-thread block limit).
  - h, c, and the gate buffer live in shared memory across all T timesteps.
    The entire forward time-loop runs inside a single kernel — no per-step
    kernel launches, no HBM roundtrip for the recurrent state.
  - W_hh stays in HBM but persists in L2 after the first load (L2 is 40 MB on
    Ampere; W_hh at H=128 fp32 is 256 KB).

Cell modifications match the rest of PRLSTM:
  1. g_t uses sigmoid (not tanh)
  2. The W_hh h_{t-1} + b_hh contribution is pre-scaled by the per-gate `a`
     (caller passes the already-scaled W_hh_scaled and b_hh_scaled).
"""

import torch
import cupy

# -------------------------------------------------------------------- kernel --
# One TB per batch element; 4H threads per TB.
# Shared mem layout: [h(H) | c(H) | gates(4H)]  -> 6H floats.
_KERNEL_SRC = r"""
extern "C" __global__ void lstm_persistent_fwd(
    const float* __restrict__ gates_ih_all,  // [T, B, 4H]
    const float* __restrict__ W_hh_scaled,   // [4H, H]
    const float* __restrict__ b_hh_scaled,   // [4H]
    const float* __restrict__ h0,            // [B, H]
    const float* __restrict__ c0,            // [B, H]
    float* __restrict__ h_save,              // [T+1, B, H]
    float* __restrict__ c_save,              // [T+1, B, H]
    float* __restrict__ saved_acts,          // [T, B, 5H] = i, f, g, o, tanh_c
    int T, int B, int H
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int gate_i = tid;                  // 0..4H

    extern __shared__ float smem[];
    float* h_sh = smem;                    // [H]
    float* c_sh = smem + H;                // [H]
    float* g_sh = smem + 2 * H;            // [4H]

    // ---- init from h0 / c0 ----
    if (tid < H) {
        float h0v = h0[b * H + tid];
        float c0v = c0[b * H + tid];
        h_sh[tid] = h0v;
        c_sh[tid] = c0v;
        h_save[0 * B * H + b * H + tid] = h0v;
        c_save[0 * B * H + b * H + tid] = c0v;
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        // Step A: gates_hh = W_hh_scaled @ h + b_hh_scaled, then + gates_ih_all[t]
        float dot = 0.0f;
        const float* w_row = W_hh_scaled + gate_i * H;
        #pragma unroll 8
        for (int j = 0; j < H; j++) {
            dot += w_row[j] * h_sh[j];
        }
        dot += b_hh_scaled[gate_i];
        dot += gates_ih_all[t * B * 4 * H + b * 4 * H + gate_i];
        g_sh[gate_i] = dot;
        __syncthreads();

        // Step B: cell ops, only threads 0..H-1
        if (tid < H) {
            float i_pre = g_sh[0 * H + tid];
            float f_pre = g_sh[1 * H + tid];
            float g_pre = g_sh[2 * H + tid];
            float o_pre = g_sh[3 * H + tid];

            float i = 1.0f / (1.0f + __expf(-i_pre));
            float f = 1.0f / (1.0f + __expf(-f_pre));
            float g = 1.0f / (1.0f + __expf(-g_pre));   // sigmoid g, not tanh
            float o = 1.0f / (1.0f + __expf(-o_pre));

            float c_prev = c_sh[tid];
            float c_new = f * c_prev + i * g;
            // tanh via fast __expf: tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
            float e2c = __expf(2.0f * c_new);
            float tanh_c = (e2c - 1.0f) / (e2c + 1.0f);
            float h_new = o * tanh_c;

            // save for backward
            int s_base = t * B * 5 * H + b * 5 * H;
            saved_acts[s_base + 0 * H + tid] = i;
            saved_acts[s_base + 1 * H + tid] = f;
            saved_acts[s_base + 2 * H + tid] = g;
            saved_acts[s_base + 3 * H + tid] = o;
            saved_acts[s_base + 4 * H + tid] = tanh_c;

            // update shared state for next step
            c_sh[tid] = c_new;
            h_sh[tid] = h_new;

            // write h_save[t+1], c_save[t+1]
            h_save[(t + 1) * B * H + b * H + tid] = h_new;
            c_save[(t + 1) * B * H + b * H + tid] = c_new;
        }
        __syncthreads();
    }
}
"""

# Compile once on first use (lazy import-time would force NVRTC at module load).
_kernel = None
def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cupy.RawKernel(_KERNEL_SRC, "lstm_persistent_fwd",
                                 options=("--use_fast_math",))
    return _kernel


@torch.no_grad()
def persistent_forward(gates_ih_all, W_hh_scaled, b_hh_scaled, h0, c0):
    """
    Inputs (all on CUDA, fp32):
        gates_ih_all  [T, B, 4H]
        W_hh_scaled   [4H, H]
        b_hh_scaled   [4H]
        h0, c0        [B, H] each
    Returns:
        h_save        [T+1, B, H]
        c_save        [T+1, B, H]
        saved_acts    [T, B, 5H]
    """
    T, B, four_H = gates_ih_all.shape
    H = four_H // 4
    assert gates_ih_all.dtype == torch.float32 and gates_ih_all.is_cuda, "fp32 CUDA only"
    assert h0.shape == (B, H) and c0.shape == (B, H)
    assert W_hh_scaled.shape == (4 * H, H)
    assert b_hh_scaled.shape == (4 * H,)
    assert 4 * H <= 1024, f"persistent kernel needs 4H <= 1024 threads/block, got 4H={4*H}"

    device = gates_ih_all.device
    h_save     = torch.empty(T + 1, B, H, device=device, dtype=torch.float32)
    c_save     = torch.empty(T + 1, B, H, device=device, dtype=torch.float32)
    saved_acts = torch.empty(T, B, 5 * H, device=device, dtype=torch.float32)

    smem_bytes = 6 * H * 4   # h(H) + c(H) + gates(4H) floats

    kernel = _get_kernel()
    kernel(
        (B,),                 # grid
        (4 * H,),             # block
        (gates_ih_all.data_ptr(), W_hh_scaled.data_ptr(), b_hh_scaled.data_ptr(),
         h0.data_ptr(), c0.data_ptr(),
         h_save.data_ptr(), c_save.data_ptr(), saved_acts.data_ptr(),
         T, B, H),
        shared_mem=smem_bytes,
    )
    return h_save, c_save, saved_acts

// HeadStartLSTM: fused-cell Metal compute kernels.
//
// Strategy (mirrors what the Triton path does on CUDA, scaled to MPS):
//
//   * Per-step recurrent matmul `h @ W_hh.T` and the batched input GEMM
//     `linear(x, W_ih, b_ih)` go through PyTorch's MPSGraph matmul — that's
//     the right tool for the GEMM work because Apple's matmul uses
//     tile-based reuse via the matrix engine.
//   * The per-step elementwise chain (add b_hh_scaled, add gates_ih, 4×
//     sigmoid, mul/add for c, tanh, mul for h) collapses into a single
//     Metal launch via the `headstartlstm_cell_fwd` kernel below.
//
// On MPS this cuts the launch count per step from ~10 (one launch per
// PyTorch op) to 2 (matmul + fused cell). At small/medium H that closes
// most of the gap to `nn.LSTM`, which is doing the whole T-step recurrence
// inside one fused MPSGraph LSTM op.
//
// A persistent variant (one Metal launch for all T steps) was attempted
// and removed: it was memory-bandwidth-bound because the in-kernel matmul
// can't reuse W_hh across threads the way MPSGraph's tiled matmul does.
//
// Cell modifications match the rest of HeadStartLSTM:
//   (1) g_t uses sigmoid (not tanh)
//   (2) caller pre-scales W_hh / b_hh per-gate by `a`.

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

#include <string>
#include <vector>

using namespace at::native::mps;

// =====================================================================
// Metal shader source: two kernels — forward cell, backward cell.
// =====================================================================
static const char* HeadStartLSTM_METAL_SRC = R"MTL(
#include <metal_stdlib>
using namespace metal;

// Per-(batch, h-position) thread. Reads three contributions to the gate
// pre-activations (recurrent matmul, input matmul, b_hh_scaled) and emits
// h_t, c_t plus saved activations for the backward.
kernel void headstartlstm_cell_fwd(
    device const float* mm_result    [[buffer(0)]],   // [B, 4H] = h_{t-1} @ W_hh^T
    device const float* gates_ih     [[buffer(1)]],   // [B, 4H] = linear(x_t, W_ih, b_ih)
    device const float* b_hh_scaled  [[buffer(2)]],   // [4H]
    device const float* c_prev       [[buffer(3)]],   // [B, H]
    device       float* h_out        [[buffer(4)]],   // [B, H]
    device       float* c_out        [[buffer(5)]],   // [B, H]
    device       float* saved        [[buffer(6)]],   // [B, 5H] = [i, f, g, o, tanh_c]
    constant uint& H_                [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint H = H_;
    const uint b = gid.y;
    const uint h = gid.x;
    if (h >= H) return;

    const uint mm_row = b * 4 * H;
    float i_pre = mm_result[mm_row + 0 * H + h] + gates_ih[mm_row + 0 * H + h] + b_hh_scaled[0 * H + h];
    float f_pre = mm_result[mm_row + 1 * H + h] + gates_ih[mm_row + 1 * H + h] + b_hh_scaled[1 * H + h];
    float g_pre = mm_result[mm_row + 2 * H + h] + gates_ih[mm_row + 2 * H + h] + b_hh_scaled[2 * H + h];
    float o_pre = mm_result[mm_row + 3 * H + h] + gates_ih[mm_row + 3 * H + h] + b_hh_scaled[3 * H + h];

    float i_act = 1.0f / (1.0f + exp(-i_pre));
    float f_act = 1.0f / (1.0f + exp(-f_pre));
    float g_act = 1.0f / (1.0f + exp(-g_pre));   // sigmoid g (mod #1)
    float o_act = 1.0f / (1.0f + exp(-o_pre));

    float c_p   = c_prev[b * H + h];
    float c_n   = f_act * c_p + i_act * g_act;
    float t_c   = tanh(c_n);
    float h_n   = o_act * t_c;

    c_out[b * H + h] = c_n;
    h_out[b * H + h] = h_n;

    const uint s_row = b * 5 * H;
    saved[s_row + 0 * H + h] = i_act;
    saved[s_row + 1 * H + h] = f_act;
    saved[s_row + 2 * H + h] = g_act;
    saved[s_row + 3 * H + h] = o_act;
    saved[s_row + 4 * H + h] = t_c;
}

// Per-step backward of the cell. Takes incoming dh, dc and the saved
// activations; writes dgates [B, 4H] (consumed by the outer recurrent
// matmul to produce dh_{t-1}) and dc_prev [B, H].
kernel void headstartlstm_cell_bwd(
    device const float* dh           [[buffer(0)]],   // [B, H]  — incoming dh
    device const float* dc           [[buffer(1)]],   // [B, H]  — incoming dc (from t+1)
    device const float* c_prev       [[buffer(2)]],   // [B, H]
    device const float* saved        [[buffer(3)]],   // [B, 5H]
    device       float* dgates       [[buffer(4)]],   // [B, 4H]
    device       float* dc_prev      [[buffer(5)]],   // [B, H]
    constant uint& H_                [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint H = H_;
    const uint b = gid.y;
    const uint h = gid.x;
    if (h >= H) return;

    const uint s_row = b * 5 * H;
    float i_act  = saved[s_row + 0 * H + h];
    float f_act  = saved[s_row + 1 * H + h];
    float g_act  = saved[s_row + 2 * H + h];
    float o_act  = saved[s_row + 3 * H + h];
    float tanh_c = saved[s_row + 4 * H + h];

    float c_p    = c_prev[b * H + h];
    float dh_v   = dh[b * H + h];
    float dc_up  = dc[b * H + h];

    float do_act    = dh_v * tanh_c;
    float dc_total  = dc_up + dh_v * o_act * (1.0f - tanh_c * tanh_c);
    float df_act    = dc_total * c_p;
    float dc_p      = dc_total * f_act;
    float di_act    = dc_total * g_act;
    float dg_act    = dc_total * i_act;

    // sigmoid: d_pre = d_act * act * (1 - act)
    float di_pre = di_act * i_act * (1.0f - i_act);
    float df_pre = df_act * f_act * (1.0f - f_act);
    float dg_pre = dg_act * g_act * (1.0f - g_act);
    float do_pre = do_act * o_act * (1.0f - o_act);

    const uint g_row = b * 4 * H;
    dgates[g_row + 0 * H + h] = di_pre;
    dgates[g_row + 1 * H + h] = df_pre;
    dgates[g_row + 2 * H + h] = dg_pre;
    dgates[g_row + 3 * H + h] = do_pre;

    dc_prev[b * H + h] = dc_p;
}
)MTL";


// =====================================================================
// Pipeline state cache.
// =====================================================================
static MetalShaderLibrary& get_lib() {
    static MetalShaderLibrary lib(HeadStartLSTM_METAL_SRC);
    return lib;
}


// =====================================================================
// Forward cell — replaces ~10 PyTorch ops per step with one Metal launch.
// =====================================================================
std::vector<torch::Tensor> headstartlstm_metal_cell_fwd(
    torch::Tensor mm_result,      // [B, 4H]
    torch::Tensor gates_ih,       // [B, 4H]
    torch::Tensor b_hh_scaled,    // [4H]
    torch::Tensor c_prev          // [B, H]
) {
    TORCH_CHECK(mm_result.is_mps(),     "mm_result must be on MPS");
    TORCH_CHECK(mm_result.dtype() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(mm_result.is_contiguous() && gates_ih.is_contiguous()
                && b_hh_scaled.is_contiguous() && c_prev.is_contiguous(),
                "all inputs must be contiguous");

    const uint32_t B  = (uint32_t)mm_result.size(0);
    const uint32_t H4 = (uint32_t)mm_result.size(1);
    TORCH_CHECK(H4 % 4 == 0, "mm_result last dim must be 4H");
    const uint32_t H = H4 / 4;
    TORCH_CHECK(c_prev.size(0) == B && c_prev.size(1) == H, "c_prev shape mismatch");

    auto opts = mm_result.options();
    auto h_out = torch::empty({B, H},     opts);
    auto c_out = torch::empty({B, H},     opts);
    auto saved = torch::empty({B, 5 * H}, opts);

    id<MTLComputePipelineState> pso = get_lib().getPipelineStateForFunc("headstartlstm_cell_fwd");
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
            [enc setComputePipelineState:pso];
            [enc setBuffer:getMTLBufferStorage(mm_result)
                    offset:mm_result.storage_offset() * mm_result.element_size()   atIndex:0];
            [enc setBuffer:getMTLBufferStorage(gates_ih)
                    offset:gates_ih.storage_offset() * gates_ih.element_size()     atIndex:1];
            [enc setBuffer:getMTLBufferStorage(b_hh_scaled)
                    offset:b_hh_scaled.storage_offset() * b_hh_scaled.element_size() atIndex:2];
            [enc setBuffer:getMTLBufferStorage(c_prev)
                    offset:c_prev.storage_offset() * c_prev.element_size()         atIndex:3];
            [enc setBuffer:getMTLBufferStorage(h_out) offset:0 atIndex:4];
            [enc setBuffer:getMTLBufferStorage(c_out) offset:0 atIndex:5];
            [enc setBuffer:getMTLBufferStorage(saved) offset:0 atIndex:6];
            [enc setBytes:&H length:sizeof(H) atIndex:7];

            // (H threads in x, B threads in y). Pick a 1D threadgroup along x.
            NSUInteger tg_w = std::min<NSUInteger>(H, pso.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(H, B, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
        }
    });
    return {h_out, c_out, saved};
}


// =====================================================================
// Backward cell — single Metal launch per step.
// =====================================================================
std::vector<torch::Tensor> headstartlstm_metal_cell_bwd(
    torch::Tensor dh,             // [B, H]
    torch::Tensor dc,             // [B, H]
    torch::Tensor c_prev,         // [B, H]
    torch::Tensor saved           // [B, 5H]
) {
    TORCH_CHECK(dh.is_mps() && dh.dtype() == torch::kFloat32, "fp32 MPS only");
    TORCH_CHECK(dh.is_contiguous() && dc.is_contiguous()
                && c_prev.is_contiguous() && saved.is_contiguous(),
                "inputs must be contiguous");
    const uint32_t B = (uint32_t)dh.size(0);
    const uint32_t H = (uint32_t)dh.size(1);

    auto opts = dh.options();
    auto dgates  = torch::empty({B, 4 * H}, opts);
    auto dc_prev = torch::empty({B, H},     opts);

    id<MTLComputePipelineState> pso = get_lib().getPipelineStateForFunc("headstartlstm_cell_bwd");
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
            [enc setComputePipelineState:pso];
            [enc setBuffer:getMTLBufferStorage(dh)
                    offset:dh.storage_offset() * dh.element_size() atIndex:0];
            [enc setBuffer:getMTLBufferStorage(dc)
                    offset:dc.storage_offset() * dc.element_size() atIndex:1];
            [enc setBuffer:getMTLBufferStorage(c_prev)
                    offset:c_prev.storage_offset() * c_prev.element_size() atIndex:2];
            [enc setBuffer:getMTLBufferStorage(saved)
                    offset:saved.storage_offset() * saved.element_size() atIndex:3];
            [enc setBuffer:getMTLBufferStorage(dgates)  offset:0 atIndex:4];
            [enc setBuffer:getMTLBufferStorage(dc_prev) offset:0 atIndex:5];
            [enc setBytes:&H length:sizeof(H) atIndex:6];

            NSUInteger tg_w = std::min<NSUInteger>(H, pso.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(H, B, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
        }
    });
    return {dgates, dc_prev};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("headstartlstm_cell_fwd", &headstartlstm_metal_cell_fwd,
          "HeadStartLSTM fused-cell forward (one Metal launch per step)");
    m.def("headstartlstm_cell_bwd", &headstartlstm_metal_cell_bwd,
          "HeadStartLSTM fused-cell backward (one Metal launch per step)");
}

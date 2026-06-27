// PRLSTM: parallelisable-recurrent LSTM, C++ ATen-ops baseline.
//
// Single-layer, unidirectional LSTM forward, modified in exactly two ways
// from the canonical cell (see torch/nn/modules/rnn.py):
//
//   (1) g_t uses sigmoid(.) instead of tanh(.)
//   (2) the recurrent contribution W_hh h_{t-1} + b_hh is multiplied by a
//       learnable per-gate scalar a = [a_i, a_f, a_g, a_o].
//
// Cell:
//   gates  = linear(x_t, W_ih, b_ih) + a_expanded * linear(h_{t-1}, W_hh, b_hh)
//          where a_expanded broadcasts a per-gate over the [4H] gate axis
//   i,f,g,o = chunk(gates, 4, dim=-1)
//   i,f,g,o = sigmoid(i), sigmoid(f), sigmoid(g), sigmoid(o)   // (1)
//   c_t    = f * c_{t-1} + i * g
//   h_t    = o * tanh(c_t)
//
// Everything else (weight layout, gate ordering [i,f,g,o], bias split across
// b_ih + b_hh) matches torch.nn.LSTM so weights can be loaded/saved in the
// same format (with the extra learnable `a`).
//
// All math goes through ATen tensor ops, so device dispatch and autograd are
// handled automatically (no custom backward needed for this baseline).

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> prlstm_forward(
    torch::Tensor input,        // [T, B, input_size]
    torch::Tensor h0,           // [B, hidden_size]
    torch::Tensor c0,           // [B, hidden_size]
    torch::Tensor weight_ih,    // [4*H, input_size]   gates in order [i, f, g, o]
    torch::Tensor weight_hh,    // [4*H, H]
    torch::Tensor bias_ih,      // [4*H]
    torch::Tensor bias_hh,      // [4*H]
    torch::Tensor a_per_gate    // [4]                 one scalar per gate
) {
    TORCH_CHECK(input.dim() == 3, "input must be [T, B, input_size]");
    TORCH_CHECK(h0.dim() == 2 && c0.dim() == 2, "h0, c0 must be [B, hidden_size]");
    TORCH_CHECK(a_per_gate.dim() == 1 && a_per_gate.size(0) == 4,
                "a_per_gate must be a 1-D tensor of size 4");
    TORCH_CHECK(weight_hh.size(0) == 4 * h0.size(1),
                "weight_hh.size(0) must be 4 * hidden_size");

    const auto T = input.size(0);
    const auto B = input.size(1);
    const auto H = h0.size(1);

    // Broadcast the per-gate scalar over the [4H] gate axis once, before the loop.
    auto a_expanded   = a_per_gate.repeat_interleave(H);      // [4H]
    auto W_hh_scaled  = a_expanded.unsqueeze(1) * weight_hh;  // [4H, H]
    auto b_hh_scaled  = a_expanded * bias_hh;                 // [4H]

    torch::Tensor h = h0;
    torch::Tensor c = c0;

    std::vector<torch::Tensor> outputs;
    outputs.reserve(T);

    for (int64_t t = 0; t < T; t++) {
        auto x_t = input[t];                                          // [B, input_size]
        auto gates = at::linear(x_t, weight_ih, bias_ih)
                   + at::linear(h,   W_hh_scaled, b_hh_scaled);       // [B, 4H]

        auto chunks = gates.chunk(4, /*dim=*/1);
        auto i = chunks[0].sigmoid();
        auto f = chunks[1].sigmoid();
        auto g = chunks[2].sigmoid();   // (1) sigmoid here, not tanh
        auto o = chunks[3].sigmoid();

        c = f * c + i * g;
        h = o * c.tanh();
        outputs.push_back(h);
    }

    auto output = torch::stack(outputs, /*dim=*/0);   // [T, B, H]
    return {output, h, c};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prlstm_forward", &prlstm_forward,
          "PRLSTM forward (sigmoid g_t + per-gate a-scaled hh contribution)");
}

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, int bit);
std::vector<torch::Tensor> backward(torch::Tensor grad_output, torch::Tensor x1, torch::Tensor x2, int bit);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &forward, "quant attn forward");
	m.def("backward", &backward, "quant attn backward");
}

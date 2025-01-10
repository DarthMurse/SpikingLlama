#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor alpha, int bit);
std::vector<torch::Tensor> backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor x_q, int bit);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &forward, "quant relu forward");
	m.def("backward", &backward, "quant relu backward");
}

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, int bit)
{

}

std::vector<torch::Tensor backward(torch::Tensor grad_output, torch::Tensor x1, torch::Tensor x2, int bit)
{

}

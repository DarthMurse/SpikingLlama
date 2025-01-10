#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int get_cell_num(int bit)
{
	int result = 1;
	for (int i = 0; i < bit; i++)
		result *= 2;
	return result - 1;
}
// We assume that the input tensors are all oftype bfloat16.
template <typename scalar_t>
__global__ void forward_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ alpha, 
		scalar_t* __restrict__ y, scalar_t* __restrict__ x_out, scalar_t* __restrict__ x_q,
		int first_size, int S, int D, int cell)
{
	int a = blockDim.x * blockIdx.x + threadIdx.x;
	int b = blockDim.y * blockIdx.y + threadIdx.y;
	int c = blockDim.z * blockIdx.z + threadIdx.z;

	if (a < first_size && b < S && c < D)
	{
		auto scale = alpha[0];
		auto tmp = x[a*S*D + b*D + c] / scale;
		if (tmp <= 0)
			tmp = 0;
		x_out[a*S*D + b*D + c] = tmp;

		if (tmp >= 1)
			tmp = 1;
		
		tmp = 1.0 * int(tmp * cell + 0.5) / cell;
		x_q[a*S*D + b*D + c] = tmp;
		tmp *= scale;
		y[a*S*D + b*D + c] = tmp;
	}
}

template<typename scalar_t>
__global__ void backward_kernel(const scalar_t* __restrict__ grad_output, const scalar_t* __restrict__ x, const scalar_t* __restrict__ x_q, scalar_t* __restrict__ grad_input, scalar_t* __restrict__ grad_alpha, int first_size, int S, int D, int cell)
{
	int a = blockDim.x * blockIdx.x + threadIdx.x;
	int b = blockDim.y * blockIdx.y + threadIdx.y;
	int c = blockDim.z * blockIdx.z + threadIdx.z;

	if (a < first_size && b < S && c < D)
	{
		scalar_t origin = x[a*S*D + b*D + c];
		if (origin > 1)
		{
			grad_alpha[a*S*D + b*D + c] = grad_output[a*S*D + b*D + c];
		}
		else
		{
			scalar_t grad_o = grad_output[a*S*D + b*D + c];
			grad_input[a*S*D + b*D + c] = grad_o;
			grad_alpha[a*S*D + b*D + c] = grad_o * (x_q[a*S*D + b*D + c] - origin);
		}

	}
}

std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor alpha, int bit)
{
	// Expect x to be of dim >= 3, alpha to be of dim = 1.
	CHECK_INPUT(x);
	CHECK_INPUT(alpha);
	int cell = get_cell_num(bit);
	const auto S = x.size(-2), D = x.size(-1);
	int first_size = 1;
	auto size = x.sizes();
	for (int i = 0; i < size.size() - 2; i++)
		first_size *= size[i];

	auto y = torch::zeros(x.sizes(), x.options());
	auto x_out = torch::zeros(x.sizes(), x.options());
	auto x_q = torch::zeros(x.sizes(), x.options());

	dim3 threads(8, 8, 8), blocks(ceil(first_size / 8.0), ceil(S / 8.0), ceil(D / 8.0));

	if (x.dtype() == torch::kFloat32)
	{
		AT_DISPATCH_FLOATING_TYPES(x.type(), "quantrelu cuda forward", ([&]{
			forward_kernel<scalar_t><<<blocks, threads>>>(
						x.data_ptr<scalar_t>(),
						alpha.data_ptr<scalar_t>(),
						y.data_ptr<scalar_t>(),
						x_out.data_ptr<scalar_t>(),
						x_q.data_ptr<scalar_t>(),
						first_size,
						S,
						D,
						cell
					);
			}));
	}
	else
	{
		AT_DISPATCH_REDUCED_FLOATING_TYPES(x.type(), "quantrelu cuda forward", ([&]{
			forward_kernel<scalar_t><<<blocks, threads>>>(
						x.data_ptr<scalar_t>(),
						alpha.data_ptr<scalar_t>(),
						y.data_ptr<scalar_t>(),
						x_out.data_ptr<scalar_t>(),
						x_q.data_ptr<scalar_t>(),
						first_size,
						S,
						D,
						cell
					);
			}));
	}
	return {y, x_out, x_q};
}

std::vector<torch::Tensor> backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor x_q, int bit)
{
	CHECK_INPUT(grad_output);
	CHECK_INPUT(x);
	int cell = get_cell_num(bit);

	const auto S = x.size(-2), D = x.size(-1);
	int first_size = 1;
	auto size = grad_output.sizes();
	for (int i = 0; i < size.size() - 2; i++)
		first_size *= size[i];

	auto grad_input = torch::zeros(grad_output.sizes(), grad_output.options());
	auto grad_alpha = torch::zeros({first_size*S, D}, grad_output.options());

	dim3 threads(8, 8, 8), blocks(ceil(first_size / 8.0), ceil(S / 8.0), ceil(D / 8.0));

	if (grad_output.dtype() == torch::kFloat32)
	{
		AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "quantrelu cuda backward", ([&]{
			backward_kernel<scalar_t><<<blocks, threads>>>(
						grad_output.data_ptr<scalar_t>(),
						x.data_ptr<scalar_t>(),
						x_q.data_ptr<scalar_t>(),
						grad_input.data_ptr<scalar_t>(),
						grad_alpha.data_ptr<scalar_t>(),
						first_size,
						S,
						D,
						cell
					);
			}));
	}
	else
	{
		AT_DISPATCH_REDUCED_FLOATING_TYPES(grad_output.type(), "quantrelu cuda backward", ([&]{
			backward_kernel<scalar_t><<<blocks, threads>>>(
						grad_output.data_ptr<scalar_t>(),
						x.data_ptr<scalar_t>(),
						x_q.data_ptr<scalar_t>(),
						grad_input.data_ptr<scalar_t>(),
						grad_alpha.data_ptr<scalar_t>(),
						first_size,
						S,
						D,
						cell
					);
			}));
	}
	return {grad_input, grad_alpha.sum()};
}


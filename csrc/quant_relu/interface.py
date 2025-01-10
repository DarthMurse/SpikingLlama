from torch.utils.cpp_extension import load
import torch
#quant_relu_module = load(name="QuantReLU", sources=["quant_relu.cpp", "quant_relu.cu"], verbose=True)
qrelu_module = load(name="QuanReLU", sources=['csrc/quant_relu/quant_relu.cpp', 'csrc/quant_relu/quant_relu.cu'], verbose=True)
def quant_relu(x, alpha, bit=4):
    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            y, x, x_q = qrelu_module.forward(x, alpha, bit)
            ctx.save_for_backward(x, x_q)
            return y

        @staticmethod
        def backward(ctx, grad_output):
            x, x_q = ctx.saved_tensors
            grad_input, grad_alpha = qrelu_module.backward(grad_output, x, x_q, bit)
            return grad_input, grad_alpha

    return _uq.apply(x, alpha)

#x = torch.rand([4,4,4]).cuda()
#alpha = torch.rand([4]).cuda()
#y = quant_relu(x, alpha, 4)

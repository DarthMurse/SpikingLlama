from torch.utils.cpp_extension import load
import torch
#quant_relu_module = load(name="QuantReLU", sources=["quant_relu.cpp", "quant_relu.cu"], verbose=True)
qattn_module = load(name="QuanAttn", sources=['csrc/quant_attn/quant_attn.cpp', 'csrc/quant_attn/quant_attn.cu'], verbose=True)
def quant_attn(q, k, v, bit=4):
    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v):
            y, x1, x2 = qrelu_module.forward(q, k, v, bit)
            ctx.save_for_backward(x1, x2)
            return y

        @staticmethod
        def backward(ctx, grad_output):
            x1, x2 = ctx.saved_tensors
            grad_q, grad_k, grad_v = qrelu_module.backward(grad_output, x1, x2, bit)
            return grad_q, grad_k, grad_v

    return _uq.apply(q, k, v)

#x = torch.rand([4,4,4]).cuda()
#alpha = torch.rand([4]).cuda()
#y = quant_relu(x, alpha, 4)

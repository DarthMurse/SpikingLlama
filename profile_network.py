import torch
import torch.nn.functional as F
from src.quant_model import QuantGPT, act_quantization
from csrc.quant_relu.interface import *
from src.model import GPT
from src import Config
from time import time
import random
import math

'''
model_name = "tiny_LLaMA_120M"
config = Config.from_name(model_name)
model = QuantGPT(config).to("cuda")

torch.manual_seed(0)
x = torch.randint(1, 3000, [1, 2048])
x = x.cuda()
'''

def torch_attn(q, k, v, alpha):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scale = 1 / math.sqrt(q.size(-1))
    L, S = q.size(-1), k.size(-2)
    attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
    temp_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(diagonal=0)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(q.dtype)
    attn_weight = q @ k.transpose(-1, -2) * scale
    attn_weigt = act_quantization(4)(F.relu(attn_weigt), alpha)
    y = attn_weigt @ v
    return y.tranpose(1, 2)

y = random.random()
q = torch.rand([1, 100, 5, 20], dtype=torch.bfloat16).cuda().requires_grad_(True)
k = torch.rand([1, 100, 1, 20], dtype=torch.bfloat16).cuda().requires_grad_(True)
v = torch.rand([1, 100, 1, 20], dtype=torch.bfloat16).cuda().requires_grad_(True)
alpha = torch.tensor(y, dtype=torch.bfloat16).cuda().requires_grad_(True)

print("----------forward-----------")
torch.cuda.synchronize()
start = time()
y1 = torch_attn(q, k, v, alpha)
torch.cuda.synchronize()
end = time()
print(f"original: {end-start}s")
print(y1)
torch.cuda.synchronize()
start = time()
y2 = quant_attn(x, alpha, 4)
torch.cuda.synchronize()
end = time()
print(f"custom: {end-start}s")
print(y2)

print("------------backward----------")
dummy_grad = torch.rand(y1.shape, dtype=torch.bfloat16).cuda()
alpha.grad = torch.zeros(alpha.shape, dtype=torch.bfloat16).cuda()
q.grad = torch.zeros(q.shape, dtype=torch.bfloat16).cuda()
k.grad = torch.zeros(k.shape, dtype=torch.bfloat16).cuda()
v.grad = torch.zeros(v.shape, dtype=torch.bfloat16).cuda()
torch.cuda.synchronize()
start = time()
y1.backward(dummy_grad)
torch.cuda.synchronize()
end = time()
print(f"original: {end-start}s")
print(alpha.grad)
alpha.grad = torch.zeros(alpha.shape, dtype=torch.bfloat16).cuda()
q.grad = torch.zeros(q.shape, dtype=torch.bfloat16).cuda()
k.grad = torch.zeros(k.shape, dtype=torch.bfloat16).cuda()
v.grad = torch.zeros(v.shape, dtype=torch.bfloat16).cuda()
torch.cuda.synchronize()
start = time()
grad2 = y2.backward(dummy_grad)
torch.cuda.synchronize()
end = time()
print(f"custom: {end-start}s")
print(alpha.grad)

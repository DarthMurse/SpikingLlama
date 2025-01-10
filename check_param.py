import torch

def print_info(x, name):
    x = x.flatten()
    min_v = x.min().item()
    max_v = x.max().item()
    mean_v = x.mean().item()
    std_v = x.std().item()
    print(f"{name} size: {x.shape[0]}, mean: {mean_v: .3f}, std_v: {std_v: .3f}, max value: {max_v: .3f}, min value: {min_v: .3f}")

state = torch.load('out/spiking-llama-1b/iter-039000-ckpt.pth')
model_state = state['model']
attn_encoder_beta = model_state['transformer.h.0.attn.attn_encoder.zero_point']
ffn_encoder_beta = model_state['transformer.h.0.mlp.swiglu.encoder4.zero_point']
print_info(attn_encoder_beta, 'attn_encoder_beta')
print_info(ffn_encoder_beta, 'ffn_encoder_beta')


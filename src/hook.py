import torch
from src.quant_model import QuantGPT
from src.config import Config

class Hook:
    def __init__(self):
        self.output = None
        self.grad_in = None
        self.grad_out = None

    def hook_fn(self, module, input, output):
        with torch.no_grad():
            output = output
            percentage = zero_percent(output)
            self.output = percentage.item()

    def hook_bn(self, module, grad_in, grad_out):
        with torch.no_grad():
            grad_in = grad_in[0]
            grad_out = grad_out[0]
            percent_in = zero_percent(grad_in)
            percent_out = zero_percent(grad_out)
            self.grad_in = percent_in.item()
            self.grad_out = percent_out.item()

def zero_percent(x):
    epsilon = 1e-9
    x = x.flatten()
    total_num = x.shape[0]
    percent = (x < epsilon).float().sum() / total_num
    return percent

def get_fea_by_hook(model, layer_name):
    hooks = {}
    for name, module in model.named_modules():
        if layer_name in name:
            hook = Hook()
            module.register_forward_hook(hook.hook_fn)
            module.register_full_backward_hook(hook.hook_bn)
            hooks[name] = hook
    return hooks

if __name__ == "__main__":
    model_name = "tiny_LLaMA_1b"
    config = Config.from_name(model_name)
    model = QuantGPT(config)
    hooks = get_fea_by_hook(model, 'encoder')
    for name, hook in hooks.items():
        print(name)

"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi
from torch import sqrt
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
#from flash_attn import flash_attn_func
from src.config import Config
#from xformers.ops import SwiGLU
from .fused_rotary_embedding import apply_rotary_emb_func
#from csrc.quant_relu.interface import *
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

def act_quantization(b):
    def uniform_quant(x, b):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            x = x.div(alpha)
            x_c = x.clamp(min=0, max=1)
            x_q = uniform_quant(x_c, b)
            ctx.save_for_backward(x, x_q)
            x_q = x_q.mul(alpha)
            return x_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            x, x_q = ctx.saved_tensors
            i = (x > 1).float()
            grad_alpha = (grad_output * (i + (x_q - x)  * (1 - i))).sum()
            grad_input = grad_input * (1 - i)
            return grad_input, grad_alpha

    return _uq().apply

class QuantReLU(nn.ReLU):
    def __init__(self, dim, bit=4):
        super().__init__()
        self.bit = bit
        self.dim = dim
        self.act_alq = act_quantization(self.bit)
        #self.act_alpha = torch.nn.Parameter(torch.tensor(6.0))

    def forward(self, x):
        #'''
        x = F.relu(x)
        return x
        #return self.act_alq(x, self.act_alpha)
        '''
        return quant_relu(x, self.act_alpha.to(x.dtype), self.bit)
        '''

def act_heaveside(scale):
    # Surrogate function f(x) = aH(x-b) = 2 / pi * a * arctan((x-b)*scale)
    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha, beta):
            y = x - beta
            positive = (y > 0).float()
            out = alpha * positive
            ctx.save_for_backward(y, alpha)
            return out.to(x.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            y, alpha = ctx.saved_tensors
            alpha_d = 1 / pi * torch.arctan(y) + 1 / 2
            grad_alpha = alpha_d * grad_input
            arctan_d = 1 / pi * scale / ((scale * y) ** 2 + 1)
            grad_beta = -alpha * arctan_d * grad_input
            grad_input = alpha * arctan_d * grad_input
            return grad_input.to(grad_output.dtype), grad_alpha.to(grad_output.dtype), grad_beta.to(grad_output.dtype)

    return _uq().apply

class SimpleParamHeaveside(nn.Module):
    def __init__(self, size, token, normed=False):
        super().__init__()
        self.normed = normed
        #self.zero_point = torch.nn.Parameter(torch.zeros([size]))
        #self.embed_scale = torch.nn.Parameter(torch.ones([size]))
        #A = torch.arange(token)
        #A = torch.exp(A/token - 1)
        #self.token_scale = torch.nn.Parameter(A)
        #self.embed_bias = torch.nn.Parameter(torch.zeros([size]))
        #self.token_bias = torch.nn.Parameter(torch.zeros([token]))
        #self.heaveside = act_heaveside(4.0)

    def forward(self, x):
        #x1 = self.heaveside(x, self.embed_scale, self.zero_point)
        x1 = x
        return x1

class ParamHeaveside(nn.Module):
    def __init__(self, size, token):
        super().__init__()
        self.zero_point1 = torch.nn.Parameter(torch.zeros([size]))
        self.zero_point2 = torch.nn.Parameter(torch.rand([size]))
        self.zero_point3 = torch.nn.Parameter(torch.rand([size]))
        #self.zero_point2 = torch.nn.Parameter(torch.ones([size]))
        self.embed_scale1 = torch.nn.Parameter(torch.ones([size]))
        self.embed_scale2 = torch.nn.Parameter(torch.rand([size]))
        self.embed_scale3 = torch.nn.Parameter(torch.rand([size]))
        #self.embed_scale2 = torch.nn.Parameter(torch.ones([size]))
        #A = torch.arange(token)
        #A = torch.exp(A/token - 1)
        #self.token_scale = torch.nn.Parameter(A)
        #self.embed_bias = torch.nn.Parameter(torch.zeros([size]))
        #self.token_bias = torch.nn.Parameter(torch.zeros([token]))
        self.heaveside = act_heaveside(4.0)

    def forward(self, x):
        x1 = self.heaveside(x, self.embed_scale1, self.zero_point1)
        x2 = self.heaveside(x, self.embed_scale2, self.zero_point2)
        x3 = self.heaveside(x, self.embed_scale3, self.zero_point3)
        return x1 + x2 + x3

    def update(self, step):
        '''
        scale = [4, 6, 8, 10, 12]
        step_num = [10000, 20000, 30000, 40000]
        if step <= step_num[0]:
            self.heaveside = act_heaveside(scale[0])
        elif step <= step_num[1]:
            self.heaveside = act_heaveside(scale[1])
        elif step <= step_num[2]:
            self.heaveside = act_heaveside(scale[2])
        elif step <= step_num[3]:
            self.heaveside = act_heaveside(scale[3])
        else:
            self.heaveside = act_heaveside(scale[4])
        '''
        scale = 4 + 16 / math.pi * math.atan(step / 40000)
        self.heaveside = act_heaveside(scale)

class QuantGPT(nn.Module):
    def __init__(self, config: Config, layer=range(1, 13)) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.layer = layer

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        mlist = nn.ModuleList()
        for i in range(config.n_layer):
            if i+1 == layer[-1]:
                mlist.append(Block(config, quant=True, grad_on=True))
            elif i+1 in layer:
                mlist.append(Block(config, quant=True, grad_on=True))
            else:
                mlist.append(Block(config))

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=mlist,
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        '''
        for param in self.transformer.wte.parameters():
            param.requires_grad = False
        for param in self.transformer.ln_f.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
        '''

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            # RWKV: set it to 1e-4
            # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX       
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (name == "w3.weight" and isinstance(module, SwiGLU) or (name=="proj.weight" and isinstance(module, CausalSelfAttention))):  #if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)
        

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def update(self, step):
        for layer in self.transformer.h:
            layer.update(step)

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = idx.size()
        layer = self.layer
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        cos, sin = self.rope_cache
        if use_kv_cache:

            cos = cos.index_select(0, input_pos)
            sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            cos = cos[:T]
            sin = sin[:T]
            mask = None

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        loss = 0
        if not use_kv_cache:
            for i in layer:
                x, *_ = self.transformer.h[i-1](x, (cos, sin), max_seq_length)
                #print(x.flatten().abs().max().item())
            for i in range(layer[-1], self.config.n_layer):
                x, *_ = self.transformer.h[i](x, (cos, sin), max_seq_length)
                #print(x.flatten().abs().max().item())
            
        else:
            self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length, cos.size(-1) * 2)
            for i in layer:
                x, self.kv_caches[i] = self.transformer.h[i-1](x, (cos, sin), max_seq_length, mask, input_pos, self.kv_caches[i])
                #print(x.flatten().abs().max().item())
            for i in range(layer[-1], self.config.n_layer):
                x, self.kv_caches[i] = self.transformer.h[i](x, (cos, sin), max_seq_length, mask, input_pos, self.kv_caches[i])
                #print(x.flatten().abs().max().item())

        x = self.transformer.ln_f(x)

        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            #dtype=idx.dtype,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups

        k_cache_shape = (
            B,
            max_seq_length,
            heads,
            rope_cache_length + self.config.head_size - int(self.config.rotary_percentage * self.config.head_size),
        )
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (torch.zeros(k_cache_shape, device=device), torch.zeros(v_cache_shape, device=device))
            for _ in range(self.config.n_layer)
        ]


class Block(nn.Module):
    def __init__(self, config: Config, quant=False, grad_on=False) -> None:
        super().__init__()
        self.quant = quant
        if not quant:
            self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        else:
            self.norm_1 = MyNorm(config.n_embd, eps=config.norm_eps)
            #self.lora = Lora(config)
        self.attn = CausalSelfAttention(config, quant=True)
        if not config.shared_attention_norm:
            if not quant:
                self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
            else:
                self.norm_2 = MyNorm(config.n_embd, eps=config.norm_eps)
        #self.mlp = config.mlp_class(config)
        self.mlp = GptNeoxMLP(config, quant=True)
        self.config = config
        if not grad_on:
            for param in self.parameters():
                param.requires_grad = False
    
    def update(self, step):
        self.attn.update(step)
        self.mlp.update(step)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        #if self.quant:
        #    addition = self.lora(x)
        n_1 = self.norm_1(x)
        #print(f"device: {x.device}, max value = {n_1.flatten().abs().max().item()}")
        h, new_kv_cache = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache)

        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            
            x = x + h
            x = x + self.mlp(self.norm_2(x))

        #if self.quant:
        #    x = x + addition

        return x, new_kv_cache

class Lora(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rank = 4
        self.linear1 = nn.Linear(config.n_embd, self.rank, bias=False)
        self.linear2 = nn.Linear(self.rank, config.n_embd, bias=False)
        self.output = ParamHeaveside(config.n_embd, config.block_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.output(out)
        return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, quant=False) -> None:
        super().__init__()
        self.quant = quant
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        #'''
        if self.quant:
            self.encoder1 = SimpleParamHeaveside(config.n_embd, config.block_size)
            #self.encoder2 = ParamHeaveside(config.head_size, config.block_size)
            #self.encoder3 = ParamHeaveside(config.n_embd, config.block_size)
            self.encoder3 = QuantReLU(config.n_embd)
            self.attn_encoder = QuantReLU(config.block_size)
            #self.attn_encoder = QuantReLU()
            #self.softmax_encoder = AutoEncoder(config.block_size, config.block_size)
        #'''

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def update(self, step):
        self.encoder3.update(step)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        #'''
        if self.quant:
            x = self.encoder1(x)
        #'''
        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B,  T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.config.head_size)
        #v = v.reshape(B, T, -1)
        #v = self.encoder2(v)
        v = v.reshape(B, T, -1, self.config.head_size)

        cos, sin = rope

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)
        
        # n_elem = int(self.config.rotary_percentage * self.config.head_size)
    
        # q_roped = apply_rope(q[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # k_roped = apply_rope(k[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # print( (q_roped - q).sum())
        # q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        # k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        if self.quant:
            y = self.encoder3(y)
        y = self.proj(y)

        return y, kv_cache

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        '''
        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)
        '''
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)

        L, S = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
        if mask is None:
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(diagonal=0)
            post_mask = temp_mask.float().to(q.dtype)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)
        else:
            if mask.dtype == torch.bool:
                post_mask = mask.float().to(q.dtype)
                mask.masked_fill_(mask.logical_not(), float("-inf"))
            else:
                post_mask = torch.exp(mask)
                attn_bias += mask
        attn_weight = q @ k.transpose(-2, -1) * scale_factor

        if self.quant:
            #attn_weight = attn_weight.transpose(-1, -2)
            #attn_weight = self.attn_encoder(attn_weight)
            attn_weight = self.attn_encoder(attn_weight + attn_bias)
            #attn_weight = F.relu(attn_weight) / L
            #attn_weight = attn_weight.transpose(-2, -1)
            attn_weight = torch.dropout(attn_weight, 0.0, train=True)
            #attn_weight = attn_weight * post_mask   # [B, H, S, S]
            y = attn_weight @ v # [B, H, S, D_H]
            return y.transpose(1, 2)
        else:
            attn_weight += attn_bias
            attn_weight = F.relu(attn_weight)
            attn_weight = torch.dropout(attn_weight, 0.0, train=True)
            y = attn_weight @ v
            return y.transpose(1, 2)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config, quant=False) -> None:
        super().__init__()
        self.quant = quant
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        if quant:
            #self.act2 = ParamHeaveside(config.intermediate_size, config.block_size)
            self.act2 = QuantReLU(config.intermediate_size)
            #self.act1 = ParamHeaveside(config.n_embd, config.block_size)
            self.act1 = QuantReLU(config.n_embd)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
    
    def update(self, step):
        self.act1.update(step)
        self.act2.update(step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant:
            x = self.act1(x)
        x = self.fc(x)
        if not self.quant:
            x = torch.nn.functional.relu(x)
        else:
            x = self.act2(x)
        return self.proj(x)

class LLaMAMLP(nn.Module):
    def __init__(self, config: Config, quant=False):
        super().__init__()
        self.swiglu = SwiGLU(config, quant)
        self.quant = quant

    def forward(self, x):
        return self.swiglu(x)

class SwiGLU(nn.Module):
    def __init__(self, config: Config, quant=False) -> None:
        super().__init__()
        self.quant = quant
        #'''
        if self.quant:
            self.encoder1 = ParamHeaveside(config.n_embd)
            #self.encoder1 = QuantReLU()
            self.encoder2 = ParamHeaveside(config.n_embd)
            #self.encoder2 = QuantReLU()
            self.encoder3 = ParamHeaveside(config.intermediate_size)
            #self.encoder3 = QuantReLU()
            self.encoder4 = ParamHeaveside(config.intermediate_size, normed=True)
            #self.encoder4 = QuantReLU()
        #'''
        self.w1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.w2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.w3 = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        #self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #'''
        if self.quant:
            x1 = self.encoder1(x)
            x2 = self.encoder2(x)
        else:
            x1 = x2 = x
        #'''
        x1 = x2 = x
        x_fc_1 = self.w1(x1)
        x_fc_2 = self.w2(x2)
        #'''
        if self.quant:
            x = self.encoder3(x_fc_1) * x_fc_2
        else:
            x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        if self.quant:
            x = self.encoder4(x)
        #'''
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        y = self.w3(x)

        return y
        #return self.swiglu(x)


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)

class MyNorm(nn.Module):
    """
    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        #self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.size = size
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        #norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        #x_normed = x * torch.rsqrt(norm_x + self.eps)
        #x_normed = x
        #D = self.size
        #return self.weight * x_normed / (D ** 0.5)
        #return self.weight * x_normed
        return x

    #def reset_parameters(self):
        #torch.nn.init.ones_(self.weight)

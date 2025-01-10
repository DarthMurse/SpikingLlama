#from src.model import GPT
#from src.spike_model import SpikeGPT, IF, SpikeInnerProduct
from src.quant_model import QuantGPT
from src.config import Config
from src.tokenizer import Tokenizer

import torch
from torch import nn
import os
from collections import namedtuple

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import math
from typing import List, Literal, Optional, Tuple, TypedDict

import transformers
from transformers import AutoTokenizer
from tokenizers import Tokenizer as HFTokenizer

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


# export HF_ENDPOINT=https://hf-mirror.com
# python3 eval.py --model spikellama --tasks=piqa,winogrande,lambada,arc_easy


# A wrapper to the original model
class SpikeGPTFull(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        model_name = "tiny_LLaMA_120M"
        self.config = Config.from_name(model_name)

        # Choose the model you want to evaluate.
        self.model = QuantGPT(self.config)
        checkpoint = torch.load(pretrained)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        tokenizer_path = Path("checkpoints/")
        self.tokenizer = Tokenizer(tokenizer_path)

    @torch.no_grad()
    def forward(self, x, max_len=None, input_pos=None):
        # Normal GPT generation, no input_pairs or target_pairs
        lm_logits = self.model.forward(x, max_seq_length=max_len, input_pos=input_pos)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

# A wrapper for huggingface evaluation toolset
@register_model("spikellama")
class SpikeLlamaWrapper(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    # change you checkpoint path here (pretrained)
    def __init__(self, pretrained="out/264.pth", max_length=512, batch_size=None, device="cuda",
                 dtype=torch.float32):
        LM.__init__(self)
        self._model = SpikeGPTFull(pretrained).to(device=device, dtype=dtype)

        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self.add_bos_token = False
        self.logits_cache = False
        self.revision = "main"
        self.peft = None
        self.delta = None
        self.pretrained = None

        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()

if __name__ == "__main__":
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    cli_evaluate()

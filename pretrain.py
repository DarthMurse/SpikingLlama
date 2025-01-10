import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy, DeepSpeedStrategy, DDPStrategy
from torch.utils.data import DataLoader
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
#from src.model import GPT
from src.quant_model import QuantGPT, Block, Config, CausalSelfAttention
from src.packed_dataset import CombinedDataset, PackedDataset
from src.speed_monitor import SpeedMonitorFabric as Monitor
from src.speed_monitor import estimate_flops, measure_flops
from src.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from src import FusedCrossEntropyLoss
from src.hook import *
#from src.grab_act import get_fea_by_hook, test_grad_vanish, remove_hook
from pytorch_lightning.loggers import WandbLogger
import random

'''
data stored in data/openwebtext_processed
checkpotins stored in out/spiking-llama-120M
'''

model_name = "tiny_LLaMA_120M"
name = "spike-llama-120M"
out_dir = Path("out") / name

# Hyperparameters
GPU_NUM = 1
num_of_devices = GPU_NUM
micro_batch_size = 64
global_batch_size = GPU_NUM * micro_batch_size * 1  # global_batch_size = GPU_NUM * micro_batch_size * gradient_accumulation_steps
learning_rate = 4e-4
max_step = 80000 * 6
warmup_steps = 2000
log_step_interval = 10
eval_iters = 100
save_step_interval = 10000
eval_step_interval = 1000


weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps


max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", name, flush_logs_every_n_steps=log_iter_interval)
wandb_logger = WandbLogger()


def setup(
    devices: int = GPU_NUM,
    train_data_dir: Path = Path("data/openwebtext_processed/train"),
    val_data_dir: Optional[Path] = Path("data/openwebtext_processed/validation"),
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
) -> None:
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy={Block},
                state_dict_type="full",
                limit_all_gathers=True,
                sharding_strategy="FULL_SHARD",
                cpu_offload=False,
            )
            #strategy = DeepSpeedStrategy()
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=DDPStrategy(process_group_backend="gloo"), precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    #fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()

    # Load teacher and student models, student models with same initialization to teacher model.
    with fabric.init_module(empty_init=False):
        torch_model = QuantGPT(config)
        torch_model.apply(partial(torch_model._init_weights ,n_layer=config.n_layer))
        #checkpoint = torch.load("/data1/SpikingLlama/163.pth")
        #checkpoint = torch.load("out/teacher.pth")
        #torch_model.load_state_dict(checkpoint, strict=False)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(torch_model):,}")

    model = fabric.setup(torch_model)
    #model.mark_forward_method(model.update)
    #abstract_optimizer = torch.optim.AdamW([torch.zeros([1])], lr=learning_rate)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)
    #model, optimizer = fabric.setup(torch_model, optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))[-1]
        state = {"model": model, "hparams": hparams, "iter_num": 0, "step_count": 0}
    if resume :
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state, strict=False)
        state['optimizer'] = optimizer

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

# Training details here
def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    #hooks = get_fea_by_hook(model, 'encoder')

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = QuantGPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    
    
    initial_iter = state["iter_num"]
    curr_iter = 0
            
    loss_func = FusedCrossEntropyLoss()
    for  train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break
        
        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        #if state["iter_num"] % 10000 == 0:
        #    model.update(state["iter_num"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        mu = 0.00
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            #input_pairs = input_pairs.to(torch.bfloat16)
            #target_pairs = target_pairs.to(torch.bfloat16)

            #logits, mseloss = model(input_ids, input_pairs=input_pairs, target_pairs=target_pairs)
            logits = model(input_ids)
            #print(logits.flatten().max())
            # Consider the global and local loss
            global_loss = loss_func(logits, targets)
            loss = global_loss
            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()
        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )

        '''
        zero_percent = {}
        for name, hook in hooks.items():
            fabric.print(
                    f"--------------- {name} --------------------\n"
                    f"output zero percentage: {hook.output: .3f}\n"
                    f"grad in zero percentage: {hook.grad_in: .3f}\n"
                    f"grad out grad zero percentage: {hook.grad_out: .3f}"
                    )
            zero_percent[name] = hook.output
        '''
 
        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )

            
        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": val_loss.item(), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            #for name, value in zero_percent.items():
            #    fabric.log_dict({"metric/"+name: value, "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.barrier()
        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

        
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        #logits, _ = model(input_ids)
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = loss.item()
        
    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    prefix = split
    filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
    random.seed(seed)
    random.shuffle(filenames)

    dataset = PackedDataset(
        filenames,
        # n_chunks control the buffer size. 
        # Note that the buffer size also impacts the random shuffle
        # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
        n_chunks=8,
        block_size=block_size,
        shuffle=shuffle,
        seed=seed+fabric.global_rank,
        num_processes=fabric.world_size,
        process_rank=fabric.global_rank,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/slimpajama"),
    val_data_dir: Optional[Path] = Path("data/slimpajama"),
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="valid"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)

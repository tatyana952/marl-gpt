import multiprocessing

# Ensure the 'spawn' start method is used
multiprocessing.set_start_method("spawn", force=True)

from loguru import logger

import math
import os
import time
import json
from contextlib import nullcontext
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# from gpt.model_critic_v2 import CriticWithLoss, CriticGPTConfig
from gpt.model_ac import CriticWithLoss, CriticGPTConfig
from utils.multi_env_dataset import MultiEnvAggregateDataset

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 500
log_interval = 1
eval_iters = 40
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "smac-gpt"
# wandb_run_name = 'mini-mapf-gpt' # 'run' + str(time.time())
# data
dataset = "mazes-82M"
gradient_accumulation_steps = 16  # used to simulate larger batch sizes
batch_size = 256  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256
# model
n_layer = 5
n_head = 5
n_embd = 160
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 30000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 30000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = "cuda:0"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
history_len = 6
# -----------------------------------------------------------------------------
config_of_data = "dataset/train"
max_action_size = 13
# ------ Critic
calculate_rtg = True
gamma = 0.95
group_reward = True
num_bins = 128
max_value = 5
sigma_loss = 0.01
target_type = "TD_rtg"
rand_act = True
n_attr = 128
n_agents = 20
n_team = 2
mask_ratio = 0.5
target_network_frequency = 50
tau = 0.7
q_n_step = 0
alpha_cql = 0.5
loss_type = "CQL"
num_sampled_actions = 4

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("gpt/configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
current_train_index = 0
current_valid_index = 0

with open(config_of_data, "r") as f:
    data_config = json.load(f)

ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

dataloader = MultiEnvAggregateDataset(
    batch_size=batch_size,
    dataset_config=data_config["train"],
    device=device,
    max_block_size=block_size,
    max_action_size=max_action_size,
)
data = iter(dataloader)

dataloader_val = MultiEnvAggregateDataset(
    batch_size=batch_size,
    dataset_config=data_config["val"],
    device=device,
    max_block_size=block_size,
    max_action_size=max_action_size,
)
data_val = iter(dataloader_val)

X, Y, Y_mask, X_next, X_info = next(data)
logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")
action_size = X["action_mask"].shape[-1]
vocab_size = num_bins
block_size = X["obs"].shape[-1]
print(f"{action_size=}, {block_size=}")


def get_batch(data):
    x, vec_y, vec_y_mask, x_next, x_info = next(data)
    y = torch.tensor(vec_y, dtype=torch.long, device=device)
    y_mask = torch.tensor(vec_y_mask, dtype=torch.long, device=device)
    return x, y, y_mask, x_next, x_info


iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
    history_len=history_len,
    n_team=n_team,
    n_agents=n_agents,
    n_attr=n_attr,  # smac = 16,
    action_size=action_size,
    gamma=gamma,
    max_value=max_value,
    sigma_loss=sigma_loss,
    target_type=target_type,
    alpha_cql=alpha_cql,
    loss_type=loss_type,
    num_sampled_actions=num_sampled_actions,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    logger.info("Initializing a new model from scratch")
    gptconf = CriticGPTConfig(**model_args)
    model = CriticWithLoss(gptconf)
elif init_from == "resume":
    logger.info(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = CriticGPTConfig(**model_args)
    model = CriticWithLoss(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

logger.info("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))

model.to(device)
# target model:
target_network = CriticWithLoss(gptconf).to(device)
target_network.load_state_dict(model.state_dict())

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

if compile and not device == "mps":
    logger.info("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    try:
        model = torch.compile(model)
    except AttributeError:
        logger.warning("torch compile(model) requires PyTorch >= 2.0")

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        keys = ["critic_loss", "bc_loss", "loss_td", "loss_r", "actor_loss"]
        all_losses = {key: torch.zeros(eval_iters) for key in keys}
        vals = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == "train":
                X, Y, Y_mask, X_next, X_info = get_batch(data)
            else:
                X, Y, Y_mask, X_next, X_info = get_batch(data_val)
            with ctx:
                target = raw_model.calculate_target_val(X, X_next, X_info)
                _, val_logits, loss, loss_info = model(X, Y, Y_mask, target)
                val = raw_model.from_logits_to_val(val_logits).mean()
            losses[k] = loss.item()
            for key, val_loss in loss_info.items():
                all_losses[key][k] = val_loss.item()
            vals[k] = val.item()
        out[split] = dict()
        out[split]["loss"] = losses.mean()
        for key, val_loss in loss_info.items():
            out[split][key] = all_losses[key].mean()
        out[split]["Q"] = vals.mean()
    model.train()
    return out


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


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, config=config)

# training loop
X, Y, Y_mask, X_next, X_info = get_batch(data)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        model.eval()
        losses = estimate_loss()
        model.train()
        logger.info(
            f"step {iter_num}: train loss {losses['train']['loss']:.4f}, val loss {losses['val']['loss']:.4f}"
        )
        if wandb_log:
            dict_log_loss = {
                f"{split}/{key_loss}": val_loss
                for split, loss_dict in losses.items()
                for key_loss, val_loss in loss_dict.items()
            }
            wandb.log(
                {
                    "iter": iter_num,
                    **dict_log_loss,
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                    "epoch": dataloader.num_epoch,
                }
            )
        if losses["val"]["loss"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]["loss"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                logger.info(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            with torch.no_grad():
                target = target_network.calculate_target_val(X, X_next, X_info)
            _, _, loss, _ = model(X, Y, Y_mask, target)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, Y_mask, X_next, X_info = get_batch(data)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # target model
    if iter_num % target_network_frequency == 0:
        for target_network_param, model_param in zip(
            target_network.parameters(), raw_model.parameters()
        ):
            target_network_param.data.copy_(
                tau * model_param.data + (1.0 - tau) * target_network_param.data
            )

    # timing and logging
    t1 = time.monotonic()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        logger.info(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
            f"mfu {running_mfu * 100:.2f}%, {iter_num} / {max_iters}"
        )

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

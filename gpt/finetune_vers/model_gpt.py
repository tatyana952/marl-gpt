# The last var of BC+Q
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger
from torch.nn import functional as F

from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.torch_utils import calc_num_elements

from sample_factory.utils.utils import log

from torch import nn as nn


def sample_valid_actions(action_mask: torch.Tensor, sample_actions: int):
    batch_size, action_size = action_mask.size()

    # Generate random noise for shuffling
    noise = torch.rand((batch_size, action_size), device=action_mask.device)

    # Apply mask: set noise to -inf where action is invalid
    noise.masked_fill_(~action_mask, -float("inf"))

    # Get top-k actions based on noise
    _, topk_actions = torch.topk(noise, k=sample_actions, dim=-1)  # (B, S)

    return topk_actions


def sample_from_val(
    all_val: torch.Tensor, num_sample_actions: int = 1, reversed: bool = True
):
    # Compute 1 - softmax(A)
    probs = F.softmax(all_val, dim=-1)
    if reversed:
        probs = 1.0 - probs
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Create categorical distribution
    actions = torch.multinomial(
        probs, num_samples=num_sample_actions, replacement=False
    ).squeeze()

    return actions


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class NonCausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            logger.warning(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = NonCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class CriticGPTConfig:
    block_size: int = 161
    vocab_size: int = 67
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    history_len: int = 4
    n_team: int = 2
    n_agents: int = 20
    n_attr: int = 128  # smac = 16
    action_size: int = 12
    gamma: float = 0.95
    max_value: float = 20
    sigma_loss: float = 0.01
    target_type: str = "rtg"
    critic_coef: float = 0.4
    alpha_cql: float = 1
    loss_type: str = "CQL"
    num_sampled_actions: int = 4


class Critic_GPT(nn.Module):
    def __init__(self, config: CriticGPTConfig):
        super().__init__()
        self.config = config
        assert config.vocab_size % config.action_size == 0
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(1, config.n_embd),
                wpe_n_agent=nn.Embedding(config.n_agents, config.n_embd),
                wpe_n_team=nn.Embedding(config.n_team, config.n_embd),
                wpe_n_time=nn.Embedding(config.history_len, config.n_embd),
                wpe_n_attr=nn.Embedding(config.n_attr, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer - 1)]),
                critic_layers=Block(config),
                act_layers=Block(config),
                ln_critic_f=LayerNorm(config.n_embd, bias=config.bias),
                ln_actor_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head_critic = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head_actor = nn.Linear(config.n_embd, config.action_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte_act.weight = (  # TODO
        #     self.transformer.act_head.weight
        # )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )
    
    def freeze_actor(self):
        """Freeze shared transformer + actor head; keep entire critic path trainable."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.lm_head_actor.parameters():
            param.requires_grad = False
        # Unfreeze all critic-related parameters
        for param in self.transformer.critic_layers.parameters():
            param.requires_grad = True
        for param in self.transformer.ln_critic_f.parameters():
            param.requires_grad = True
        for param in self.lm_head_critic.parameters():
            param.requires_grad = True
    
    def unfreeze_actor(self):
        """Unfreeze all actor-related parameters"""
        for param in self.transformer.parameters():
            param.requires_grad = True
        for param in self.lm_head_actor.parameters():
            param.requires_grad = True
    
    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_shared_representation(self, idx):
        """Get shared transformer representation up to critic/actor split"""

        obs = idx["obs"]
        group_pos = idx["group_pos"]
        agent_pos = idx["agent_pos"]
        time_pos = idx["time_pos"]
        attr_pos = idx["attr_pos"]

        b, t = obs.shape  # (batch_size, block_size)
        x_reshaped = obs.view(b * t, 1)  # Reshape to [batch_size*block_size, 1]
        tok_emb_flat = self.transformer.wte(
            x_reshaped
        )  # [batch_size*block_size, n_embd]
        tok_emb = tok_emb_flat.view(
            b, t, -1
        )  # Reshape back to [batch_size, block_size, n_embd]

        # ------------

        pos_agent_emb = self.transformer.wpe_n_agent(
            agent_pos
        )  # position embeddings of shape (batch_size, block_size + 1, n_embd)
        pos_team_emb = self.transformer.wpe_n_team(
            group_pos
        )  # position embeddings of shape (batch_size, block_size + 1, n_embd)
        pos_time_emb = self.transformer.wpe_n_time(
            time_pos
        )  # position embeddings of shape (batch_size, block_size + 1, n_embd)
        pos_attr_emb = self.transformer.wpe_n_attr(
            attr_pos
        )  # position embeddings of shape (batch_size, block_size + 1, n_embd)

        x = self.transformer.drop(
            tok_emb + pos_agent_emb + pos_team_emb + pos_time_emb + pos_attr_emb
        )
        for block in self.transformer.h:
            x = block(x)

        return x

    def forward(self, idx, act=None):
        b, t = idx["obs"].shape  # (batch_size, block_size)
        x = self._get_shared_representation(idx)
        
        # Critic
        x_critic = self.transformer.critic_layers(x.detach())
        x_critic = self.transformer.ln_critic_f(x_critic)
        critic_logits = self.lm_head_critic(x_critic)
        val_logits = critic_logits[:, -1, :]
        val_logits = val_logits.reshape(b, self.config.action_size, -1)

        if act is not None:
            val_logits = val_logits[torch.arange(b), act]

        # Actor
        x_actor = self.transformer.act_layers(x)
        x_actor = self.transformer.ln_actor_f(x_actor)
        logits = self.lm_head_actor(x_actor)
        act_logits = logits[:, -1, :]

        return act_logits, x_critic[:, -1, :]

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.debug(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logger.debug(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        logger.debug(f"using fused AdamW: {use_fused}")

        return optimizer

    def configure_optimizers_2(self, weight_decay, learning_rate, betas, device_type):
        critic_lr = learning_rate
        actor_lr = learning_rate / 5
        shared_lr = min(actor_lr, critic_lr)
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        def get_decay_groups(params):
            decay_params = [p for p in params if p.dim() >= 2]
            nodecay_params = [p for p in params if p.dim() < 2]
            return decay_params, nodecay_params
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # Actor params split
        actor_params = [p for pn, p in param_dict.items() if "act_layers" in pn or "lm_head_actor" in pn or "ln_actor_f" in pn]
        actor_decay, actor_nodecay = get_decay_groups(actor_params)

        # Critic params split
        critic_params = [p for pn, p in param_dict.items() if "critic_layers" in pn or "lm_head_critic" in pn or "ln_critic_f" in pn]
        critic_decay, critic_nodecay = get_decay_groups(critic_params)

        # Shared params: exclude all actor and critic params
        actor_param_ids = set(id(p) for p in actor_params)
        critic_param_ids = set(id(p) for p in critic_params)

        shared_params = [p for pn, p in param_dict.items() if (id(p) not in actor_param_ids) and (id(p) not in critic_param_ids)]

        shared_decay, shared_nodecay = get_decay_groups(shared_params)

        optim_groups = [
            {"params": actor_decay, "lr": actor_lr, "weight_decay": weight_decay},
            {"params": actor_nodecay, "lr": actor_lr, "weight_decay": 0.0},
            {"params": critic_decay, "lr": critic_lr, "weight_decay": weight_decay},
            {"params": critic_nodecay, "lr": critic_lr, "weight_decay": 0.0},
            {"params": shared_decay, "lr": shared_lr, "weight_decay": weight_decay},
            {"params": shared_nodecay, "lr": shared_lr, "weight_decay": 0.0},
        ]

        num_actor_params = sum(p.numel() for p in actor_params)
        num_critic_params = sum(p.numel() for p in critic_params)
        num_shared_params = sum(p.numel() for p in shared_params)
        logger.debug(f"Actor params: {num_actor_params:,}, Critic params: {num_critic_params:,}, Shared params: {num_shared_params:,}")


        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, betas=betas, **extra_args
        )
        logger.debug(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

class SMACEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        gpt_cfg = CriticGPTConfig()
        self.encoder = Critic_GPT(gpt_cfg)

        self.encoder_out_size = gpt_cfg.action_size

        log.debug('Encoder output size: %r', self.encoder_out_size)

    def forward(self, x):
        x = x['obs']
        act_logits, critic_logits = self.encoder(x)

        return act_logits, critic_logits

    def get_out_size(self):
        return self.encoder_out_size

    @staticmethod
    def make_smac_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
        return SMACEncoder(cfg, obs_space)
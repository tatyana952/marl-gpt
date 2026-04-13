import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor
import logging
import numpy as np

log = logging.getLogger(__name__)


from sample_factory.model.actor_critic import ActorCriticSharedWeights, ActorCritic
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.action_parameterization import (
    ActionParameterizationContinuousNonAdaptiveStddev,
)
from sample_factory.model.action_parameterization import ActionsParameterization
from sample_factory.model.encoder import Encoder

from sample_factory.algo.utils.action_distributions import (
    calc_num_action_parameters,
    get_action_distribution,
    is_continuous_action_space,
)
from sample_factory.utils.normalize import ObservationNormalizer

from gpt.finetune_vers.model_gpt import Critic_GPT, CriticGPTConfig


def make_config():
    # TODO FINE-TUNE: change parameters if training from scratch
    block_size = 400
    max_action_size = 11
    history_len = 10

    num_bins = 11
    max_value = 5
    sigma_loss = 0.01

    n_attr = 129
    n_agents = 20
    n_team = 2

    # 7M  (TODO FINE-TUNE: use n_layer=2, n_head=2, n_embd=16 for a 2M debug model)
    n_layer = 8
    n_head = 8
    n_embd = 256

    vocab_size = num_bins
    dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias = False
    action_size = max_action_size
    gamma = 0.99

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
    )

    # Add CriticGPT config as an attribute
    return CriticGPTConfig(**model_args)


def load_model(ckpt_path, device, model_args=None):
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]

    if model_args:
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
    else:
        model_args = checkpoint_model_args

    # create the model
    gptconf = CriticGPTConfig(**model_args)
    model = Critic_GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    return model, gptconf


def init_model(ckpt_path):
    if ckpt_path is not None:
        model, config = load_model(ckpt_path, "cuda")
    else:
        config = make_config()
        model = Critic_GPT(config)
    return model, config


class ActionParameterizationMask(ActionsParameterization):
    """
    Action parameterization that works with CriticGPT output
    """

    def __init__(self, cfg, core_out_size, action_space, action_size):
        super().__init__(cfg, action_space)

        # Use the action size from CriticGPT config
        self.action_size = action_size

        # Linear layer to project from core output to action parameters
        self.num_action_outputs = calc_num_action_parameters(action_space)

    def forward(self, actor_core_output, action_mask=None):
        """Generate action distribution from actor output"""
        action_distribution_params = actor_core_output[:, : self.num_action_outputs]
        if action_mask is not None:
            action_distribution_params[action_mask < 1] = -100

        action_distribution = get_action_distribution(
            self.action_space, raw_logits=action_distribution_params
        )
        return action_distribution_params, action_distribution


class ActorCriticSharedWeightsSMAC(ActorCriticSharedWeights):
    """
    Actor-Critic model that wraps CriticGPT as the core model
    """

    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
        raw_path = getattr(cfg, "gpt_model_path", "out/ckpt.pt")
        model_path = None if raw_path.lower() == "none" else raw_path

        model, self.critic_gpt_config = init_model(model_path)
        super().__init__(model_factory, obs_space, action_space, cfg)

        # Initialize the CriticGPT model as the core
        self.model = model
        # Calculate decoder output size (this is the embedding size from CriticGPT)
        decoder_out_size: int = self.critic_gpt_config.action_size

        # Set up action parameterization
        self.action_parameterization = self.get_action_parameterization(
            decoder_out_size
        )

        # Value head - projects from embedding to single value
        self.critic_linear = nn.Linear(self.critic_gpt_config.n_embd, 1, bias=False)

        self.last_action_mask = None
        self.last_action_distribution = None
        self.last_token = True

        if model_path is not None:
            self.freeze_actor()

    def _prepare_critic_gpt_input(
        self, normalized_obs_dict: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Prepare input dictionary for CriticGPT model
        This needs to be adapted based on how your observations are structured
        """
        obs = normalized_obs_dict["obs"]
        batch_size, obs_dim = obs.shape
        block_size = self.model.config.block_size

        # Reshape observations to match CriticGPT's expected input format
        if obs_dim != block_size:
            # Pad or truncate to match block_size
            obs_reshaped = dict()
            for key, obs in normalized_obs_dict.items():
                if key == "action_mask":
                    continue

                if obs_dim < block_size:
                    padded_obs = torch.zeros(batch_size, block_size, device=obs.device)
                    padded_obs[:, :obs_dim] = obs
                    obs_reshaped[key] = padded_obs
                else:
                    obs_reshaped[key] = obs[:, :block_size]
        else:
            obs_reshaped = normalized_obs_dict

        for key, obs in obs_reshaped.items():
            if key == "obs":
                continue
            obs_reshaped[key] = obs.to(torch.long)

        if self.last_token:
            obs_reshaped["attr_pos"][:, -1] = 128

        return obs_reshaped

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        """Process observations through the encoder part"""
        # Prepare input for CriticGPT
        critic_gpt_input = self._prepare_critic_gpt_input(normalized_obs_dict)

        # Get shared representation from CriticGPT
        shared_repr = self.model._get_shared_representation(critic_gpt_input)

        return shared_repr

    def get_action_parameterization(self, decoder_output_size: int):
        """Set up action parameterization"""
        if not self.cfg.adaptive_stddev and is_continuous_action_space(
            self.action_space
        ):
            action_parameterization = ActionParameterizationContinuousNonAdaptiveStddev(
                self.cfg,
                decoder_output_size,
                self.action_space,
            )
        else:
            action_parameterization = ActionParameterizationMask(
                self.cfg,
                decoder_output_size,
                self.action_space,
                self.critic_gpt_config.action_size,
            )
        return action_parameterization

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool, action_mask=None
    ) -> TensorDict:
        """Generate values and actions from core output"""
        # Critic
        x_critic = self.model.transformer.critic_layers(core_output)
        x_critic = self.model.transformer.ln_critic_f(x_critic)
        # critic_logits = self.model.lm_head_critic(x_critic)
        # val_logits = critic_logits[:, -1, :]

        # Actor
        x_actor = self.model.transformer.act_layers(core_output)
        x_actor = self.model.transformer.ln_actor_f(x_actor)
        logits = self.model.lm_head_actor(x_actor)
        act_logits = logits[:, -1, :]

        # Critic: compute values
        values = self.critic_linear(x_critic)[:, -1, :].squeeze()

        result = TensorDict(values=values)
        if values_only:
            return result

        # Actor: compute action distribution
        action_distribution_params, self.last_action_distribution = (
            self.action_parameterization(act_logits, action_mask)
        )
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(
        self, normalized_obs_dict, rnn_states, values_only=False, action_mask=None
    ) -> TensorDict:
        if "action_mask" in normalized_obs_dict:
            action_mask = normalized_obs_dict.pop("action_mask", None)
        # Get shared representation
        x = self.forward_head(normalized_obs_dict)

        # Process through RNN core if needed (CriticGPT handles this internally)
        x, new_rnn_states = self.forward_core(x, rnn_states)

        # Generate final outputs
        result = self.forward_tail(
            x, values_only, sample_actions=True, action_mask=action_mask
        )
        result["new_rnn_states"] = new_rnn_states

        return result

    def forward_core(self, head_output, rnn_states):
        """
        Core processing - in this case, CriticGPT handles the core processing
        """
        return head_output, rnn_states

    def freeze_actor(self):
        self.model.freeze_actor()

    def unfreeze_actor(self):
        self.model.unfreeze_actor()


def smac_make_actor_critic_func(
    cfg: Config, obs_space: ObsSpace, action_space: ActionSpace
) -> ActorCritic:
    """Factory function to create the actor-critic model"""
    from sample_factory.algo.utils.context import global_model_factory

    model_factory = global_model_factory()

    return ActorCriticSharedWeightsSMAC(model_factory, obs_space, action_space, cfg)

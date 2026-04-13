from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass

import torch
from loguru import logger

from gpt.model_ac import CriticWithLoss as GPT
from gpt.model_ac import CriticGPTConfig as GPTConfig
from utils.env_observation import reshape_obs


@dataclass
class InferenceConfig:
    name: Literal["MARL-GPT"] = "MARL-GPT"
    path_to_weights: Optional[str] = "weights/model-6M.pt"
    device: str = "cuda"
    parallel_backend: str = "balanced_dask"
    num_process: int = 4
    preprocessing: Optional[str] = None
    run_episode_func: str = "default"
    seed: Optional[int] = 0
    model_type: Literal["critic", "actor"] = "actor"

    last_token: bool = False
    env_specific: bool = False
    env_indx: int = 1
    sample_actions: bool = False


def strip_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    """
    strips the given prefix from the keys in the state dictionary
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix) :]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class MARLGPTInference:
    def __init__(self, cfg: InferenceConfig, net=None, gpt_config=None):
        self.cfg: InferenceConfig = cfg

        if (
            self.cfg.device in ["mps", "cuda"] and not torch.cuda.is_available()
            if self.cfg.device == "cuda"
            else not torch.backends.mps.is_available()
        ):
            logger.warning(f"{self.cfg.device} is not available, using cpu instead!")
            self.cfg.device = "cpu"

        if gpt_config is None or net is None:
            checkpoint = torch.load(
                Path(self.cfg.path_to_weights), map_location=self.cfg.device
            )
            model_state_dict = strip_prefix_from_state_dict(checkpoint["model"])
            config_dict = checkpoint.get("model_args")
            gpt_config = GPTConfig(**config_dict)

        if net is not None:
            self.net = net
        else:
            self.net = GPT(gpt_config)
            self.net.load_state_dict(model_state_dict, strict=False)
            self.net.to(self.cfg.device)
            self.net.eval()
        self.block_size = gpt_config.block_size
        self.action_size = gpt_config.action_size
        print(f"block_size = {self.block_size}, action_size = {self.action_size}")

        self._env_indicator = 0
        self._last_tok_attr = gpt_config.n_attr - 1  # last valid embedding index
        if self.cfg.env_specific:
            self._env_indicator = self.cfg.env_indx
        self.last_obs = None

    def add_last_token(self, dict_obs):
        dict_obs["attr_pos"][:, -1] = self._last_tok_attr
        dict_obs["obs"][:, -1] = self._env_indicator

    def act(self, observations):
        # Reshape to max size
        obs_res_dict = reshape_obs(observations, self.block_size, self.action_size)
        self.last_obs = obs_res_dict

        if self.cfg.last_token:
            self.add_last_token(obs_res_dict)

        if "actor" == self.cfg.model_type:
            act = self.net.act_actor(obs_res_dict, self.cfg.sample_actions)
        elif "critic" == self.cfg.model_type:
            act = self.net.act(obs_res_dict, self.cfg.sample_actions)
        else:
            act = self.net.act_actor_critic(obs_res_dict, self.cfg.sample_actions)

        actions = torch.squeeze(act).tolist()
        if not isinstance(actions, list):
            actions = [actions]
        return actions

    def reset_states(self):
        pass

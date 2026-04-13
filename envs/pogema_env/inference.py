from pathlib import Path
from typing import Literal, Optional
import json

import cppimport.import_hook
import torch
from huggingface_hub import hf_hub_download
from pogema_toolbox.algorithm_config import AlgoBase
from pogema_toolbox.registry import ToolboxRegistry
from pydantic import Extra

from gpt.model import GPT, GPTConfig

# from gpt.vector_model import GPT, GPTConfig
from envs.pogema_env.observation_generator import ObservationGenerator, InputParameters
from utils.env_observation import from_vector_to_tensor


class MAPFGPTInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["MAPF-GPT"] = "MAPF-GPT"
    num_agents: int = 13
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    path_to_weights: Optional[str] = "weights/model-6M.pt"
    device: str = "cuda"
    context_size: int = 256
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
    mask_greed_action: bool = False
    repo_id: str = "aandreychuk/MAPF-GPT"
    grid_step: int = 64
    save_cost2go: bool = False
    batch_size: int = 2048
    task_type_id: int = -1
    num_process: int = 8
    vector_observation: bool = True


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


def pad_with_zeros(input_tensor, d):
    a, b = input_tensor.shape
    # Create a new tensor of zeros with shape (a, d)
    padded = torch.zeros((a, d), dtype=input_tensor.dtype, device=input_tensor.device)
    # Copy original values into the first `b` columns
    padded[:, :b] = input_tensor
    return padded


class MAPFGPTInference:
    def __init__(self, cfg: MAPFGPTInferenceConfig, net=None):
        self.cfg: MAPFGPTInferenceConfig = cfg
        self.input_parameters = InputParameters(
            cfg.cost2go_value_limit,
            cfg.num_agents,
            cfg.num_previous_actions,
            cfg.context_size,
            cfg.cost2go_radius,
            cfg.agents_radius,
            cfg.grid_step,
            cfg.save_cost2go,
            cfg.task_type_id,
        )
        self.observation_generator = None
        self.last_actions = None

        path_to_weights = Path(self.cfg.path_to_weights)
        if path_to_weights.name in ["model-2M.pt", "model-6M.pt", "model-85M.pt"]:
            hf_hub_download(
                repo_id=self.cfg.repo_id,
                filename=path_to_weights.name,
                local_dir=path_to_weights.parent,
            )
            ToolboxRegistry.info(
                f"Using weights loaded from huggingface: {path_to_weights}"
            )

        if ("cuda" in self.cfg.device and not torch.cuda.is_available()) or (
            self.cfg.device == "mps" and not torch.backends.mps.is_available()
        ):
            ToolboxRegistry.warning(
                f"{self.cfg.device} is not available, using cpu instead!"
            )
            self.cfg.device = "cpu"

        self.torch_generator = torch.Generator(device=self.cfg.device)
        self.torch_generator.manual_seed(0)

        checkpoint = torch.load(
            Path(self.cfg.path_to_weights), map_location=self.cfg.device
        )

        model_state_dict = strip_prefix_from_state_dict(checkpoint["model"])
        config_dict = checkpoint.get("model_args")
        gpt_config = GPTConfig(**config_dict)
        self.block_size = gpt_config.block_size
        self.act_size = gpt_config.vocab_size
        if net is not None:
            self.net = net
        else:
            self.net = GPT(gpt_config)
            self.net.load_state_dict(model_state_dict, strict=False)
            self.net.to(self.cfg.device)
            self.net.eval()
        self.load_pos_config()

    def load_pos_config(self):
        config_path = "envs/pogema_env/pogema_pos.json"
        with open(config_path, "r") as f:
            self.pos_config = json.load(f)

    def from_tensor_to_dict(self, obs_tensor):
        obs_res_dict = dict()
        obs_res_dict["obs"] = obs_tensor
        obs_res_dict["group_pos"] = from_vector_to_tensor(
            self.pos_config["pos_team"], obs_tensor.shape
        ).to(self.cfg.device)
        obs_res_dict["agent_pos"] = from_vector_to_tensor(
            self.pos_config["pos_agent"], obs_tensor.shape
        ).to(self.cfg.device)
        obs_res_dict["time_pos"] = from_vector_to_tensor(
            self.pos_config["pos_time"], obs_tensor.shape
        ).to(self.cfg.device)
        obs_res_dict["attr_pos"] = from_vector_to_tensor(
            self.pos_config["pos_attr"], obs_tensor.shape
        ).to(self.cfg.device)
        obs_res_dict["action_mask"] = from_vector_to_tensor(
            [1 for _ in range(5)], obs_tensor.shape
        ).to(self.cfg.device)
        return obs_res_dict

    def generate_observations_for_model(self, observations):
        if isinstance(observations[0], dict):
            positions = [obs["global_xy"] for obs in observations]
            goals = [obs["global_target_xy"] for obs in observations]
            if self.observation_generator is None:
                self.observation_generator = ObservationGenerator(
                    observations[0]["global_obstacles"].copy().astype(int).tolist(),
                    self.input_parameters,
                )
                self.observation_generator.create_agents(positions, goals)
                self.last_actions = [-1 for _ in range(len(observations))]
            self.observation_generator.update_agents(
                positions, goals, self.last_actions
            )
            # if self.cfg.vector_observation:
            # inputs = self.observation_generator.generate_vector_observations()
            # else:
            inputs = self.observation_generator.generate_observations()
            inputs = self.observation_generator.encoded_to_vector_observations(inputs)
        else:
            inputs = observations
        obs_tensor = torch.tensor(inputs, device=self.cfg.device)
        obs_res_dict = self.from_tensor_to_dict(obs_tensor)
        for key, input_tensor in obs_res_dict.items():
            if key == "action_mask":
                obs_res_dict[key] = pad_with_zeros(input_tensor, self.act_size)
            else:
                obs_res_dict[key] = pad_with_zeros(input_tensor, self.block_size)
        return obs_res_dict

    def act(self, observations):
        tensor_obs = self.generate_observations_for_model(observations)
        # print(tensor_obs["obs"][0])
        # assert False
        actions = torch.squeeze(self.net.act(tensor_obs)).tolist()
        if not isinstance(actions, list):
            actions = [actions]
        self.last_actions = actions.copy()
        return actions

    def reset_states(self):
        self.observation_generator = None
        self.torch_generator.manual_seed(0)

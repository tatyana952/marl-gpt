import gymnasium as gym
import json

import cppimport.import_hook
import torch
from pogema_toolbox.algorithm_config import AlgoBase
from pydantic import Extra
from envs.pogema_env.observation_generator import ObservationGenerator, InputParameters
from utils.env_observation import from_vector_to_tensor


class MAPFGPTInferenceConfig(AlgoBase, extra=Extra.forbid):
    num_agents: int = 13
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    context_size: int = 256
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
    mask_greed_action: bool = False
    grid_step: int = 64
    save_cost2go: bool = False
    batch_size: int = 2048
    task_type_id: int = -1
    num_process: int = 8
    vector_observation: bool = True


def pad_with_zeros(input_tensor, d):
    a, b = input_tensor.shape
    # Create a new tensor of zeros with shape (a, d)
    padded = torch.zeros((a, d), dtype=input_tensor.dtype, device=input_tensor.device)
    # Copy original values into the first `b` columns
    padded[:, :b] = input_tensor
    return padded


class InferencePogemaWrapper(gym.Wrapper):
    def __init__(self, env, cfg: MAPFGPTInferenceConfig):
        super().__init__(env)
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
        self.torch_generator = torch.Generator(device=self.cfg.device)
        self.torch_generator.manual_seed(0)
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
            inputs = self.observation_generator.generate_observations()
            inputs = self.observation_generator.encoded_to_vector_observations(inputs)
        else:
            inputs = observations
        obs_tensor = torch.tensor(inputs, device=self.cfg.device)
        obs_res_dict = self.from_tensor_to_dict(obs_tensor)
        return obs_res_dict

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        new_observations = self.generate_observations_for_model(observations)
        return new_observations, infos

    def step(self, action):
        observations, rewards, terminated_flags, truncated_flags, infos = self.env.step(
            action
        )
        self.last_actions = action.copy()
        new_observations = self.generate_observations_for_model(observations)
        return new_observations, rewards, terminated_flags, truncated_flags, infos

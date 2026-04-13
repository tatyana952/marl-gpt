from typing import Literal, Optional

import argparse
from pathlib import Path
import yaml
import json

import cppimport.import_hook
import torch
from pogema_toolbox.algorithm_config import AlgoBase
from pogema_toolbox.registry import ToolboxRegistry
from pydantic import Extra

# from gpt.vector_model import GPT, GPTConfig
from observation_generator import ObservationGenerator, InputParameters
import numpy as np

# import torch
from pogema_toolbox.create_env import Environment
from envs.pogema_env.create_env import create_eval_env


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


class MAPFGPTInference:
    def __init__(self, cfg: MAPFGPTInferenceConfig, net=None):
        self.pos_type = torch.int
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

        if ("cuda" in self.cfg.device and not torch.cuda.is_available()) or (
            self.cfg.device == "mps" and not torch.backends.mps.is_available()
        ):
            ToolboxRegistry.warning(
                f"{self.cfg.device} is not available, using cpu instead!"
            )
            self.cfg.device = "cpu"

        self.torch_generator = torch.Generator(device=self.cfg.device)
        self.torch_generator.manual_seed(0)

    def act(self, observations):
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
            if self.cfg.vector_observation:
                inputs = self.observation_generator.generate_vector_observations()
            else:
                inputs = self.observation_generator.generate_observations()
        else:
            inputs = observations
        if len(inputs) > self.cfg.batch_size:
            for i in range(0, len(inputs), self.cfg.batch_size):
                batch_inputs = inputs[i : i + self.cfg.batch_size]
                tensor_obs = torch.tensor(
                    batch_inputs, dtype=torch.long, device=self.cfg.device
                )
        else:
            tensor_obs = torch.tensor(inputs, dtype=torch.long, device=self.cfg.device)

        return tensor_obs

    def pos_embed(self):
        cost_to_go_tokens = (self.cfg.cost2go_radius * 2 + 1) ** 2

        num_agents = self.cfg.num_agents
        history_len: int = self.cfg.num_previous_actions
        stable_agent_feat = 2 + 2 + 1
        one_agent_tokens: int = history_len + stable_agent_feat

        # generate pos for attribute
        one_agent_pos_attr = torch.cat(
            [
                torch.arange(1, stable_agent_feat + 1),  # stable feature
                torch.full(
                    (history_len,), stable_agent_feat + 1, dtype=self.pos_type
                ),  # history (the same token)
            ]
        )
        min_cos_to_go_pos = stable_agent_feat + 2
        pos_attr = torch.cat(
            [
                torch.arange(
                    min_cos_to_go_pos, min_cos_to_go_pos + cost_to_go_tokens
                ),  # cost_to_go
                *[one_agent_pos_attr for _ in range(num_agents)],
            ]
        )

        # generate pos for agent
        agent_indx_pos = (
            torch.arange(num_agents)
            .unsqueeze(1)
            .repeat(1, one_agent_tokens)
            .reshape(-1)
        )
        pos_agent = torch.cat(
            [
                torch.zeros((cost_to_go_tokens,), dtype=self.pos_type),  # cost_to_go
                agent_indx_pos,
            ]
        )

        # generate pos for time
        # Create base pattern: m zeros + descending n-1 to 0
        agent_time_pos = torch.cat(
            (
                torch.zeros(stable_agent_feat, dtype=self.pos_type),
                torch.arange(history_len - 1, -1, -1, dtype=self.pos_type),
            )
        )
        pos_time = torch.cat(
            [
                torch.zeros((cost_to_go_tokens,), dtype=self.pos_type),  # cost_to_go
                agent_time_pos.unsqueeze(0).repeat(num_agents, 1).reshape(-1),
            ]
        )

        # generate pos for team
        pos_team = torch.zeros(pos_attr.shape, dtype=self.pos_type)
        return pos_attr, pos_agent, pos_time, pos_team

    def reset_states(self):
        self.observation_generator = None
        self.torch_generator.manual_seed(0)


def generate_pos():
    parser = argparse.ArgumentParser(description="MAPF-GPT Inference Script")
    parser.add_argument(
        "--animation",
        action="store_false",
        help="Enable animation (default: %(default)s)",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=32,
        help="Number of agents (default: %(default)d)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: %(default)d)"
    )
    parser.add_argument(
        "--map_name",
        type=str,
        default="validation-random-seed-001",
        help="Map name (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: cuda, cpu, mps (default: %(default)s)",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=128,
        help="Maximum episode steps (default: %(default)d)",
    )
    parser.add_argument(
        "--show_map_names",
        action="store_true",
        help="Shows names of all available maps",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["2M", "6M", "85M"],
        default="2M",
        help="Model to use: 2M, 6M, 85M (default: %(default)s)",
    )

    # loading maps from eval folders
    for maps_file in Path("eval_configs").rglob("maps.yaml"):
        with open(maps_file, "r") as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)

    args = parser.parse_args()

    if args.show_map_names:
        for map_ in ToolboxRegistry.get_maps():
            print(map_)
        return

    env_cfg = Environment(
        with_animation=args.animation,
        observation_type="MAPF",
        on_target="nothing",
        map_name=args.map_name,
        max_episode_steps=args.max_episode_steps,
        num_agents=args.num_agents,
        seed=args.seed,
        obs_radius=5,
        collision_system="soft",
    )

    env = create_eval_env(env_cfg)
    algo = MAPFGPTInference(MAPFGPTInferenceConfig(device=args.device))
    algo.reset_states()
    obs, _ = env.reset(seed=env.grid_config.seed)
    res = algo.act(obs)
    res2 = algo.pos_embed()
    return res, res2


def main():
    res, pos = generate_pos()
    pos_attr, pos_agent, pos_time, pos_team = pos
    pos_dict = {
        "pos_attr": pos_attr.tolist(),
        "pos_agent": pos_agent.tolist(),
        "pos_time": pos_time.tolist(),
        "pos_team": pos_team.tolist(),
    }
    with open("pogema_pos.json", "w") as f:
        json.dump(pos_dict, f)


if __name__ == "__main__":
    main()

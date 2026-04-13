import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np
import torch

from envs.smac_env.config import SMACGPTInferenceConfig
from envs.smac_env.decomposer import SC2Decomposer
from envs.smac_env.obs_tokenizer import smacTokenizer
from envs.general_envs_wrapeers import InferenceHistoryWrapper


class SeparatedObs(gym.Wrapper):
    def __init__(
        self,
        env,
        task_name: str,
        np_obs_type: bool = True,
        save_original_obs: bool = False,
        smac_v: int = 2,
        zero_padding: bool = False,
        with_own_coord: bool = False,
    ):
        super().__init__(env)
        self.decomposer = SC2Decomposer(task_name, smac_v, zero_padding, with_own_coord)
        obs_shape = self.decomposer.get_obs_shape()
        original_observation_space = self.observation_space
        self.observation_space = Dict(
            {
                "obs_own": Box(-1, 1, shape=obs_shape["own"]),
                "obs_ally": Box(-1, 1, shape=obs_shape["ally"]),
                "obs_enemy": Box(-1, 1, shape=obs_shape["enemy"]),
                "action_mask": original_observation_space["action_mask"],
            }
        )
        self.save_original_obs = save_original_obs
        if self.save_original_obs:
            self.observation_space["obs"] = original_observation_space["obs"]
        self.np_obs_type = np_obs_type

    def change_observations(self, observations) -> list[dict]:
        new_observations: list[dict] = list()
        for obs in observations:
            own_obs, enemy_feats, ally_feats = self.decomposer.decompose_obs(obs["obs"])
            new_obs = {
                "action_mask": obs["action_mask"],
                "obs_own": own_obs,
                "obs_ally": ally_feats,
                "obs_enemy": enemy_feats,
            }
            if self.save_original_obs:
                new_obs["obs"] = obs["obs"]
            if not self.np_obs_type:
                new_obs = {
                    key: np.array(elem).tolist() for key, elem in new_obs.items()
                }
            new_observations.append(new_obs)
        return new_observations

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        new_observations = self.change_observations(observations)
        return new_observations, infos

    def step(self, action):
        observations, rewards, terminated_flags, truncated_flags, infos = self.env.step(
            action
        )
        new_observations = self.change_observations(observations)
        return new_observations, rewards, terminated_flags, truncated_flags, infos


class InferenceSmacWrapper(InferenceHistoryWrapper):
    def __init__(self, env, cfg: SMACGPTInferenceConfig, pos_dict=None):
        tokenizer = smacTokenizer(cfg.map_type, pos_dict)
        super().__init__(env, cfg, tokenizer)

        # Add race in observation
        race_code = {"protoss": -1, "zerg": 1, "terran": 0}
        self.race_pos_code = 16
        self.map_type = race_code[cfg.map_type]
        self.obs_with_race = cfg.obs_with_race
        self.obs_per_agent = cfg.obs_per_agent

        obs_shape = (env.initial_obs_space*cfg.history_len+1,)

        self.observation_space = Dict({"obs"           : Box(-1, 1, shape=obs_shape),
                                       "group_pos"     : Box(-1, 1, shape=obs_shape),
                                       "agent_pos"     : Box(-1, 1, shape=obs_shape),
                                       "time_pos"      : Box(-1, 1, shape=obs_shape),
                                       "attr_pos"      : Box(-1, 1, shape=obs_shape),
                                       "action_mask": Box(0, 1, shape=(env.initial_act_space,), dtype=np.int8),
                                       })
        self.action_space = Discrete(env.initial_act_space)

    def append_agent_race(self, obs_res_dict: dict):
        device = obs_res_dict["obs"].device
        type_indx = torch.full((obs_res_dict["obs"].shape[0], 1), self.map_type).to(
            device
        )
        obs_res_dict["obs"] = torch.cat((obs_res_dict["obs"], type_indx), dim=-1)

        for key, elem in obs_res_dict.items():
            if "pos" in key:
                pos_type = torch.tensor((0,), dtype=torch.int, device=device)
                obs_res_dict[key] = torch.cat((elem, pos_type))

        obs_res_dict["attr_pos"][-1] = self.race_pos_code

    def separete_obs(self, observation: dict):
        num_agents = observation["obs"].shape[0]
        sep_obs = [dict() for _ in range(num_agents)]
        for key, val in observation.items():
            if key in ("obs", "action_mask"):
                continue
            for obs in sep_obs:
                obs[key] = val.cpu().numpy()
        for key in ("obs", "action_mask"):
            for i, obs in enumerate(sep_obs):
                obs[key] = observation[key][i].cpu().numpy()
        return sep_obs

    def change_observations(self, observations):
        obs_res_dict = super().change_observations(observations)

        # Add agent race
        if self.obs_with_race:
            self.append_agent_race(obs_res_dict)
        if self.obs_per_agent:
            obs_res_dict = self.separete_obs(obs_res_dict)
        return obs_res_dict

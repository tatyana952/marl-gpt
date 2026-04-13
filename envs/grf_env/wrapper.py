import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete

from envs.general_envs_wrapeers import InferenceHistoryWrapper
from envs.grf_env.config import GRFInferenceConfig

from envs.grf_env.obs_tokenizer import grfTokenizer


class InferenceGrfWrapper(InferenceHistoryWrapper):
    def __init__(self, env, cfg: GRFInferenceConfig):
        tokenizer = grfTokenizer("grf")
        super().__init__(env, cfg, tokenizer)

        self.obs_per_agent = cfg.obs_per_agent
        obs_shape = (env.initial_obs_space * cfg.history_len,)
        self.observation_space = Dict(
            {
                "obs": Box(-1, 1, shape=obs_shape),
                "group_pos": Box(-1, 1, shape=obs_shape),
                "agent_pos": Box(-1, 1, shape=obs_shape),
                "time_pos": Box(-1, 1, shape=obs_shape),
                "attr_pos": Box(-1, 1, shape=obs_shape),
                "action_mask": Box(0, 1, shape=(env.initial_act_space,), dtype=np.int8),
            }
        )
        self.action_space = Discrete(env.initial_act_space)

    def _ensure_action_mask(self, observations):
        if "action_mask" in observations[0]:
            return observations
        for agent_obs in observations:
            # 19 legal actions – everything allowed (=1)
            agent_obs["action_mask"] = [1] * 19
        return observations

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
        observations = self._ensure_action_mask(observations)
        obs_res_dict = super().change_observations(observations)
        if self.obs_per_agent:
            obs_res_dict = self.separete_obs(obs_res_dict)
        return obs_res_dict


class LogWrapper(gym.Wrapper):
    def __init__(self, env, metrics_log_name="metrics"):
        super().__init__(env)
        self.final_info = {}
        self.metrics_log_name = metrics_log_name

    def reset(self, **kwargs):
        self.final_info = {}
        observations, infos = self.env.reset(**kwargs)
        return observations, infos

    def step(self, action):
        observations, rewards, terminated_flags, truncated_flags, infos = self.env.step(
            action
        )
        if infos and len(infos) > 0:
            self.final_info = infos[0].get("episode_stats", {})
        if all(terminated_flags) or all(truncated_flags):
            final_score = self.final_info.get("current_score_L_R", [0, 0])
            goal_diff = self.final_info.get(
                "goal_difference", final_score[0] - final_score[1]
            )

            infos[0][self.metrics_log_name] = {
                "goal_diff": goal_diff,
                "final_score": final_score,
                "left_score": final_score[0],
                "right_score": final_score[1],
                **self.final_info,
            }
        return observations, rewards, terminated_flags, truncated_flags, infos

import numpy as np
import gymnasium
from gymnasium.spaces import Discrete, Box, Dict

import gfootball.env as football_env

from envs.grf_env.config import GRFConfig


def make_grf_v0(cfg: GRFConfig):
    env = GRFv0(cfg.model_dump())
    env = AutoResetWrapper(env)
    return env


def sf_make_grf_v0(full_env_name, cfg=None, _env_config=None, render_mode=None):
    return make_grf_v0(GRFConfig.get_scenario_config(cfg.experiment))


class GRFv0(gymnasium.Env):
    def __init__(self, config):
        self.is_multiagent = True
        self._env = football_env.create_environment(**config)

        shape = self._env.observation_space.shape
        if len(self._env.observation_space.shape) > 1:
            self.num_agents = self._env.observation_space.shape[0]
            shape = shape[1:]
        else:
            self.num_agents = 1
        self.observation_space = Dict(
            {
                "obs": Box(-10, 10, shape=shape),
            }
        )
        self.action_space = Discrete(19)
        self._steps = None
        self._previous_game_mode = 0
        self._free_kick_count = 0
        self._goal_kick_count = 0
        self._corner_count = 0
        self._throw_in_count = 0
        self._penalty_count = 0
        self.initial_obs_space = shape[0]
        self.initial_act_space = 19

    def step(self, actions):
        observations, rew, done, info = self._env.step(actions)

        terminated_flags = np.array([done for _ in range(self.num_agents)])
        truncated_flags = np.array([False for _ in range(self.num_agents)])

        # Get current score and game mode from the core environment
        current_raw_score = [0, 0]
        current_game_mode = None
        football_env = self._env.unwrapped
        if hasattr(football_env, "_env") and hasattr(football_env._env, "observation"):
            core_observation = football_env._env.observation()
            if isinstance(core_observation, dict):
                if "score" in core_observation:
                    current_raw_score = core_observation["score"]
                if "game_mode" in core_observation:
                    current_game_mode = core_observation["game_mode"]

        # Track game mode transitions and count set pieces
        if current_game_mode is not None:
            if self._previous_game_mode != current_game_mode:
                if current_game_mode == 2:
                    self._goal_kick_count += 1
                elif current_game_mode == 3:
                    self._free_kick_count += 1
                elif current_game_mode == 4:
                    self._corner_count += 1
                elif current_game_mode == 5:
                    self._throw_in_count += 1
                elif current_game_mode == 6:
                    self._penalty_count += 1
            self._previous_game_mode = current_game_mode

        left_score = current_raw_score[0]
        right_score = current_raw_score[1]
        goal_difference = left_score - right_score

        infos_list = []
        for i in range(self.num_agents):
            agent_stats = {}
            if i == 0:
                agent_stats = info.copy()
                agent_stats["current_score_L_R"] = [left_score, right_score]
                agent_stats["goal_difference"] = goal_difference
                agent_stats["free_kick_count"] = self._free_kick_count
                agent_stats["goal_kick_count"] = self._goal_kick_count
                agent_stats["corner_count"] = self._corner_count
                agent_stats["throw_in_count"] = self._throw_in_count
                agent_stats["penalty_count"] = self._penalty_count
                agent_stats["current_game_mode"] = current_game_mode

            infos_list.append({"episode_stats": agent_stats})

        observations = [
            {
                "obs": obs,
            }
            for obs in self.handle_single_agent_data(observations)
        ]

        return (
            observations,
            self.handle_single_agent_data(rew),
            terminated_flags,
            truncated_flags,
            infos_list,
        )

    def reset(self, *, seed=None, **kwargs):
        if seed is not None:
            import random, numpy as np

            random.seed(seed)
            np.random.seed(seed)

        self._steps = 0
        self._free_kick_count = 0
        self._goal_kick_count = 0
        self._corner_count = 0
        self._throw_in_count = 0
        self._penalty_count = 0

        observations = self._env.reset()
        observations = [
            {
                "obs": obs,
            }
            for obs in self.handle_single_agent_data(observations)
        ]

        return observations, [{} for _ in range(self.num_agents)]

    def handle_single_agent_data(self, data):
        return [data] if self.num_agents == 1 else data


class AutoResetWrapper(gymnasium.Wrapper):
    def step(self, action):
        observations, rewards, terminated_flags, truncated_flags, infos = self.env.step(
            action
        )
        if all(terminated_flags) or all(truncated_flags):
            observations, _ = self.env.reset()

        return observations, rewards, terminated_flags, truncated_flags, infos


def main():
    env = make_grf_v0(GRFConfig.get_scenario_config("academy_run_to_score_with_keeper"))
    # env = make_grf_v0(GRFConfig.get_scenario_config('11vs11a2'))
    observations, info = env.reset()

    terminations = truncations = [False, ...]
    while not (all(terminations) or all(truncations)):
        actions = []
        for obs in observations:
            action = env.action_space.sample()
            actions.append(action)

        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(infos, rewards)


if __name__ == "__main__":
    main()

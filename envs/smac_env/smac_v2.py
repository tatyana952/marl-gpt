from smacv2.env import StarCraftCapabilityEnvWrapper
from smacv2.env import StarCraft2Env
import numpy as np
import gymnasium
from gymnasium.spaces import Discrete, Box, Dict

from envs.smac_env.config import SMAC2ArgsConfig


class SMACv2(gymnasium.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config, smac_v: int = 2):
        if smac_v == 2:
            self._env = StarCraftCapabilityEnvWrapper(**config)
        elif smac_v == 1:
            self._env = StarCraft2Env(**config)
        self._ready_agents = []
        self.observation_space = Dict(
            {
                "obs": Box(-1, 1, shape=(self._env.get_obs_size(),)),
                "action_mask": Box(0, 1, shape=(self._env.get_total_actions(),)),
            }
        )
        self.action_space = Discrete(self._env.get_total_actions())
        self.env_info = self._env.get_env_info()
        self.num_agents = self.env_info["n_agents"]
        self._steps = None
        self.initial_obs_space = self._env.get_obs_size()
        self.initial_act_space = self._env.get_total_actions()

    def step(self, actions):
        actions = self.handle_unavailable_actions(actions)
        rew, done, info = self._env.step(actions)

        self._steps += 1
        observations = self._env.get_obs()
        rewards = [rew / self.num_agents for _ in range(self.num_agents)]

        return_obs = [dict() for _ in range(self.num_agents)]
        for idx, obs in enumerate(observations):
            return_obs[idx] = {
                "action_mask": np.array(self._env.get_avail_agent_actions(idx)),
                "obs": obs,
            }

        # terminated_flags = np.array([bool(return_obs[idx]['action_mask'][0]) for idx in range(self.num_agents)])
        # truncated_flags = np.array([self._steps >= MAX_EPISODE_STEPS for _ in range(self.num_agents)])

        terminated_flags = np.array([done for _ in range(self.num_agents)])
        truncated_flags = np.array([False for _ in range(self.num_agents)])
        infos = [
            {"episode_stats": info} if i == 0 else {} for i in range(self.num_agents)
        ]
        # for idx, info in enumerate(infos):
        #    infos[idx]['is_active'] = not bool(return_obs[idx]['action_mask'][0])

        return return_obs, rewards, terminated_flags, truncated_flags, infos

    def reset(self, seed=None, **kwargs):
        self._steps = 0
        if seed is not None:
            self.seed(seed)

        obs_list, state_list = self._env.reset()
        return_obs = [dict() for _ in range(self.num_agents)]
        for i, obs in enumerate(obs_list):
            return_obs[i] = {
                "action_mask": np.array(self._env.get_avail_agent_actions(i)),
                "obs": obs,
            }

        self._ready_agents = list(range(len(obs_list)))
        return return_obs, {}

    def handle_unavailable_actions(self, actions):
        rng = np.random.default_rng()
        fixed_actions = []

        for idx, action in enumerate(actions):
            available_actions = self._env.get_avail_agent_actions(idx)

            if available_actions[action] == 0:
                valid_actions = np.nonzero(available_actions)[
                    0
                ]  # get indices where action is available
                action = rng.choice(valid_actions)  # pick a random valid action
            fixed_actions.append(action)
        return fixed_actions

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()

    def seed(self, seed):
        # todo check seeding
        pass

    def save_replay(self):
        self._env.save_replay()


class LogSMACWrapper(gymnasium.Wrapper):
    def __init__(self, env, metrics_log_name="metrics"):
        super().__init__(env)
        self.metrics_log_name = metrics_log_name

    def step(self, actions):
        observations, reward, terminated_flags, truncated_flags, infos = self.env.step(
            actions
        )

        if all(terminated_flags) or all(truncated_flags):
            infos[0][self.metrics_log_name] = infos[0]["episode_stats"]

        return observations, reward, terminated_flags, truncated_flags, infos


class AutoResetWrapper(gymnasium.Wrapper):
    def step(self, action):
        observations, rewards, terminated_flags, truncated_flags, infos = self.env.step(
            action
        )
        if all(terminated_flags) or all(truncated_flags):
            observations, _ = self.env.reset()
        return observations, rewards, terminated_flags, truncated_flags, infos


def main():
    env = SMACv2(SMAC2ArgsConfig.get_scenario_config("protoss_5_vs_5").model_dump())
    observations, info = env.reset()

    terminations = truncations = [False, ...]
    while not (all(terminations) or all(truncations)):
        actions = []
        for obs in observations:
            avail_actions = obs["action_mask"]
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.choice(avail_actions_ind)
            actions.append(action)

        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(infos)


if __name__ == "__main__":
    main()

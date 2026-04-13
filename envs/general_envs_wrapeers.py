import gymnasium as gym
from utils.env_observation import stack_obs, stack_dict_tensor


class InferenceHistoryWrapper(gym.Wrapper):
    def __init__(self, env, cfg, tokenizer):
        super().__init__(env)
        self.cfg = cfg
        self.tokenizer = tokenizer

        # History in obs:
        self.history = list()
        self.history_length = (cfg.history_len - 1) * cfg.history_step + 1
        self.history_step = cfg.history_step

    def change_observations(self, observations):
        tensor_obs = stack_obs(observations, self.cfg.device)

        # update history
        self.history.append(tensor_obs)
        self.history = self.history[-self.history_length :]

        # stack history into one observation
        tensor_obs = stack_dict_tensor(
            self.history, self.history_length, self.history_step, self.cfg.device
        )

        # form the observation with pos
        obs_res_dict = dict()
        (
            obs_res_dict["obs"],
            obs_res_dict["group_pos"],
            obs_res_dict["agent_pos"],
            obs_res_dict["time_pos"],
            obs_res_dict["attr_pos"],
        ) = self.tokenizer.get_obs_pos_vectors(tensor_obs)
        obs_res_dict["action_mask"] = tensor_obs["action_mask"]
        return obs_res_dict

    def reset(self, **kwargs):
        self.history = list()
        observations, infos = self.env.reset(**kwargs)
        new_observations = self.change_observations(observations)
        return new_observations, infos

    def step(self, action):
        observations, rewards, terminated_flags, truncated_flags, infos = self.env.step(
            action
        )
        new_observations = self.change_observations(observations)
        return new_observations, rewards, terminated_flags, truncated_flags, infos

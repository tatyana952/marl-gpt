import glob
import os
import torch

from dataclasses import dataclass


def build_masked_history(
    obs: torch.Tensor,
    dones: torch.Tensor,
    indxs: torch.Tensor,
    h_len: int,
    h_step: int = 1,
) -> torch.Tensor:
    """
    Args:
        obs: (n, ...) input tensor
        dones: (n,) boolean mask tensor
        indxs: (batch_size,) indices to process
        h_len: history length => (obs_{t - h_len + 1}, ..., obs_t)
        h_step: timestep between two obs (..., obs_{t - h_step}, obs_t)
    Returns:
        W: (batch_size, h_len, m, k) masked output tensor
    """
    n = obs.shape[0]
    batch_size = indxs.shape[0]
    device = obs.device

    # 1. Create history indices with clamping and validity mask
    h = (h_len - 1) * h_step + 1
    offsets = torch.arange(-h + 1, 1, device=device)  # (h,)
    history_indices = indxs.view(-1, 1) + offsets  # (batch_size, h)

    # Handle negative indices (set to 0 but mark as invalid)
    valid_mask = history_indices >= 0
    history_indices = history_indices.clamp(min=0, max=n - 1)

    # 2. Gather history for obs and dones
    H = obs[history_indices]  # (batch_size, h, m, k)
    dones_hist = dones[history_indices]  # (batch_size, h)

    # 3. Find earliest True in history (excluding last element)
    dones_hist_excl_last = dones_hist[:, :-1]  # (batch_size, h-1)
    positions = torch.arange(h - 1, device=device).view(1, -1)  # (1, h-1)

    # Find first True position (or h if none)
    true_positions = torch.where(dones_hist_excl_last, positions, h)
    earliest_true_pos, _ = true_positions.min(dim=1)  # (batch_size,)
    earliest_true_pos[earliest_true_pos >= h] = -1

    # 4. Create mask for valid elements
    pos_grid = torch.arange(h, device=device).view(1, -1)  # (1, h)
    mask = pos_grid > earliest_true_pos.view(-1, 1)  # (batch_size, h)

    # Combine with validity mask and expand for broadcasting
    mask = (mask & valid_mask).float()
    mask = mask[(...,) + (None,) * (len(H.shape) - len(mask.shape))]

    # 5. Apply mask => shape = (batch_size, h, m, k)
    res = H * mask

    # 6. the first obs in history = the last obs
    res = torch.flip(res, dims=[1])

    # 7. Use h_step: (..., obs_{t - 1}, obs_t) => (..., obs_{t - h_step}, obs_t)
    if h_step > 1:
        indx_with_h_step = torch.arange(0, h, h_step)
        res = res[:, indx_with_h_step]

    return res  # shape = (batch_size, h_len, m, k)


def one_group_one_reward(tensor, indxs):
    max_episode_len = 0
    max_episode = None
    for first_indx, last_indx in indxs:
        episode_len = last_indx - first_indx
        if episode_len > max_episode_len:
            max_episode = tensor[first_indx:last_indx]
            max_episode_len = episode_len

    for first_indx, last_indx in indxs:
        episode_len = last_indx - first_indx
        if max_episode.shape[0] > episode_len:
            tensor[first_indx:last_indx] = max_episode[:episode_len]


def calculate_rtg_and_success(
    dataset, gamma, num_agents, group_reward: bool = False, q_n_step: int = 0
):
    dones = dataset["dones"].to(torch.bool)
    rewards = dataset["rewards"].to(torch.float)

    n = dones.shape[0]
    gamma_pow = torch.zeros((n,))
    standart_rtg = torch.zeros((n,))

    gamma_step = 0
    if q_n_step > 0:
        buffer = torch.zeros(q_n_step, device=rewards.device)
        weights = gamma ** torch.arange(q_n_step, device=rewards.device)
    else:
        prev_rew = 0

    for done_indx in reversed(range(n)):
        if dones[done_indx]:
            gamma_step = 1

            if q_n_step > 0:
                buffer.zero_()
            else:
                prev_rew = 0
        else:
            gamma_step = gamma * gamma_step

        # Calculate standart RTG:
        if q_n_step > 0:
            buffer[1:] = buffer[:-1].clone()
            buffer[0] = rewards[done_indx]
            standart_rtg[done_indx] = (buffer * weights).sum()
        else:
            prev_rew = rewards[done_indx] + gamma * prev_rew
            standart_rtg[done_indx] = prev_rew

        # Calculate gamma_pow:
        gamma_pow[done_indx] = gamma_step

    # Calculate gamma_pow for success:
    if "info_battle_won" in dataset:
        gamma_pow = gamma_pow * dataset["info_battle_won"]

    if group_reward:
        agent_indx = 0
        last_indx = n
        list_of_indxs = list()
        for done_indx in reversed(range(n - 1)):
            if dones[done_indx]:
                # Collect indxs agent from one group
                agent_indx += 1
                list_of_indxs.append((done_indx + 1, last_indx))
                last_indx = done_indx + 1

                if agent_indx == num_agents:
                    one_group_one_reward(gamma_pow, list_of_indxs)
                    one_group_one_reward(standart_rtg, list_of_indxs)
                    agent_indx = 0
                    list_of_indxs = list()
    print(
        f"RTG: {standart_rtg.min()} --> {standart_rtg.mean()} --> {standart_rtg.max()}"
    )
    print(f"SR: {gamma_pow.min()} --> {gamma_pow.mean()} --> {gamma_pow.max()}")
    return standart_rtg, gamma_pow


def get_equal_part_ones_indexes(A: torch.Tensor, part_size: int):
    num_elements = A.numel()
    num_parts = round(num_elements / part_size)
    one_indices = torch.nonzero(A, as_tuple=False).squeeze()
    n = one_indices.numel()
    num_parts = min(num_parts, n)
    one_indices = torch.cat((torch.tensor([0]), one_indices))
    chosen_indices = torch.linspace(0, n, steps=num_parts + 1)[1:-1].round().long()
    selected = one_indices[chosen_indices] + 1
    selected = torch.cat((torch.tensor([0]), selected, torch.tensor([num_elements])))
    return selected.tolist()


class HistoryParameterIterator:
    def __init__(self, h_min, h_max, h_step, h_change):
        self.h_min = h_min
        self.h_max = h_max
        self.h_step = h_step
        self.h_change = h_change
        self._repeat_main = 6
        self._build_sequence()
        self._index = 0

    def _build_sequence(self):
        if not self.h_change:
            # Always return h_max
            self.sequence = [self.h_max]
        else:
            # Build [h_min, h_min + h_step, ..., h_max, h_min, h_min + h_step, ...]
            steps = []
            val = self.h_min
            while val <= self.h_max:
                steps.append(val)
                val += self.h_step
            if steps[-1] != self.h_max:
                steps[-1] = self.h_max
            for _ in range(self._repeat_main):
                steps.append(self.h_max)
            self.sequence = steps
        self._len_sequence = len(self.sequence)

    def next(self):
        ans = self.sequence[self._index]
        self._index = (self._index + 1) % self._len_sequence
        return ans

    def reset(self):
        self._index = 0


@dataclass
class ObservationProperty:
    map_type: str = ""
    q_n_step: int = 0
    history_len: int = 1
    history_step: int = 1

    calculate_rtg: bool = True
    gamma: float = 0.95
    num_agents: int = 5
    group_reward: bool = True
    last_reward_is_value: bool = False

    several_history_len: bool = False
    history_len_step: int = 1
    history_len_min: int = 2

    part_of_dataset: int = 1

    load_part_file: bool = False


class HistoryDataLoader:
    def __init__(
        self,
        config,
        batch_size,
        file_paths=None,
        device="cpu",
        folder_path=None,
        tokenizer=None,
        file_type: str = "*.pt",
    ):
        self.file_paths = file_paths
        if folder_path is not None:
            self.file_paths = sorted(glob.glob(os.path.join(folder_path, file_type)))
            assert len(self.file_paths) > 0
            if config.part_of_dataset > 1:
                num_files = len(self.file_paths) 
                part_of_dataset = max(num_files // config.part_of_dataset, 1)
                self.file_paths = self.file_paths[:part_of_dataset]
                print(f"Use {part_of_dataset} files from dataset (max={num_files}, part={config.part_of_dataset})")
        self.target_tensors = None
        self.input_dict_tensor = dict()
        self.batch_size = batch_size
        self.drop_last = True
        self.device = device
        self.num_epoch = 0

        self._h_iterator = HistoryParameterIterator(
            config.history_len_min,
            config.history_len,
            config.history_len_step,
            config.several_history_len,
        )
        self.history_len = config.history_len
        self.history_step = config.history_step
        self.tokenizer = tokenizer

        self._load_part_file = config.load_part_file
        self._part_size = 2000000
        self._indx_part = 1
        self._all_indx_part = None

    def part_file_loading(self, filename):
        if (self._all_indx_part is None) or (
            self._indx_part >= len(self._all_indx_part)
        ):
            self._indx_part = 1
            self.original_input_dict_tensor = torch.load(filename)
            self._all_indx_part = get_equal_part_ones_indexes(self.original_input_dict_tensor["dones"], self._part_size)
        self.input_dict_tensor = {
            key: val[self._all_indx_part[self._indx_part - 1]:self._all_indx_part[self._indx_part]]
            for key, val in self.original_input_dict_tensor.items()
        }
        self.indices = torch.randperm(self.input_dict_tensor["actions"].shape[0])
        self._indx_part += 1

    def load_and_transfer_data_file(self, filename):
        if self._load_part_file:
            self.part_file_loading(filename)
        else:
            self.input_dict_tensor = torch.load(filename)
            self.indices = torch.randperm(self.input_dict_tensor["actions"].shape[0])

    def _build_with_history(self, indxs, target, dones):
        return dict()  # TODO

    def get_data_by_indx_to_dict(self, first_indx, t: int = 0):
        indxs = self.indices[first_indx : first_indx + self.batch_size]
        if t > 0:
            indxs = torch.clamp(indxs + t, min=0, max=self.indices.shape[0] - 1)
        target = self.input_dict_tensor["actions"][indxs].to(self.device)
        dones = self.input_dict_tensor["dones"]
        self.history_len = self._h_iterator.next()
        obs_dict = self._build_with_history(indxs, target, dones)
        return obs_dict, target

    def _build_action_mask(self, obs_res_dict, obs_dict):
        pass  # TODO

    def get_data_by_indx_to_obs_and_pos(self, first_indx, t: int = 0):
        obs_dict, target = self.get_data_by_indx_to_dict(first_indx, t)
        obs_res_dict = dict()
        (
            obs_res_dict["obs"],
            obs_res_dict["group_pos"],
            obs_res_dict["agent_pos"],
            obs_res_dict["time_pos"],
            obs_res_dict["attr_pos"],
        ) = self.tokenizer.get_obs_pos_vectors(obs_dict)
        self._build_action_mask(obs_res_dict, obs_dict)
        return obs_res_dict, target

    def __iter__(self):
        while True:
            for file_path in self.file_paths:
                self.load_and_transfer_data_file(file_path)
                for i in range(0, len(self.indices), self.batch_size):
                    if self.drop_last and (i + self.batch_size > len(self.indices)):
                        continue
                    yield self.get_data_by_indx_to_obs_and_pos(i)
            self.num_epoch += 1


class CriticDataLoader:
    def __init__(
        self,
        config,
        batch_size,
        file_paths=None,
        device="cpu",
        folder_path=None,
        dataloader_class: HistoryDataLoader = None,
    ):
        self.dataloader = dataloader_class(
            config, batch_size, file_paths, device, folder_path
        )
        self.gamma = config.gamma
        self.num_agents = config.num_agents
        self.calculate_rtg = config.calculate_rtg
        self.group_reward = config.group_reward
        self.q_n_step = config.q_n_step
        self.last_reward_is_value = config.last_reward_is_value
        self.num_epoch = 0

    def load_and_transfer_data_file(self, filename):
        self.dataloader.load_and_transfer_data_file(filename)
        standart_rtg, success_rtg = calculate_rtg_and_success(
            self.dataloader.input_dict_tensor,
            self.gamma,
            self.num_agents,
            self.group_reward,
            self.q_n_step,
        )
        if self.calculate_rtg:
            self.dataloader.input_dict_tensor["standart_rtg"] = standart_rtg
            self.dataloader.input_dict_tensor["success_rtg"] = success_rtg

    def add_extra_information_for_critic(self, first_indx):
        extra_info_dict = dict()
        indxs = self.dataloader.indices[
            first_indx : first_indx + self.dataloader.batch_size
        ]
        if self.calculate_rtg:
            extra_info_dict["standart_rtg"] = self.dataloader.input_dict_tensor[
                "standart_rtg"
            ][indxs].to(self.dataloader.device)
            extra_info_dict["success_rtg"] = self.dataloader.input_dict_tensor[
                "success_rtg"
            ][indxs].to(self.dataloader.device)
        extra_info_dict["rewards"] = self.dataloader.input_dict_tensor["rewards"][
            indxs
        ].to(self.dataloader.device)
        extra_info_dict["dones"] = self.dataloader.input_dict_tensor["dones"][indxs].to(
            self.dataloader.device
        )
        if self.last_reward_is_value:
            extra_info_dict["rewards"][extra_info_dict["dones"]] = extra_info_dict[
                "standart_rtg"
            ][extra_info_dict["dones"]]
        return extra_info_dict

    def get_data_by_indx_to_obs_and_pos(self, first_indx):
        obs_res_dict, target = self.dataloader.get_data_by_indx_to_obs_and_pos(
            first_indx, t=0
        )
        obs_res_dict_next, _ = self.dataloader.get_data_by_indx_to_obs_and_pos(
            first_indx, t=1
        )
        extra_info_dict = self.add_extra_information_for_critic(first_indx)
        return obs_res_dict, target, obs_res_dict_next, extra_info_dict

    def __iter__(self):
        while True:
            for file_path in self.dataloader.file_paths:
                self.load_and_transfer_data_file(file_path)
                for i in range(
                    0, len(self.dataloader.indices), self.dataloader.batch_size
                ):
                    if self.dataloader.drop_last and (
                        i + self.dataloader.batch_size > len(self.dataloader.indices)
                    ):
                        continue
                    yield self.get_data_by_indx_to_obs_and_pos(i)
            self.num_epoch += 1


def main():
    config = ObservationProperty(history_len=3)
    dl = HistoryDataLoader(
        config, 4, folder_path="dataset_1/clean_data/clean_data_p5_300k"
    )
    data = iter(dl)
    obs, target = next(data)
    print("----- Shapes -------")
    for key in obs.keys():
        print(f"{key}: {obs[key].shape}")


if __name__ == "__main__":
    main()

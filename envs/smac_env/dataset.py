import json
import torch

from dataclasses import dataclass
from envs.smac_env.obs_tokenizer import smacTokenizer
from utils.dataset import (
    build_masked_history,
    ObservationProperty,
    HistoryDataLoader,
    CriticDataLoader,
)
from utils.aggregated_data_loader import AggregateDataset


def one_tr_data(traj):
    actions = torch.tensor(traj["actions"]).reshape(-1)
    indices = torch.randperm(actions.shape[0])
    actions = actions[indices]

    keys = traj["observations"][0][0].keys()

    obs_dict = {key: list() for key in keys}
    for traj_step in traj["observations"][:-1]:
        for traj_agent in traj_step:
            for key in keys:
                obs_dict[key].append(traj_agent[key])

    obs_dict = {key: torch.tensor(elem)[indices] for key, elem in obs_dict.items()}

    return obs_dict, actions


def read_all_traj(dataset):
    # num_traj
    # - ['observations',
    #  - num_steps+1, num_agents
    #   - keys: obs_...
    # - 'actions',
    #  - num_steps, num_agents
    # - 'info']

    obs_dict = {
        "action_mask": list(),
        "obs_own": list(),
        "obs_ally": list(),
        "obs_enemy": list(),
    }
    actions = list()
    for traj in dataset:
        traj_obs_dict, traj_actions = one_tr_data(traj)
        actions.append(traj_actions)
        for key, elem in obs_dict.items():
            elem.append(traj_obs_dict[key])

    actions = torch.cat(actions, dim=0)
    for key in obs_dict.keys():
        obs_dict[key] = torch.cat(obs_dict[key], dim=0)

    # Try to filter die agent

    # indices = obs_dict["action_mask"][:, 0] == 0
    # for key in obs_dict.keys():
    #     obs_dict[key] = obs_dict[key][indices]
    # actions = actions[indices]

    return obs_dict, actions


def get_dict_obs_size(obs_data):
    obs_size = dict()
    for key, elem in obs_data.items():
        obs_size[key] = elem.shape[1:]
    return obs_size


def get_block_size(obs_data):
    block_size: int = 0
    for key, elem in obs_data.items():
        if key == "action_mask":
            continue
        block_size += elem[0].numel()
    return block_size


def get_block_and_obs_size(obs_data):
    block_size = get_block_size(obs_data)
    obs_size = get_dict_obs_size(obs_data)
    return block_size, obs_size


def read_data(filename):
    with open(filename) as f:
        dataset = json.load(f)
    return read_all_traj(dataset)


def load_json(filename):
    with open(filename) as f:
        dataset = json.load(f)
    input_dict_tensor, target_tensors = read_all_traj(dataset)
    print(f"Load data from: {filename}")
    return input_dict_tensor, target_tensors


def load_torch(filename):
    loaded_data = torch.load(filename)

    actions = loaded_data["actions"]

    # Shuffle
    indices = torch.randperm(actions.shape[0])
    actions = actions[indices]

    # Obs
    obs_dict = dict()
    obs_keys = ["action_mask", "obs_own", "obs_ally", "obs_enemy"]
    for key in obs_keys:
        obs_dict[key] = loaded_data[key]
    return obs_dict, actions


@dataclass
class ObservationPropertySmac(ObservationProperty):
    map_type: str = "protoss"
    obs_with_race: bool = True


class HistoryDataLoaderSmac(HistoryDataLoader):
    def __init__(
        self,
        config: ObservationPropertySmac,
        batch_size,
        file_paths=None,
        device="cpu",
        folder_path=None,
    ):
        tokenizer = smacTokenizer(config.map_type)
        super().__init__(config, batch_size, file_paths, device, folder_path, tokenizer)

        self.obs_keys = ["obs_own", "obs_ally", "obs_enemy"]
        race_code = {"protoss": -1, "zerg": 1, "terran": 0}
        self.race_pos_code = 16
        self.map_type = race_code[config.map_type]
        self.obs_with_race = config.obs_with_race

    def _build_with_history(self, indxs, target, dones):
        obs_dict = dict()
        for key in self.obs_keys:
            obs = self.input_dict_tensor[key]
            obs_dict[key] = build_masked_history(
                obs, dones, indxs, self.history_len
            ).to(self.device)
        obs_dict["action_mask"] = self.input_dict_tensor["action_mask"][indxs].to(
            self.device
        )
        return obs_dict

    def append_agent_race(self, obs_res_dict: dict):
        type_indx = torch.full((obs_res_dict["obs"].shape[0], 1), self.map_type).to(
            self.device
        )
        obs_res_dict["obs"] = torch.cat((obs_res_dict["obs"], type_indx), dim=-1)

        for key, elem in obs_res_dict.items():
            if "pos" in key:
                pos_type = torch.tensor((0,), dtype=torch.int, device=self.device)
                obs_res_dict[key] = torch.cat((elem, pos_type))

        obs_res_dict["attr_pos"][-1] = self.race_pos_code

    def _build_action_mask(self, obs_res_dict, obs_dict):
        obs_res_dict["action_mask"] = obs_dict["action_mask"]
        if self.obs_with_race:
            self.append_agent_race(obs_res_dict)


class AggregateDatasetSMAC(AggregateDataset):
    def __init__(
        self,
        folder_paths: list[str],
        map_types: list[str],
        device: str,
        batch_sizes: list[int],
        general_config: dict,
        max_block_size: int = 0,
        max_action_size: int = 0,
    ):
        super().__init__(
            folder_paths,
            device,
            batch_sizes,
            general_config,
            max_block_size,
            max_action_size,
        )
        self._init_datasets(folder_paths, batch_sizes, general_config, map_types)

    def _init_datasets(self, folder_paths, batch_sizes, general_config, map_types):
        for folder_path, batch_size, map_type in zip(
            folder_paths, batch_sizes, map_types
        ):
            config = ObservationPropertySmac(
                map_type=map_type,
                history_len=general_config["history_len"],
                history_step=general_config["history_step"],
            )
            dataset = HistoryDataLoaderSmac(
                config=config,
                batch_size=batch_size,
                folder_path=folder_path,
                device=self.device,
            )
            self.datasets.append(dataset)


class AggregateCriticDatasetSMAC(AggregateDataset):
    def __init__(
        self,
        folder_paths: list[str],
        map_types: list[str],
        device: str,
        batch_sizes: list[int],
        general_config: dict,
        max_block_size: int = 0,
        max_action_size: int = 0,
    ):
        super().__init__(
            folder_paths,
            device,
            batch_sizes,
            general_config,
            max_block_size,
            max_action_size,
        )
        self._init_datasets(folder_paths, batch_sizes, general_config, map_types)
        if self.env_specific:
            self._env_indicator = 1

    def _init_datasets(self, folder_paths, batch_sizes, general_config, map_types):
        for folder_path, batch_size, map_type in zip(
            folder_paths, batch_sizes, map_types
        ):
            config = ObservationPropertySmac(
                map_type=map_type,
                history_len=general_config["history_len"],
                history_step=general_config["history_step"],
                last_reward_is_value=True,
                q_n_step=general_config.get("q_n_step", 1),
                part_of_dataset=general_config.get("part_of_dataset", 1),
                several_history_len=general_config.get("several_history_len", False),
            )
            dataset = CriticDataLoader(
                config=config,
                batch_size=batch_size,
                folder_path=folder_path,
                device=self.device,
                dataloader_class=HistoryDataLoaderSmac,
            )
            self.datasets.append(dataset)


def main_aggr():
    folder_paths = [
        "dataset_1/clean_data/clean_data_p5_300k",
        "dataset_1/clean_data/zerg_5_vs_5",
    ]
    map_types = ["protoss", "zerg"]
    batch_sizes = [4, 4]
    config = {"history_len": 3, "history_step": 1}
    aggregated_dataset = AggregateDatasetSMAC(
        folder_paths=folder_paths,
        map_types=map_types,
        device="cuda:0",
        batch_sizes=batch_sizes,
        general_config=config,
    )
    data = iter(aggregated_dataset)
    qx, qy = next(data)
    for key in qx.keys():
        print(f"{key}: {qx[key].shape}")


def main():
    config = ObservationPropertySmac(history_len=3)
    dl = HistoryDataLoaderSmac(
        config, 4, folder_path="dataset_1/clean_data/clean_data_p5_300k"
    )
    data = iter(dl)
    obs, target = next(data)
    print("----- Shapes -------")
    for key in obs.keys():
        print(f"{key}: {obs[key].shape}")


if __name__ == "__main__":
    main()

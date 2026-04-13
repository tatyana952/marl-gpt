import json
import random
import os
import torch
import pyarrow as pa
import numpy as np

from utils.dataset import ObservationProperty, HistoryDataLoader, CriticDataLoader
from utils.aggregated_data_loader import AggregateDataset
from utils.env_observation import from_vector_to_tensor


class HistoryDataLoaderPogema(HistoryDataLoader):
    def __init__(
        self,
        config,
        batch_size,
        file_paths=None,
        device="cpu",
        folder_path=None,
    ):
        super().__init__(
            config,
            batch_size,
            file_paths,
            device,
            folder_path,
            tokenizer=None,
            file_type="*.arrow",
        )

        self.target_dtype = torch.int8
        self.input_dtype = torch.float16

        if device != "cpu":
            ddp_local_rank = int(device.split(":")[-1])
            ddp_world_size = os.environ.get("WORLD_SIZE")
        else:
            ddp_local_rank = None
        random.shuffle(self.file_paths)

        if (
            "validation" not in folder_path
            and ddp_local_rank is not None
            and ddp_world_size is not None
        ):
            self.file_paths = self.file_paths[
                int(ddp_local_rank) :: int(ddp_world_size)
            ]
        self.load_pos_config()

    def load_pos_config(self):
        config_path = "envs/pogema_env/pogema_pos.json"
        with open(config_path, "r") as f:
            self.pos_config = json.load(f)

    def output_tensor_with_pos(self, obs_tensor):
        obs_res_dict = dict()
        obs_res_dict["obs"] = obs_tensor
        obs_res_dict["group_pos"] = from_vector_to_tensor(
            self.pos_config["pos_team"], obs_tensor.shape
        ).to(self.device)
        obs_res_dict["agent_pos"] = from_vector_to_tensor(
            self.pos_config["pos_agent"], obs_tensor.shape
        ).to(self.device)
        obs_res_dict["time_pos"] = from_vector_to_tensor(
            self.pos_config["pos_time"], obs_tensor.shape
        ).to(self.device)
        obs_res_dict["attr_pos"] = from_vector_to_tensor(
            self.pos_config["pos_attr"], obs_tensor.shape
        ).to(self.device)
        obs_res_dict["action_mask"] = from_vector_to_tensor(
            [1 for _ in range(5)], obs_tensor.shape
        ).to(self.device)
        return obs_res_dict

    @staticmethod
    def _generate_dones(num_elements, num_steps):
        dones = torch.zeros(num_elements, dtype=torch.bool)
        dones[torch.arange(1, num_elements + 1) % num_steps == 0] = True
        return dones

    def load_and_transfer_data_file(self, filename):
        self.input_dict_tensor = dict()

        with pa.memory_map(filename) as source:
            table = pa.ipc.open_file(source).read_all()

        self.input_dict_tensor["obs"] = torch.tensor(
            np.stack(table["input_tensors"].to_numpy()), dtype=self.input_dtype
        )
        self.input_dict_tensor["actions"] = torch.tensor(
            table["gt_actions"].to_numpy(), dtype=self.target_dtype
        )
        self.input_dict_tensor["rewards"] = (
            torch.tensor(table["rewards"].to_numpy(), dtype=self.input_dtype) / 5
        )  # TODO
        self.input_dict_tensor["dones"] = self._generate_dones(
            self.input_dict_tensor["actions"].shape[0], 256
        )

        self.indices = torch.randperm(self.input_dict_tensor["actions"].shape[0])

    def get_data_by_indx_to_obs_and_pos(self, first_indx, t: int = 0):
        indxs = self.indices[first_indx : first_indx + self.batch_size]
        if t > 0:
            indxs = torch.clamp(indxs + t, min=0, max=self.indices.shape[0] - 1)
        target = self.input_dict_tensor["actions"][indxs].to(self.device)
        obs_dict = self.output_tensor_with_pos(
            self.input_dict_tensor["obs"][indxs].to(self.device)
        )
        return obs_dict, target


class AggregateDatasetPogema(AggregateDataset):
    def __init__(
        self,
        folder_paths: list[str],
        device: str,
        batch_sizes: list[int],
        general_config: dict,
        max_block_size: int = 0,
        max_action_size: int = 0,
    ):
        """
        Aggregates datasets from multiple folders into a single dataset.

        Args:
            folder_paths (list): List of folder paths containing datasets.
            map_types (list): List of map types
            device (str): Device to load the data onto (e.g., 'cuda:0').
            batch_sizes (list): List of batch sizes for each dataset.
            general_config (dict): Dict with general properties.
        """
        assert len(folder_paths) == len(
            batch_sizes
        ), "Each dataset must have a corresponding batch size and map types."

        super().__init__(
            folder_paths,
            device,
            batch_sizes,
            general_config,
            max_block_size,
            max_action_size,
        )
        self._init_datasets(folder_paths, batch_sizes, general_config)
        if self.env_specific:
            self._env_indicator = 2

    def _init_datasets(self, folder_paths, batch_sizes, general_config):
        for folder_path, batch_size in zip(folder_paths, batch_sizes):
            config = ObservationProperty(
                num_agents=0,
                group_reward=False,
                q_n_step=general_config.get("q_n_step", 0),
                part_of_dataset=general_config.get("part_of_dataset", 1),
            )
            dataset = CriticDataLoader(
                config=config,
                batch_size=batch_size,
                folder_path=folder_path,
                device=self.device,
                dataloader_class=HistoryDataLoaderPogema,
            )
            self.datasets.append(dataset)


def main_v1():
    config = ObservationProperty(history_len=3)
    dl = HistoryDataLoaderPogema(
        config, 3, folder_path="/marl-gpt/dataset_1/clean_data/pogema_ll/mazes"
    )
    data = iter(dl)
    obs, target = next(data)
    print("----- Shapes -------")
    for key in obs.keys():
        if obs[key] is not None:
            print(f"{key}: {obs[key].shape}")


def main():
    folder_paths = [
        "/marl-gpt/dataset_1/clean_data/pogema_ll/mazes",
        "/marl-gpt/dataset_1/clean_data/pogema_ll/mazes",
    ]
    batch_sizes = [4, 4]
    config = {"mask_ratio": 0.5, "rand_act": True}
    aggregated_dataset = AggregateDatasetPogema(
        folder_paths=folder_paths,
        device="cuda:0",
        batch_sizes=batch_sizes,
        general_config=config,
    )
    data = iter(aggregated_dataset)
    res = next(data)
    for key in res[0].keys():
        print(f"{key}: {res[0][key].shape}")


if __name__ == "__main__":
    main_v1()

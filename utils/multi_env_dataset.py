import torch
from utils.aggregated_data_loader import concat_and_fill
from envs.smac_env.dataset import AggregateCriticDatasetSMAC
from envs.grf_env.dataset import AggregateCriticDatasetGRF

info_datasets = {
    "smac": {
        "config": {"history_len": 3, "history_step": 1},
        "map_types": [],
        "folder_paths": [],
    },
    "pogema": {
        "folder_paths": [],
    },
    "grf": {
        "folder_paths": [],
    },
}


def compute_batch_sizes(dataset_config, batch_size):
    # Count how many values are not equal to V
    N = sum(1 for val in dataset_config.values() if len(val["folder_paths"]) > 0)

    # Avoid division by zero
    if N == 0:
        raise ValueError("Not enough information about folders")

    # Create new dictionary
    batch_sizes_dict = {
        key: 0 if len(val["folder_paths"]) == 0 else (batch_size // N)
        for key, val in dataset_config.items()
    }

    return batch_sizes_dict


class MultiEnvAggregateDataset:
    def __init__(
        self,
        batch_size: int,
        dataset_config: dict,
        device: str,
        max_block_size: int,
        max_action_size: int,
    ):
        self.device: str = device
        self.batch_sizes = compute_batch_sizes(dataset_config, batch_size)
        self.dataloaders = dict()
        if "smac" in self.batch_sizes:
            self.init_smac_dataset(
                self.batch_sizes["smac"],
                dataset_config,
                max_block_size,
                max_action_size,
            )

        if "pogema" in self.batch_sizes:
            self.init_pogema_dataset(
                dataset_config,
                self.batch_sizes["pogema"],
                max_block_size,
                max_action_size,
            )
        if "grf" in self.batch_sizes:
            self.init_grf_dataset(
                dataset_config, self.batch_sizes["grf"], max_block_size, max_action_size
            )
        self.max_block_size = max_block_size
        self.max_action_size = max_action_size

    def init_smac_dataset(
        self, batch_size, dataset_config, max_block_size: int, max_action_size: int
    ):
        if ("smac" not in dataset_config) or (
            len(dataset_config["smac"]["folder_paths"]) == 0
        ):
            print("Not enough information about smac dataset")
            return
        smac_config = dataset_config["smac"]
        num_folders = len(smac_config["map_types"])
        batch_sizes = [int(batch_size // num_folders) for _ in range(num_folders)]
        self.batch_sizes["smac"] = sum(batch_sizes)
        self.dataloaders["smac"] = AggregateCriticDatasetSMAC(
            folder_paths=smac_config["folder_paths"],
            map_types=smac_config["map_types"],
            device=self.device,
            batch_sizes=batch_sizes,
            general_config=smac_config["config"],
            max_block_size=max_block_size,
            max_action_size=max_action_size,
        )

    def init_pogema_dataset(
        self, dataset_config, batch_size, max_block_size: int, max_action_size: int
    ):
        if ("pogema" not in dataset_config) or (
            len(dataset_config["pogema"]["folder_paths"]) == 0
        ):
            print("Not enough information about pogema dataset")
            return

        from envs.pogema_env.dataset import AggregateDatasetPogema

        num_folders = len(dataset_config["pogema"]["folder_paths"])
        if num_folders == 2:
            batch_sizes = [int(batch_size // 10 * 9), int(batch_size // 10)]
        else:
            batch_sizes = [int(batch_size // num_folders) for _ in range(num_folders)]
        self.batch_sizes["pogema"] = sum(batch_sizes)
        self.dataloaders["pogema"] = AggregateDatasetPogema(
            folder_paths=dataset_config["pogema"]["folder_paths"],
            device=self.device,
            batch_sizes=batch_sizes,
            general_config=dataset_config["pogema"]["config"],
            max_block_size=max_block_size,
            max_action_size=max_action_size,
        )

    def init_grf_dataset(
        self, dataset_config, batch_size, max_block_size: int, max_action_size: int
    ):
        if ("grf" not in dataset_config) or (
            len(dataset_config["grf"]["folder_paths"]) == 0
        ):
            print("Not enough information about grf dataset")
            return
        num_folders = len(dataset_config["grf"]["folder_paths"])
        batch_sizes = [int(batch_size // num_folders) for _ in range(num_folders)]
        self.batch_sizes["grf"] = sum(batch_sizes)
        self.dataloaders["grf"] = AggregateCriticDatasetGRF(
            folder_paths=dataset_config["grf"]["folder_paths"],
            device=self.device,
            batch_sizes=batch_sizes,
            general_config=dataset_config["grf"]["config"],
            max_block_size=max_block_size,
            max_action_size=max_action_size,
        )

    def __iter__(self):
        """
        Creates an iterator that yields data batches from the aggregated datasets with specified batch sizes.
        """
        dataset_iters = [
            iter(dataloader) for key, dataloader in self.dataloaders.items()
        ]

        batch_sizes = [batch_size for batch_size in self.batch_sizes.values()]
        while True:
            (
                batch_inputs,
                batch_targets,
                batch_mask_targets,
                batch_inputs_next,
                batch_extra_info,
            ) = zip(*[next(dataset_iter) for dataset_iter in dataset_iters])
            batch_obs = dict()
            for key in batch_inputs[0].keys():
                obs_list = [batch_input[key] for batch_input in batch_inputs]
                if key == "masked_action":
                    batch_obs[key] = torch.cat(obs_list, dim=0)
                    continue
                max_block_size = (
                    self.max_action_size
                    if key == "action_mask"
                    else self.max_block_size
                )
                batch_obs[key] = concat_and_fill(
                    obs_list, 0, batch_sizes, max_block_size
                )

            batch_obs_next = dict()
            for key in batch_inputs_next[0].keys():
                obs_list = [batch_input[key] for batch_input in batch_inputs_next]
                if key == "masked_action":
                    batch_obs_next[key] = torch.cat(obs_list, dim=0)
                    continue
                max_block_size = (
                    self.max_action_size
                    if key == "action_mask"
                    else self.max_block_size
                )
                batch_obs_next[key] = concat_and_fill(
                    obs_list, 0, batch_sizes, max_block_size
                )

            batch_info = dict()
            for key in batch_extra_info[0].keys():
                info_list = [batch_input[key] for batch_input in batch_extra_info]
                batch_info[key] = torch.cat(info_list, dim=0)

            list_num_epoch = [
                dataset.num_epoch for key, dataset in self.dataloaders.items()
            ]
            list_num_epoch = torch.tensor(list_num_epoch)
            self.num_epoch = int(list_num_epoch.float().mean().item())
            true_target = torch.cat(batch_targets, dim=0)
            mask_target = torch.cat(batch_mask_targets, dim=0)
            yield batch_obs, true_target, mask_target, batch_obs_next, batch_info

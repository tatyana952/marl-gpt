from envs.grf_env.obs_tokenizer import grfTokenizer
from utils.aggregated_data_loader import AggregateDataset
from utils.env_observation import from_vector_to_tensor
from utils.dataset import (
    build_masked_history,
    ObservationProperty,
    HistoryDataLoader,
    CriticDataLoader,
)


class HistoryDataLoaderGRF(HistoryDataLoader):
    def __init__(
        self,
        config: ObservationProperty,
        batch_size,
        file_paths=None,
        device="cpu",
        folder_path=None,
    ):
        tokenizer = grfTokenizer(config.map_type)
        super().__init__(config, batch_size, file_paths, device, folder_path, tokenizer)

    def _build_with_history(self, indxs, target, dones):
        obs_dict = dict()
        obs = self.input_dict_tensor["obs"]
        obs_dict["obs"] = build_masked_history(obs, dones, indxs, self.history_len).to(
            self.device
        )
        return obs_dict

    def _build_action_mask(self, obs_res_dict, obs_dict):
        obs_res_dict["action_mask"] = from_vector_to_tensor(
            [1 for _ in range(20)], obs_res_dict["obs"].shape
        ).to(self.device)


class AggregateDatasetGRF(AggregateDataset):
    def __init__(
        self,
        folder_paths: list[str],
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
        self._init_datasets(folder_paths, batch_sizes, general_config)

    def _init_datasets(self, folder_paths, batch_sizes, general_config):
        for folder_path, batch_size in zip(folder_paths, batch_sizes):
            config = ObservationProperty(
                history_len=general_config["history_len"],
                history_step=general_config["history_step"],
            )
            dataset = HistoryDataLoaderGRF(
                config=config,
                batch_size=batch_size,
                folder_path=folder_path,
                device=self.device,
            )
            self.datasets.append(dataset)


class AggregateCriticDatasetGRF(AggregateDataset):
    def __init__(
        self,
        folder_paths: list[str],
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
        self._init_datasets(folder_paths, batch_sizes, general_config)
        if self.env_specific:
            self._env_indicator = 3

    def _init_datasets(self, folder_paths, batch_sizes, general_config):
        for folder_path, batch_size in zip(folder_paths, batch_sizes):
            config = ObservationProperty(
                history_len=general_config["history_len"],
                history_step=general_config["history_step"],
                num_agents=0,
                group_reward=False,
                q_n_step=general_config.get("q_n_step", 1),
                part_of_dataset=general_config.get("part_of_dataset", 1),
                load_part_file=bool(general_config.get("load_part_file", False)),
            )
            dataset = CriticDataLoader(
                config=config,
                batch_size=batch_size,
                folder_path=folder_path,
                device=self.device,
                dataloader_class=HistoryDataLoaderGRF,
            )
            self.datasets.append(dataset)


def main_v1():
    config = ObservationProperty(history_len=3)
    dl = HistoryDataLoaderGRF(config, 3, folder_path="dataset_1/clean_data/GRF")
    data = iter(dl)
    obs, target = next(data)
    print("----- Shapes -------")
    for key in obs.keys():
        if obs[key] is not None:
            print(f"{key}: {obs[key].shape}")


def main():
    folder_paths = [
        "dataset_1/clean_data/GRF",
        "dataset_1/clean_data/GRF",
    ]
    batch_sizes = [4, 4]
    config = {"history_len": 3, "history_step": 1}
    aggregated_dataset = AggregateDatasetGRF(
        folder_paths=folder_paths,
        device="cuda:0",
        batch_sizes=batch_sizes,
        general_config=config,
    )
    data = iter(aggregated_dataset)
    qx, qy = next(data)
    for key in qx.keys():
        print(f"{key}: {qx[key].shape}")

    for i in range(qx["obs"].shape[1]):
        r = list()
        for key in qx.keys():
            r.append(qx[key][0][i].item())
        print(f"{i}: {r}")


if __name__ == "__main__":
    main()

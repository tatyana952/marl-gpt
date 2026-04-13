import torch
from torch.utils.data import Dataset

from utils.dataset import CriticDataLoader


def concat_and_fill(
    tensor_list: list[torch.Tensor],
    value: float,
    batch_sizes: list,
    max_block_size: int = 0,
) -> torch.Tensor:
    # Find maximum column count across all tensors
    if max_block_size == 0:
        max_block_size = max(t.shape[-1] for t in tensor_list) if tensor_list else 0

    # Pad each tensor to (b_i, c) with value g
    padded_tensors = []
    for t, batch_size in zip(tensor_list, batch_sizes):
        if t.numel() == 0:  # Handle empty tensors
            continue
        tensor_shape = t.shape

        assert (
            max_block_size >= tensor_shape[-1]
        ), f"Small size of parameter max_block_size {max_block_size} < {tensor_shape[-1]}"

        padded = torch.full(
            (batch_size, max_block_size), value, dtype=t.dtype, device=t.device
        )
        padded[:, : tensor_shape[-1]] = t
        padded_tensors.append(padded)

    return torch.cat(padded_tensors, dim=0)


def sample_not_equal_batch(old_act, a_mask):
    device = a_mask.device
    b, n = a_mask.shape

    # Create indices [0, 1, ..., n-1] for each row
    indices = torch.arange(n, device=device).expand(b, n)

    # Use mask: a_mask_bool AND not equal to A[i] for each row
    a_mask_bool = a_mask.bool()
    old_act_expanded = old_act.unsqueeze(-1).expand(b, n)
    valid = a_mask_bool & (indices != old_act_expanded)

    # For each row, sample one from the valid indices
    num_valid = valid.sum(dim=1)
    probs = valid.float()
    probs[torch.isnan(probs)] = 1e-8
    probs[torch.isinf(probs)] = 10
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
    probs = probs.clamp(min=1e-8)

    # Sample one index per row
    try:
        new_acts = torch.where(
            num_valid > 0,
            torch.multinomial(probs, 1).squeeze(-1),
            torch.tensor(-1, device=device),
        )
    except:
        probs = probs * 0 + 1
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
        new_acts = torch.where(
            num_valid > 0,
            torch.multinomial(probs, 1).squeeze(-1),
            torch.tensor(-1, device=device),
        )
    return new_acts.to(old_act.dtype)


class AggregateDataset(Dataset):
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

        self.datasets = []
        self.batch_sizes = batch_sizes
        self.max_block_size = max_block_size
        self.max_action_size = max_action_size
        self.num_epoch = 0

        self.mask_ratio = general_config.get("mask_ratio", 0)
        self.rand_act = bool(general_config.get("rand_act", False))
        self.without_true_act = bool(general_config.get("without_true_act", False))
        self.with_Q = bool(general_config.get("with_Q", True))
        self.last_token = bool(general_config.get("last_token", True))
        self.env_specific = bool(general_config.get("env_specific", False))

        self.device = device

        self._env_indicator = 0
        self._last_tok_attr = 128

    def _init_datasets(self, folder_paths, batch_sizes, general_config):
        self.datasets = []  # TODO

    def merge_batchs(self, batch_inputs):
        batch_obs = dict()
        for key in batch_inputs[0].keys():
            obs_list = [batch_input[key] for batch_input in batch_inputs]
            max_block_size = (
                self.max_action_size if key == "action_mask" else self.max_block_size
            )
            batch_obs[key] = concat_and_fill(
                obs_list, 0, self.batch_sizes, max_block_size
            )
        return batch_obs

    def add_random_actions(self, batch_obs, targets, batch_obs_next, extra_info):
        indxs = batch_obs["masked_action"]
        action_mask = batch_obs["action_mask"][indxs]

        extra_info["rewards"][indxs] = -0.02
        extra_info["standart_rtg"][indxs] = 0

        targets[indxs] = sample_not_equal_batch(targets[indxs], action_mask)

        return batch_obs, targets, batch_obs_next, extra_info

    def add_last_token(self, dict_obs):
        dict_obs["attr_pos"][:, -1] = self._last_tok_attr
        dict_obs["obs"][:, -1] = self._env_indicator

    def __iter__(self):
        """
        Creates an iterator that yields data batches from the aggregated datasets with specified batch sizes.
        """
        dataset_iters = [iter(dataset) for dataset in self.datasets]

        while True:
            batch_inputs, batch_targets, batch_inputs_next, extra_info_dict = zip(
                *[next(dataset_iter) for dataset_iter in dataset_iters]
            )
            batch_obs = self.merge_batchs(batch_inputs)
            batch_obs_next = self.merge_batchs(batch_inputs_next)

            extra_info = dict()
            for key in extra_info_dict[0].keys():
                info_list = [input[key] for input in extra_info_dict]
                extra_info[key] = torch.cat(info_list, dim=0)

            list_num_epoch = [dataset.num_epoch for dataset in self.datasets]
            list_num_epoch = torch.tensor(list_num_epoch)
            self.num_epoch = int(list_num_epoch.float().mean().item())

            targets = torch.cat(batch_targets, dim=0)

            if self.mask_ratio > 0:
                batch_obs["masked_action"] = (
                    torch.rand(
                        batch_obs["obs"].shape[0],
                    )
                    > (1 - self.mask_ratio)
                ).to(batch_obs["obs"].device)

            if self.rand_act:
                batch_obs, mask_targets, batch_obs_next, extra_info = (
                    self.add_random_actions(
                        batch_obs, targets.clone(), batch_obs_next, extra_info
                    )
                )
            else:
                mask_targets = targets.clone()

            extra_info["Q_mask"] = torch.full(
                targets.shape,
                self.with_Q,
                dtype=torch.bool,
                device=batch_obs["obs"].device,
            )

            if self.last_token:
                self.add_last_token(batch_obs)
                self.add_last_token(batch_obs_next)

            if self.without_true_act:
                yield batch_obs, mask_targets, batch_obs_next, extra_info
            else:
                yield batch_obs, targets, mask_targets, batch_obs_next, extra_info


def main():
    folder_paths = [
        "dataset_1/clean_data/clean_data_p5_300k",
        "dataset_1/clean_data/zerg_5_vs_5",
    ]
    map_types = ["protoss", "zerg"]
    batch_sizes = [4, 4]
    config = {"history_len": 3, "history_step": 1, "mask_ratio": 0.5, "rand_act": True}
    aggregated_dataset = AggregateDataset(
        folder_paths=folder_paths,
        map_types=map_types,
        device="cuda:0",
        batch_sizes=batch_sizes,
        general_config=config,
        DataLoader=CriticDataLoader,
    )
    data = iter(aggregated_dataset)
    res = next(data)
    for key in res[0].keys():
        print(f"{key}: {res[0][key].shape}")


if __name__ == "__main__":
    main()

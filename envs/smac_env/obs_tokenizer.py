from __future__ import annotations
import torch
import json


def get_pos_matr(
    input_tensor: torch.Tensor, group_indx: int, device
) -> tuple[torch.Tensor]:
    shape = input_tensor.shape
    if len(shape) == 3:
        b, n, f = shape
    elif len(shape) == 2:
        b, f = shape
        n = 1

    group_pos = torch.full((n, f), group_indx).to(device)
    agent_pos = torch.arange(n).unsqueeze(1).repeat(1, f).to(device)

    return group_pos, agent_pos


def tensors_concatenate(
    input_tensor: torch.Tensor, keys: list[str], batch_size: int, device
):
    tensors_list = [input_tensor[key].reshape((batch_size, -1)) for key in keys]
    return torch.cat(tensors_list, dim=1).to(device)


def repeat_for_history(
    input_tensor: torch.Tensor, history_len: int, device, add_num: int = 0
):
    tensors_list = [input_tensor + i * add_num for i in range(history_len)]
    return torch.cat(tensors_list, dim=0).to(device)


def from_attr_dict_to_tensors(dict_attr, n_ally, n_enemy):
    res = dict()
    res["obs_own"] = torch.tensor(dict_attr["own"], dtype=torch.int)
    ally = [dict_attr["ally"] for _ in range(n_ally)]
    res["obs_ally"] = torch.tensor(ally, dtype=torch.int)
    enemy = [dict_attr["enemy"] for _ in range(n_enemy)]
    res["obs_enemy"] = torch.tensor(enemy, dtype=torch.int)
    return res


class smacTokenizer:
    def __init__(self, map_type, pos_dict=None):
        self.observation_keys = ["obs_own", "obs_ally", "obs_enemy"]

        if pos_dict is None:
            with open("envs/smac_env/attr_for_all_map.json", "r") as f:
                attr_cfg = json.load(f)
            self.dict_attr = attr_cfg[map_type]
        else:
            self.dict_attr = pos_dict

    def get_obs_pos_vectors(self, observation: dict[torch.Tensor]):
        device = observation["obs_own"].device
        batch_size = observation["obs_own"].shape[0]

        if len(observation["obs_own"].shape) == 3:
            history_len = observation["obs_own"].shape[1]
        else:
            history_len = 1

        group_pos = dict()
        agent_pos = dict()
        attr_pos = from_attr_dict_to_tensors(
            self.dict_attr,
            observation["obs_ally"].shape[-2],
            observation["obs_enemy"].shape[-2],
        )

        # b = batch_size * history_len
        # obs_own (b, f)
        new_shape = (-1,) + observation["obs_own"].shape[-1:]
        group_pos["obs_own"], agent_pos["obs_own"] = get_pos_matr(
            observation["obs_own"].reshape(new_shape), 0, device
        )
        # obs_enemy (b, n, f)
        new_shape = (-1,) + observation["obs_enemy"].shape[-2:]
        group_pos["obs_enemy"], agent_pos["obs_enemy"] = get_pos_matr(
            observation["obs_enemy"].reshape(new_shape), 1, device
        )
        # obs_ally (b, n, f)
        new_shape = (-1,) + observation["obs_ally"].shape[-2:]
        group_pos["obs_ally"], agent_pos["obs_ally"] = get_pos_matr(
            observation["obs_ally"].reshape(new_shape), 0, device
        )
        agent_pos["obs_ally"] += 1

        res_obs = tensors_concatenate(
            observation, self.observation_keys, batch_size * history_len, device
        )
        res_group_pos = tensors_concatenate(
            group_pos, self.observation_keys, 1, device
        ).squeeze()
        res_agent_pos = tensors_concatenate(
            agent_pos, self.observation_keys, 1, device
        ).squeeze()
        res_attr_pos = tensors_concatenate(
            attr_pos, self.observation_keys, 1, device
        ).squeeze()
        res_time_pos = torch.zeros(res_group_pos.shape, device=device, dtype=torch.int)

        res_obs = res_obs.reshape((batch_size, -1))
        res_group_pos = repeat_for_history(res_group_pos, history_len, device)
        res_agent_pos = repeat_for_history(res_agent_pos, history_len, device)
        res_attr_pos = repeat_for_history(res_attr_pos, history_len, device)
        res_time_pos = repeat_for_history(res_time_pos, history_len, device, 1)

        return res_obs, res_group_pos, res_agent_pos, res_time_pos, res_attr_pos


def main():
    # from smac_env.obs_tokenizer import *
    from utils.dataset import SimpleDataLoader

    filename = "/marl-gpt/dataset_1/trivial/data_protoss_5_vs_5_n0.json"
    datal = SimpleDataLoader(2, file_paths=[filename])
    data = iter(datal)
    res = next(data)

    tokenizer = smacTokenizer("protoss")
    rez_t = tokenizer.get_obs_pos_vectors(res[0])


if __name__ == "__main__":
    main()

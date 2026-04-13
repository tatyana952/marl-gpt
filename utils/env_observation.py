import torch


def from_vector_to_tensor(vector, obs_shape):
    tensor_pos = torch.tensor(vector, dtype=torch.int)
    tensor_pos = tensor_pos.unsqueeze(0).repeat(obs_shape[0], 1)
    return tensor_pos


def stack_obs(observations, device) -> dict:
    """from list of dicts to dict of lists"""
    keys = observations[0].keys()
    obs_dict = {key: list() for key in keys}
    for agent_obs in observations:
        for key in keys:
            obs_dict[key].append(agent_obs[key])

    obs_dict = {key: torch.tensor(elem).to(device) for key, elem in obs_dict.items()}
    return obs_dict


def stack_tensor_history(history, history_length, history_step, device):
    # tensor (b, ...) -> (b, h, ...)
    shape = history[-1].shape

    # Pad with zeros if history is shorter than h
    if len(history) < history_length:
        new_history = list()
        pad_size = history_length - len(history)
        new_history = [torch.zeros(shape, device=device) for _ in range(pad_size)]
        new_history.extend(history)
    else:
        new_history = history
    output_tensor = torch.stack(new_history, dim=1)  # (b, h, m, n)

    # the first obs in history = the last obs
    output_tensor = torch.flip(output_tensor, dims=[1])

    if history_step > 1:
        indx_with_h_step = torch.arange(0, history_length, history_step)
        output_tensor = output_tensor[:, indx_with_h_step]
    return output_tensor


def stack_dict_tensor(history, history_length, history_step, device):
    keys = history[-1].keys()
    obs_dict = {key: list() for key in keys}
    for key in keys:
        if key == "action_mask":
            obs_dict[key] = history[-1][key]
            continue
        history_list = [obs[key] for obs in history]
        obs_dict[key] = stack_tensor_history(
            history_list, history_length, history_step, device
        )
    return obs_dict


def reshape_tensor_to_max_size(
    tensor, batch_size: int, max_block_size: int = 0, value: int = 0
):
    tensor_shape = tensor.shape
    if max_block_size == 0:
        max_block_size = tensor.shape[-1]

    assert (
        max_block_size >= tensor_shape[-1]
    ), f"Small size of parameter max_block_size {max_block_size} < {tensor_shape[-1]}"

    padded = torch.full(
        (batch_size, max_block_size), value, dtype=tensor.dtype, device=tensor.device
    )
    padded[:, : tensor_shape[-1]] = tensor
    return padded


def reshape_obs(obs_dict, max_block_size: int = 0, max_action_size: int = 0):
    batch_obs = dict()
    batch_size = obs_dict["obs"].shape[0]
    for key in obs_dict.keys():
        max_size = max_action_size if key == "action_mask" else max_block_size
        batch_obs[key] = reshape_tensor_to_max_size(
            obs_dict[key], batch_size, max_size, 0
        )
    return batch_obs

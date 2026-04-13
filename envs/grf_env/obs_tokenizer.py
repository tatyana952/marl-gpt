from __future__ import annotations
import torch


def repeat_for_history(
    input_tensor: torch.Tensor, history_len: int, device, add_num: int = 0
):
    tensors_list = [input_tensor + i * add_num for i in range(history_len)]
    return torch.cat(tensors_list, dim=0).to(device)


class grfTokenizer:
    def __init__(self, map_type):
        self.observation_keys = ["obs"]

        self.group_pos_template, self.agent_pos_template, self.attr_pos_template = (
            self._build_static_templates()
        )

    @staticmethod
    def _build_static_templates() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_dim, n_players = 115, 11

        group_pos = torch.zeros(obs_dim, dtype=torch.int)
        agent_pos = torch.zeros(obs_dim, dtype=torch.int)
        attr_pos = torch.zeros(obs_dim, dtype=torch.int)

        idx = 0
        # left  team  (group = 0, agents 1..11)
        for p in range(n_players):
            # x
            group_pos[idx] = 0
            agent_pos[idx] = p + 1
            attr_pos[idx] = 0  # x
            idx += 1
            # y
            group_pos[idx] = 0
            agent_pos[idx] = p + 1
            attr_pos[idx] = 1  # y
            idx += 1

        # left team: directions (dx,dy)
        for p in range(n_players):
            # dx
            group_pos[idx] = 0
            agent_pos[idx] = p + 1
            attr_pos[idx] = 2  # dx
            idx += 1
            # dy
            group_pos[idx] = 0
            agent_pos[idx] = p + 1
            attr_pos[idx] = 3  # dy
            idx += 1

        # right team: coordinates
        for p in range(n_players):
            group_pos[idx] = 1
            agent_pos[idx] = p + 1
            attr_pos[idx] = 0  # x
            idx += 1

            group_pos[idx] = 1
            agent_pos[idx] = p + 1
            attr_pos[idx] = 1  # y
            idx += 1

        # right team: directions
        for p in range(n_players):
            group_pos[idx] = 1
            agent_pos[idx] = p + 1
            attr_pos[idx] = 2  # dx
            idx += 1

            group_pos[idx] = 1
            agent_pos[idx] = p + 1
            attr_pos[idx] = 3  # dy
            idx += 1

        # ball position (x,y,z)
        for coord in range(3):
            group_pos[idx] = 0
            agent_pos[idx] = 0
            attr_pos[idx] = 4 + coord  # 4,5,6
            idx += 1

        # ball direction (dx,dy,dz)
        for coord in range(3):
            group_pos[idx] = 0
            agent_pos[idx] = 0
            attr_pos[idx] = 7 + coord  # 7,8,9
            idx += 1

        # ball-ownership one-hot (3 dims)
        for coord in range(3):
            group_pos[idx] = 0
            agent_pos[idx] = 0
            attr_pos[idx] = 10 + coord
            idx += 1

        # active-player one-hot (11 dims)
        for p in range(n_players):
            group_pos[idx] = 0
            agent_pos[idx] = p + 1
            attr_pos[idx] = 13
            idx += 1

        # game-mode one-hot (7 dims)
        for mode in range(7):
            group_pos[idx] = 0
            agent_pos[idx] = 0
            attr_pos[idx] = 14 + mode
            idx += 1

        return group_pos, agent_pos, attr_pos

    def get_obs_pos_vectors(self, observation: dict[torch.Tensor]):
        device = observation["obs"].device
        batch_size = observation["obs"].shape[0]

        if len(observation["obs"].shape) == 3:
            history_len = observation["obs"].shape[1]
        else:
            history_len = 1

        if observation["obs"].ndim == 3:  # (B,H,115)
            history_len = observation["obs"].shape[1]
            res_obs = observation["obs"].reshape(batch_size, -1).to(device)
        else:  # (B,115)
            history_len = 1
            res_obs = observation["obs"].to(device)

        # pos templates to device
        res_group_pos = self.group_pos_template.to(device)
        res_agent_pos = self.agent_pos_template.to(device)
        res_attr_pos = self.attr_pos_template.to(device)
        res_time_pos = torch.zeros_like(res_group_pos, device=device, dtype=torch.int)

        res_group_pos = repeat_for_history(res_group_pos, history_len, device)
        res_agent_pos = repeat_for_history(res_agent_pos, history_len, device)
        res_attr_pos = repeat_for_history(res_attr_pos, history_len, device)
        res_time_pos = repeat_for_history(res_time_pos, history_len, device, add_num=1)

        return res_obs, res_group_pos, res_agent_pos, res_time_pos, res_attr_pos

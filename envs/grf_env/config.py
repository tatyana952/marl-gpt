from typing import Literal, Optional
from pydantic import BaseModel


class GRFInferenceConfig(BaseModel):
    device: str = "cuda"
    history_len: int = 4
    history_step: int = 1
    pos_dict: Optional[dict] = None
    map_name: str = "academy_pass_and_shoot_with_keeper"
    num_agents: int = 10
    max_episode_steps: int = 256
    obs_per_agent: bool = False


class GRFConfig(BaseModel):
    env_name: str = "11_vs_11_stochastic"
    stacked: bool = False
    representation: Literal[
        "pixels", "pixels_gray", "extracted", "simple115", "simple115v2"
    ] = "simple115v2"
    rewards: str = "scoring"
    write_goal_dumps: bool = False
    write_full_episode_dumps: bool = False
    render: bool = False
    write_video: bool = False
    dump_frequency: int = 1
    # logdir: str = ''
    # extra_players: Optional[List[str]] = None
    number_of_left_players_agent_controls: int = 1
    # number_of_right_players_agent_controls: int = 0
    # channel_dimensions: Tuple[int, int] = (96, 72)
    # other_config_options: Dict[str, Any] = {}

    @classmethod
    def get_scenario_config(cls, name: str) -> "GRFConfig":
        """Factory method for common GRF scenarios"""
        configs = {
            # Standard competitive scenarios
            "11vs11a11": cls(
                env_name="11_vs_11_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=11,
            ),
            "11vs11a2": cls(
                env_name="11_vs_11_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=2,
            ),
            "11vs11a6": cls(
                env_name="11_vs_11_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=6,
            ),
            # Easy competitive scenarios
            "11vs11a11e": cls(
                env_name="11_vs_11_easy_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=11,
            ),
            "11vs11a2e": cls(
                env_name="11_vs_11_easy_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=2,
            ),
            "11vs11a6e": cls(
                env_name="11_vs_11_easy_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=6,
            ),
            # Hard competitive scenarios
            "11vs11a11h": cls(
                env_name="11_vs_11_hard_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=11,
            ),
            "11vs11a2h": cls(
                env_name="11_vs_11_hard_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=2,
            ),
            "11vs11a6h": cls(
                env_name="11_vs_11_hard_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=6,
            ),
            # Randomised competitive scenarios
            "11vs11a11r": cls(
                env_name="11_vs_11_random_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=11,
            ),
            "11vs11a2r": cls(
                env_name="11_vs_11_random_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=2,
            ),
            # Checkpoint scenarios
            "11vs11a11h_checkpoint": cls(
                env_name="11_vs_11_hard_stochastic",
                representation="simple115v2",
                rewards="scoring,checkpoints",
                number_of_left_players_agent_controls=11,
            ),
            "11vs11a11_checkpoint": cls(
                env_name="11_vs_11_stochastic",
                representation="simple115v2",
                rewards="scoring,checkpoints",
                number_of_left_players_agent_controls=11,
            ),
            "11vs11a11e_checkpoint": cls(
                env_name="11_vs_11_easy_stochastic",
                representation="simple115v2",
                rewards="scoring,checkpoints",
                number_of_left_players_agent_controls=11,
            ),
            # Hardest enemy bot
            "11vs11a11max": cls(
                env_name="11_vs_11_max",
                representation="simple115v2",
                number_of_left_players_agent_controls=11,
            ),
            # Academy scenarios
            "academy_empty_goal_close": cls(
                env_name="academy_empty_goal_close",
                representation="simple115v2",
                number_of_left_players_agent_controls=1,
            ),
            "academy_empty_goal": cls(
                env_name="academy_empty_goal", representation="simple115v2"
            ),
            "academy_run_to_score_with_keeper": cls(
                env_name="academy_run_to_score_with_keeper",
                representation="simple115v2",
            ),
            "academy_pass_and_shoot_with_keeper": cls(
                env_name="academy_pass_and_shoot_with_keeper",
                representation="simple115v2",
                number_of_left_players_agent_controls=2,
            ),
            "academy_corner": cls(
                env_name="academy_corner", representation="simple115v2"
            ),
            "academy_counterattack_easy": cls(
                env_name="academy_counterattack_easy", representation="simple115v2"
            ),
            "academy_counterattack_hard": cls(
                env_name="academy_counterattack_hard", representation="simple115v2"
            ),
            "academy_run_to_score": cls(
                env_name="academy_run_to_score", representation="simple115v2"
            ),
            "academy_run_pass_and_shoot_with_keeper": cls(
                env_name="academy_run_pass_and_shoot_with_keeper",
                representation="simple115v2",
            ),
            "academy_3_vs_1_with_keeper": cls(
                env_name="academy_3_vs_1_with_keeper", representation="simple115v2"
            ),
            "academy_single_goal_versus_lazy": cls(
                env_name="academy_single_goal_versus_lazy", representation="simple115v2"
            ),
        }
        cfg = configs.get(name)
        if cfg is None:
            available_configs = ", ".join(configs.keys())
            raise KeyError(
                f"There is no config with name '{name}'. Available configs are: {available_configs}"
            )

        return cfg


def main():
    import yaml

    print(yaml.dump(GRFConfig.get_scenario_config("11v11_stochastic").model_dump()))


if __name__ == "__main__":
    main()

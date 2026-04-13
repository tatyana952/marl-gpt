from typing import List, Optional, Literal
from pydantic import BaseModel, conlist, confloat


class SMACGPTInferenceConfig(BaseModel):
    device: str = "cuda"
    history_len: int = 4
    history_step: int = 1
    map_type: str = "protoss"
    pos_dict: Optional[dict] = None
    smac_vers: Literal[1, 2] = 2
    obs_with_race: bool = True
    map_name: str = "protoss_5_vs_5"
    num_agents: int = 5
    max_episode_steps: int = 256
    obs_own_pos: bool = False
    obs_per_agent: bool = False


class TeamGenerationConfig(BaseModel):
    dist_type: str
    unit_types: conlist(str, min_length=1)
    weights: conlist(confloat(ge=0, le=1), min_length=1)
    exception_unit_types: Optional[List[str]] = []
    observe: bool


class StartPositionsConfig(BaseModel):
    dist_type: str
    p: confloat(ge=0, le=1)
    map_x: int
    map_y: int


class CapabilityConfig(BaseModel):
    n_units: int
    n_enemies: int
    team_gen: TeamGenerationConfig
    start_positions: StartPositionsConfig


class SMAC2ArgsConfig(BaseModel):
    continuing_episode: bool = False
    difficulty: str = "7"
    game_version: Optional[str] = None
    map_name: str
    move_amount: int = 2
    obs_all_health: bool = True
    obs_instead_of_state: bool = False
    obs_last_action: bool = False
    obs_own_health: bool = True
    obs_pathing_grid: bool = False
    obs_terrain_height: bool = False
    obs_timestep_number: bool = False
    reward_death_value: float = 10
    reward_defeat: float = 0
    reward_negative_scale: float = 0.5
    reward_only_positive: bool = True
    reward_scale: bool = True
    reward_scale_rate: float = 20
    reward_sparse: bool = False
    reward_win: float = 200
    replay_dir: str = ""
    replay_prefix: str = ""
    conic_fov: bool = False
    use_unit_ranges: bool = True
    min_attack_range: int = 2
    obs_own_pos: bool = True
    num_fov_actions: int = 12
    capability_config: CapabilityConfig
    state_last_action: bool = True
    state_timestep_number: bool = False
    step_mul: int = 8
    heuristic_ai: bool = False
    debug: bool = False
    prob_obs_enemy: float = 1.0
    action_mask: bool = True

    @staticmethod
    def base_config(faction: str, n_units: int, n_enemies: int) -> "SMAC2ArgsConfig":
        unit_types = {
            "protoss": ["stalker", "zealot", "colossus"],
            "zerg": ["zergling", "hydralisk", "baneling"],
            "terran": ["marine", "marauder", "medivac"],
        }

        # exception_unit_types field specifies unit types that should be excluded from being the sole composition of a team during scenario generation.
        # This is to prevent scenarios where a team consists entirely of units that might not function effectively on their own.
        exception_unit_types = {
            "protoss": [],
            "zerg": ["baneling"],
            "terran": ["medivac"],
        }

        return SMAC2ArgsConfig(
            map_name=f"10gen_{faction}",
            capability_config=CapabilityConfig(
                n_units=n_units,
                n_enemies=n_enemies,
                team_gen=TeamGenerationConfig(
                    dist_type="weighted_teams",
                    unit_types=unit_types[faction],
                    weights=[0.45, 0.45, 0.1],
                    exception_unit_types=exception_unit_types[faction],
                    observe=True,
                ),
                start_positions=StartPositionsConfig(
                    dist_type="surrounded_and_reflect", p=0.5, map_x=32, map_y=32
                ),
            ),
        )

    @classmethod
    def get_scenario_config(cls, name):
        configs = {
            "protoss_5_vs_5": cls.base_config("protoss", 5, 5),
            "zerg_5_vs_5": cls.base_config("zerg", 5, 5),
            "terran_5_vs_5": cls.base_config("terran", 5, 5),
            "protoss_5_vs_6": cls.base_config("protoss", 5, 6),
            "zerg_5_vs_6": cls.base_config("zerg", 5, 6),
            "terran_5_vs_6": cls.base_config("terran", 5, 6),
            "protoss_4_vs_5": cls.base_config("protoss", 4, 5),
            "zerg_4_vs_5": cls.base_config("zerg", 4, 5),
            "terran_4_vs_5": cls.base_config("terran", 4, 5),
            "protoss_6_vs_6": cls.base_config("protoss", 6, 6),
            "zerg_6_vs_6": cls.base_config("zerg", 6, 6),
            "terran_6_vs_6": cls.base_config("terran", 6, 6),
            "protoss_3_vs_3": cls.base_config("protoss", 3, 3),
            "zerg_3_vs_3": cls.base_config("zerg", 3, 3),
            "terran_3_vs_3": cls.base_config("terran", 3, 3),
            "protoss_10_vs_10": cls.base_config("protoss", 10, 10),
            "zerg_10_vs_10": cls.base_config("zerg", 10, 10),
            "terran_10_vs_10": cls.base_config("terran", 10, 10),
            "protoss_20_vs_20": cls.base_config("protoss", 20, 20),
            "zerg_20_vs_20": cls.base_config("zerg", 20, 20),
            "terran_20_vs_20": cls.base_config("terran", 20, 20),
            "protoss_10_vs_11": cls.base_config("protoss", 10, 11),
            "zerg_10_vs_11": cls.base_config("zerg", 10, 11),
            "terran_10_vs_11": cls.base_config("terran", 10, 11),
            "protoss_20_vs_23": cls.base_config("protoss", 20, 23),
            "zerg_20_vs_23": cls.base_config("zerg", 20, 23),
            "terran_20_vs_23": cls.base_config("terran", 20, 23),
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

    print(yaml.dump(SMAC2ArgsConfig.get_scenario_config("terran_5_vs_5").model_dump()))


if __name__ == "__main__":
    main()

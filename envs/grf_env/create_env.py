from envs.grf_env.config import GRFInferenceConfig, GRFConfig
from envs.grf_env.wrapper import InferenceGrfWrapper, LogWrapper
from envs.grf_env.grf import make_grf_v0


def make_grf_marl_gpt(cfg: GRFInferenceConfig, metrics_log_name: str = "metrics"):
    env_cfg = GRFConfig.get_scenario_config(cfg.map_name)
    env = make_grf_v0(env_cfg)
    env = InferenceGrfWrapper(env, cfg)
    env = LogWrapper(env, metrics_log_name)
    return env


def sf_make_gpt_grf(full_env_name, cfg=None, _env_config=None, render_mode=None):
    config_env = GRFInferenceConfig(
        history_len=4,  # TODO FINE-TUNE: change parametr for model
        pos_dict=None,
        map_name=cfg.experiment,
        obs_per_agent=True,
    )
    return make_grf_marl_gpt(config_env, "episode_extra_stats")

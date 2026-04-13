from envs.smac_env.config import SMAC2ArgsConfig, SMACGPTInferenceConfig
from envs.smac_env.smac_v2 import SMACv2, LogSMACWrapper, AutoResetWrapper
from envs.smac_env.wrapper import InferenceSmacWrapper, SeparatedObs


def make_smac_marl_gpt(map_name: str, env_config: dict, device: str = "cpu"):
    smac_v = int(env_config.get("smac_v", 2))
    config_inference = SMACGPTInferenceConfig(
        history_len=env_config.get("history_len", 6),
        history_step=env_config.get("history_step", 1),
        map_type=env_config.get("ally_type", "protoss"),
        device=device,
    )

    # Initialize the environment of the desired version
    if smac_v == 2:
        cfg = SMAC2ArgsConfig.get_scenario_config(map_name)
        env = SMACv2(cfg.model_dump(), smac_v)
    elif smac_v == 1:
        cfg = {"map_name": map_name}
        env = SMACv2(cfg, smac_v)
    else:
        raise KeyError(
            f"Wrong version of smac. Given parametr smac_v = '{smac_v}'. Available configs are: 1, 2"
        )
    env = LogSMACWrapper(env)

    # Observation for MARL-GPT model
    env = SeparatedObs(env, map_name, np_obs_type=False, smac_v=smac_v)
    env = InferenceSmacWrapper(env, config_inference)

    return env


def make_smac_marl_gpt_v2(
    config_inference: SMACGPTInferenceConfig,
    auto_reset: bool = False,
    metrics_log_name: str = "metrics",
):
    smac_v = config_inference.smac_vers
    map_name = config_inference.map_name

    # Initialize the environment of the desired version
    if smac_v == 2:
        cfg = SMAC2ArgsConfig.get_scenario_config(map_name)
        env = SMACv2(cfg.model_dump(), smac_v)
    elif smac_v == 1:
        cfg = {"map_name": map_name}
        env = SMACv2(cfg, smac_v)
    else:
        raise KeyError(
            f"Wrong version of smac. Given parametr smac_v = '{smac_v}'. Available configs are: 1, 2"
        )
    env = LogSMACWrapper(env, metrics_log_name)
    if auto_reset:
        env = AutoResetWrapper(env)

    # Observation for MARL-GPT model
    env = SeparatedObs(env, map_name, np_obs_type=False, smac_v=smac_v)
    pos_dict = env.decomposer.get_obs_attribute_as_dict()
    env = InferenceSmacWrapper(env, config_inference, pos_dict)

    return env


def make_adapted_env_smac(config_inference: SMACGPTInferenceConfig):
    smac_v = config_inference.smac_vers
    map_name = config_inference.map_name

    # Initialize the environment of the desired version
    if smac_v == 2:
        cfg = SMAC2ArgsConfig.get_scenario_config(map_name)
        env = SMACv2(cfg.model_dump(), smac_v)
    elif smac_v == 1:
        cfg = {"map_name": map_name, "obs_own_pos": config_inference.obs_own_pos}
        env = SMACv2(cfg, smac_v)
    else:
        raise KeyError(
            f"Wrong version of smac. Given parametr smac_v = '{smac_v}'. Available configs are: 1, 2"
        )
    env = LogSMACWrapper(env)

    # Observation for MARL-GPT model
    env = SeparatedObs(
        env,
        map_name,
        np_obs_type=False,
        smac_v=smac_v,
        zero_padding=True,
        with_own_coord=config_inference.obs_own_pos,
    )
    pos_dict = env.decomposer.get_obs_attribute_as_dict()
    env = InferenceSmacWrapper(env, config_inference, pos_dict)

    return env


def sf_make_gpt_smac_v2(full_env_name, cfg=None, _env_config=None, render_mode=None):
    config_env = SMACGPTInferenceConfig(
        history_len=4,  # TODO FINE-TUNE: change parametr for model
        map_type="terran",
        pos_dict=None,
        smac_vers=2,
        map_name=cfg.experiment,
        obs_per_agent=True,
    )
    return make_smac_marl_gpt_v2(
        config_env, auto_reset=True, metrics_log_name="episode_extra_stats"
    )

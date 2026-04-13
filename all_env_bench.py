from pathlib import Path
import argparse
import yaml

from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry


from gpt.inference import MARLGPTInference, InferenceConfig

PROJECT_NAME = "Benchmark"
BASE_PATH = Path("eval_configs")


def ensure_weights(eval_config):
    for algo_name, algo_cfg in eval_config["algorithms"].items():
        ToolboxRegistry.create_algorithm(algo_cfg["name"], **algo_cfg)


def registry_all_envs(list_of_envs_names):
    env_name = "pogema"
    if env_name in list_of_envs_names:
        from pogema_toolbox.create_env import Environment
        from envs.pogema_env.create_env import create_eval_env

        ToolboxRegistry.register_env(env_name, create_eval_env, Environment)

    env_name = "smacv2"
    if env_name in list_of_envs_names:
        from envs.smac_env.config import SMACGPTInferenceConfig
        from envs.smac_env.create_env import make_smac_marl_gpt_v2

        ToolboxRegistry.register_env(
            env_name, make_smac_marl_gpt_v2, SMACGPTInferenceConfig
        )

    env_name = "smacv1"
    if env_name in list_of_envs_names:
        from envs.smac_env.config import SMACGPTInferenceConfig
        from envs.smac_env.create_env import make_adapted_env_smac

        ToolboxRegistry.register_env(
            env_name, make_adapted_env_smac, SMACGPTInferenceConfig
        )

    env_name = "grf"
    if env_name in list_of_envs_names:
        from envs.grf_env.create_env import make_grf_marl_gpt
        from envs.grf_env.config import GRFInferenceConfig

        ToolboxRegistry.register_env(env_name, make_grf_marl_gpt, GRFInferenceConfig)


def main():
    list_of_envs_names = ["smacv2", "smacv1", "pogema", "grf"]

    parser = argparse.ArgumentParser(description="MARL-GPT Inference Script")
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable wandb logger (default: %(default)s)",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="smacv2",
        help="Env name: smacv2, pogema, grf (default: %(default)s)",
    )
    args = parser.parse_args()

    list_of_envs_names = [args.env_name]
    registry_all_envs(list_of_envs_names)
    ToolboxRegistry.register_algorithm("MARL-GPT", MARLGPTInference, InferenceConfig)

    folder_names = {
        "smacv2": {"smac": ["zerg_v2", "terran_v2", "protoss_v2"]},
        "pogema": {
            "01-random": ["01-random"],
            "02-mazes": ["02-mazes"],
            "03-warehouse": ["03-warehouse"],
            "04-movingai": ["04-movingai"],
        },
        "grf": {"grf": ["grf"]},
    }

    for name_env in list_of_envs_names:
        env_folder_names: dict[str, list[str]] = folder_names.get(name_env, dict())
        if len(env_folder_names) == 0:
            print(f"Unknown configuration files for environment '{name_env}'.")
            continue
        for folder, files in env_folder_names.items():
            for file_name in files:
                config_path = BASE_PATH / folder / f"{file_name}.yaml"
                with open(config_path) as f:
                    evaluation_config = yaml.safe_load(f)

                if evaluation_config["environment"]["name"] == "pogema":
                    maps_path = BASE_PATH / folder / "maps.yaml"
                    with open(maps_path, "r") as f:
                        maps = yaml.safe_load(f)
                    ToolboxRegistry.register_maps(maps)

                # ensuring model weights are downloaded
                ensure_weights(evaluation_config)

                eval_dir = BASE_PATH / folder
                initialize_wandb(
                    evaluation_config, eval_dir, args.disable_wandb, PROJECT_NAME
                )
                evaluation(evaluation_config, eval_dir=eval_dir)
                save_evaluation_results(eval_dir)


if __name__ == "__main__":
    main()

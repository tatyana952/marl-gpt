import shutil

import numpy as np
import sys

from tensorboardX import SummaryWriter
from gpt.finetune_vers.custom_learner import registry_changed_learner

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.envs.env_utils import register_env
from sample_factory.utils.utils import log, experiment_dir
from sample_factory.train import make_runner

SMACv2_ENV_NAME = "SMACv2"
GRF_ENV_NAME = "GRF"


def smac_extra_summaries(
    runner: Runner, policy_id: PolicyID, summary_writer: SummaryWriter, env_steps: int
):
    cfg = runner.cfg
    policy_avg_stats = runner.policy_avg_stats
    for key in policy_avg_stats:
        if key in ["reward", "len", "true_reward", "Done"]:
            continue

        avg = np.mean(np.array(policy_avg_stats[key][policy_id]))
        summary_writer.add_scalar(key, avg, env_steps)
        log.debug(f"{policy_id}-{key}: {round(float(avg), 3)}")


class SmacExtraSummariesObserver(AlgoObserver):
    def extra_summaries(
        self, runner: Runner, policy_id: PolicyID, writer: SummaryWriter, env_steps: int
    ) -> None:
        smac_extra_summaries(runner, policy_id, writer, env_steps)


def register_msg_handlers(cfg: Config, runner: Runner):
    runner.register_observer(SmacExtraSummariesObserver())


def override_default_params(parser):
    # Training, optimization, environment, and parallelization parameters
    # parser.set_defaults(
    #     gamma=0.99,
    #     learning_rate=5e-7,
    #     recurrence=32,
    #     rollout=32,
    #     train_for_env_steps=1000_000_000,
    #     ppo_epochs=5,
    #     exploration_loss_coeff=0.001,
    #     batch_size=512,
    #     max_grad_norm=0.5,
    #     num_envs_per_worker=2,
    #     num_workers=8,
    #     rnn_size=128,
    #     nonlinearity='relu',
    #     actor_critic_share_weights=True,
    #     normalize_input=False,
    #     encoder_mlp_layers=[512, 512, 512],
    #     decoder_mlp_layers=[256, ],
    #     value_bootstrap=False,
    #     reward_scale=0.1,
    #     kl_loss_coeff=0.3,
    #     ppo_clip_value=0.5,
    #     num_epochs=5,
    #     num_batches_per_epoch=1,
    #     value_loss_coeff=1,
    #     ppo_clip_ratio=0.1,
    # )

    parser.set_defaults(
        gamma=0.99,
        learning_rate=5e-6,
        recurrence=32,
        rollout=32,
        train_for_env_steps=1000_000_000,
        ppo_epochs=4,
        exploration_loss_coeff=0.01,
        batch_size=512,
        max_grad_norm=5.0,
        num_envs_per_worker=2,
        num_workers=12,
        rnn_size=128,
        nonlinearity="relu",
        actor_critic_share_weights=True,
        normalize_input=False,
        encoder_mlp_layers=[512, 512, 512],
        decoder_mlp_layers=[
            256,
        ],
        value_bootstrap=False,
    )

    # Experiment control, reporting, and statistics
    parser.set_defaults(
        env=SMACv2_ENV_NAME,
        experiment="terran_5_vs_5",
        restart_behavior="overwrite",
        with_wandb=False,
        wandb_project="marl-gpt-sf",
        heartbeat_reporting_interval=100,
        experiment_summaries_interval=100,
        save_best_metric="battle_won",
        keep_checkpoints=1,
        stats_avg=100,
    )


def add_extra_params(parser):
    p = parser
    # p.add_argument("--scenario", default='terran_5_vs_5', type=str, help="Default scenario")
    p.add_argument(
        "--wandb_save_weights", default=True, type=bool, help="Default scenario"
    )
    p.add_argument(
        "--gpt_model_path",
        default="out/ckpt.pt",
        type=str,
        help="Path to a pre-trained MARL-GPT checkpoint. Pass 'none' to train from scratch.",
    )


def parse_custom_args(argv=None, exp_cfg: dict = None, evaluation=False):
    parser, cfg = parse_sf_args(argv, evaluation=evaluation)
    add_extra_params(parser)
    override_default_params(parser)
    if exp_cfg is not None:
        parser.set_defaults(**exp_cfg)

    final_cfg = parse_full_cfg(parser)
    return final_cfg


def main():
    from gpt.finetune_vers.gpt_actor_critic import smac_make_actor_critic_func

    cfg = parse_custom_args()
    if cfg.env == SMACv2_ENV_NAME:
        from envs.smac_env.create_env import sf_make_gpt_smac_v2

        register_env(SMACv2_ENV_NAME, sf_make_gpt_smac_v2)
    elif cfg.env == GRF_ENV_NAME:
        from envs.grf_env.create_env import sf_make_gpt_grf

        register_env(GRF_ENV_NAME, sf_make_gpt_grf)
    global_model_factory().register_actor_critic_factory(smac_make_actor_critic_func)

    cfg, runner = make_runner(cfg)
    register_msg_handlers(cfg, runner)
    status = runner.init()

    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    if cfg.with_wandb and cfg.wandb_save_weights:
        path = experiment_dir(cfg=cfg)
        zip_name = str(path)
        shutil.make_archive(zip_name, "zip", path)
        import wandb

        wandb.save(zip_name + ".zip")
        wandb.finish()

    return status


if __name__ == "__main__":
    sys.exit(main())

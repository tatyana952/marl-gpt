# MARL-GPT: Foundation Model for Multi-Agent Reinforcement Learning

<div align="center" dir="auto">

[![arXiv](https://img.shields.io/badge/arXiv-2604.05943-b31b1b.svg)](https://arxiv.org/abs/2604.05943)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Hugging Face](https://img.shields.io/badge/Weights-MARL--GPT-blue?logo=huggingface)](https://huggingface.co/nortem/marl-gpt-model)
[![Hugging Face](https://img.shields.io/badge/Dataset-MARL--GPT-blue?logo=huggingface)](https://huggingface.co/datasets/nortem/marl-gpt-datasets)

</div>

Official implementation of **MARL-GPT**, a transformer-based foundation model for multi-agent reinforcement learning. This repository provides code for offline training on expert trajectories, model evaluation, and online fine-tuning with PPO. MARL-GPT demonstrates that a single transformer-based model can achieve competitive performance across diverse MARL environments (including StarCraft Multi-Agent Challenge (SMACv2), Google Research Football (GRF), and POGEMA) without task-specific architectural modifications.

## Additional information

1. Detailed configuration instructions for adding new environments, customizing training hyperparameters, and defining positional encodings are provided in **[SETTINGS.md](./suplementary/SETTINGS.md)**.
2. Pre-collected expert trajectories for offline training are available on Hugging Face: **[MARL-GPT Datasets](https://huggingface.co/datasets/nortem/marl-gpt-datasets)**
3. The full text of the paper, including the Appendix, is available at the following link: **[arXiv](https://arxiv.org/abs/2604.05943)**.
4. Video for the robotics experiment can be found at this link: **[video](./suplementary/marl-gpt-robotic.mp4)**.


## Installation

### Local Installation

It's recommended to use `uv` to install dependencies, but `pip` or `conda` should work too:

```bash
uv pip install -r docker/requirements.txt
```

For environments requiring StarCraft II (e.g., SMACv2), install the game binaries. On Ubuntu:

```bash
bash docker/install_sc2.sh
```

Set the `SC2PATH` environment variable to the installation directory:

```bash
export SC2PATH=/path/to/StarCraftII
```

> **Important:** `SC2PATH` must be set before running any SMACv2 experiment. Without it, the environment will fail with a cryptic import or connection error. To make it permanent, add the line to your `~/.bashrc` or `~/.zshrc`.

### Docker (Optional)

For the server-side experiments, it's recommended to use Docker images. Navigate to the `docker` folder and run:

```bash
sh build.sh
```

This will set up Docker and tag it with the appropriate name. The next steps are straightforward: create a Docker
container and include the source code files.

## Quickstart

### Offline Training

Train the MARL-GPT model on pre-collected expert trajectories:

```bash
python train_actor_critic.py config/config-train-critic.py
```

For multi-GPU training, use `torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 train_actor_critic.py config/config-train-critic.py
```

Trained checkpoints are saved to the `out/` directory.


### Online Fine-Tuning

Fine-tune a pre-trained model using online rollouts:

1. Set `history_len` in the `create_env.py` file for your target environment (look for `TODO FINE-TUNE` comments).

2. Launch fine-tuning, passing the checkpoint path via `--gpt_model_path`:

```bash
python finetune_sf.py \
  --env=GRF \
  --experiment=academy_single_goal_versus_lazy \
  --save_best_metric=goal_diff \
  --num_workers=8 \
  --gpt_model_path=out/ckpt.pt
```

To train from scratch instead of loading a checkpoint, pass `--gpt_model_path=none`.

Fine-tuned checkpoints are saved to the `train_dir/` directory.

### Evaluation

Run evaluation for a specific benchmark environment (example: SMACv2):

```bash
python all_env_bench.py --env_name smacv2
```

`all_env_bench.py` runs the evaluation suite for supported environments (see the script for available options and defaults).

#### Converting fine-tuned checkpoints (online → eval)

If you fine-tuned the model online (PPO/Sample Factory), convert the resulting checkpoints to the evaluation format:

```bash
python gpt_model_from_sf_to_bench.py \
  --gpt-path out/ckpt.pt \
  --ppo-path train_dir/checkpoint_p0/best.pth \
  --output-path out/ppo_model.pt
```

## Repository Structure

```
.
├── config/              # Training and dataset configuration files
├── docker/              # Docker setup and environment installation scripts
├── envs/                # Environment wrappers and adapters
│   ├── smac_env/        # StarCraft Multi-Agent Challenge
│   ├── grf_env/         # Google Research Football
│   └── pogema_env/      # POGEMA pathfinding
├── gpt/                 # Model architecture and training logic
│   └── finetune_vers/   # Adapt GPT model for fine-tuning
├── utils/               # Dataset loaders and utilities
├── train_actor_critic.py   # Offline training entry point
├── finetune_sf.py          # Online fine-tuning entry point
└── all_env_bench.py        # Multi-environment evaluation script
```



## Citation

If you use this code in your research, please cite:

```bibtex
@article{nesterova2026marl,
  title={MARL-GPT: Foundation Model for Multi-Agent Reinforcement Learning},
  author={Nesterova, Maria and Kolosov, Mikhail and Andreychuk, Anton and Cherepanov, Egor and Bulichev, Oleg and Kovalev, Alexey and Yakovlev, Konstantin and Panov, Aleksandr and Skrynnik, Alexey},
  journal={arXiv preprint arXiv:2604.05943},
  year={2026}
}
```
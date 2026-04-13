# SETTINGS

This document describes detailed configuration options, environment integration, and runtime settings for **MARL-GPT: Foundation Model for Multi-Agent Reinforcement Learning**.

It complements the high-level overview in [README.md](/README.md) and is intended for users who want to:
- Adapt MARL-GPT to new environments
- Customize offline training and dataset loading
- Configure online fine-tuning and evaluation pipelines

## 1. Environments

### 1.1. Adding a New Environment (Offline and Online)

To support a new MARL environment, you must:

1. Implement an environment wrapper inside `envs/` (e.g., `envs/my_env/`).
2. Define positional encoding for observations.
3. Implement dataset reading (for offline mode).
4. Register the environment in the training / fine-tuning / evaluation scripts.

A typical structure for a new environment:

```text
envs/
  my_env/
    __init__.py
    create_env.py
    obs_tokenizer.py
    dataset.py          # if using offline data
    wrapper.py          # inference / PPO wrapper (online & evaluation)
```



### 1.2. Positional Encoding

MARL-GPT relies on a structured positional encoding (`pos_encode`) that augments raw observations before they are passed to the transformer.

For each element `obs[i]`, you must define:

- `pos_time[i]`: Time step of this information  
  - `0` – current step  
  - `1` – one step back  
  - etc.

- `pos_agent[i]`: Agent index to which this information refers  
  - `0` – main (own) agent  
  - `1, 2, 3, ...` – allies (matching rows in `obs_ally`)  
  - `0, 1, ...` – enemies (matching rows in `obs_enemy`).

- `pos_group[i]`: Agent group  
  - `0` – own and allies  
  - `1` – enemies.

- `pos_attr[i]`: Feature/attribute index  
  - All instances of the same feature (across agents or time steps) should share the same attribute index.

#### Reference Implementations

Use these as templates for new environments:

- SMAC: `envs/smac_env/obs_tokenizer.py`, `envs/smac_env/decomposer.py`.
- GRF: `envs/grf_env/obs_tokenizer.py`.


## 2. Offline Training Configuration

Offline training is used to train MARL-GPT on expert trajectories stored as datasets.

### 2.1. Dataset Format

A valid dataset must contain trajectories with the following fields, stored sequentially:

- `observation`
- `action`
- `reward`
- `done` (episode termination flag).

### 2.2. Dataset Reader

Implement a dataset reader for your environment:

- Example: `envs/grf_env/dataset.py`.
- Reusable dataset utilities:
  - `utils/dataset.py`: dataset for a single environment
  - `utils/aggregated_data_loader.py`: aggregates datasets from multiple tasks/folders for one environment
- The reader should:
  - Load trajectories from disk
  - Handle any environment-specific preprocessing.

### 2.3. Dataset Registration

Register your dataset reader in the shared loader:

- File: `utils/multi_env_dataset.py`.


### 2.4. Dataset Loading Configuration

Use `config/dataset_config.json` to configure dataset loading. Typical options include:

- `env_name`: Environment identifier
- `folder_paths`: Path to the dataset file(s)
- `load_part_file`: Controls splitting the dataset file into several parts (e.g., `1` to enable partial loading).

Example snippet:

```json
"grf":
    {
        "config": {"history_len": 6,  "load_part_file": 1},
        "folder_paths": ["dataset/11vs11a11r_5to10",]
    }
```

### 2.5. Training Hyperparameters

Offline training hyperparameters are configured in `config/config-train-critic.py`. Typical settings include:

- Learning rate, batch size, number of training steps
- Model size (number of layers, hidden size, number of heads)
- Environment-related options (e.g., history length, observation shapes).

---

Run offline training via:

```bash
python train_actor_critic.py config/config-train-critic.py
```

## 3. Online Fine-Tuning (Sample Factory–Style PPO)

Online fine-tuning integrates MARL-GPT into a Sample Factory–style PPO training loop.

### 3.1. Entry Point

The main entry point is `finetune_sf.py`.

1. **Register your environment.**  
   Add something like:
   ```python
   from envs.grf_env.create_env import sf_make_gpt_grf
   
   register_env(GRF_ENV_NAME, sf_make_gpt_grf)
   ```

2. **Actor unfreezing** is handled automatically by `FreezeActorLearner` (registered via `gpt/finetune_vers/custom_learner.py`). The actor is frozen at the start and unfrozen after a fixed number of training steps.

### 3.2. Model configuration

The main MARL-GPT model is defined and configured in `gpt/finetune_vers/gpt_actor_critic.py`.

The checkpoint path is passed via the `--gpt_model_path` CLI argument (default: `out/ckpt.pt`).

1. **Fine-tuning a pre-trained model.**
   - Pass the checkpoint path at launch:
     ```bash
     python finetune_sf.py --env=GRF --gpt_model_path=out/ckpt.pt ...
     ```

2. **Training from scratch.**
   - Pass `none` as the path:
     ```bash
     python finetune_sf.py --env=GRF --gpt_model_path=none ...
     ```
   - In this case the model architecture is taken from `make_config()` inside `gpt_actor_critic.py`.
   - Hyperparameters (e.g., depth, width, dropout, attention heads) are configured there.

   > **Checkpoint compatibility note.**  
   > When a checkpoint path is provided, the model architecture (`n_layer`, `n_head`, `n_embd`, `block_size`, `n_attr`, etc.) is taken **entirely from the saved checkpoint**, not from `make_config()`. Changing these values in `make_config()` has no effect during fine-tuning — it only applies when training from scratch (`ckpt_path = None`).  
   >
   > Fields that are **not** part of the architecture (e.g. `gamma`, `loss_type`, `alpha_cql`) are not stored in the checkpoint and will use the values from `make_config()`.  
   >
   > If the checkpoint was produced by `torch.compile`, its state dict keys carry an `_orig_mod.` prefix. This is stripped automatically in `load_model()`.

3. **Freezing and unfreezing the actor.**
   - `self.freeze_actor()` freezes the actor at the beginning of training.
   - The custom learner logic unfreezes it after the specified number of iterations.


### 3.3. History Length

The observation history length (`history_len`) is configured in the environment creation logic (e.g., `create_env.py`). It controls how many past steps are concatenated/encoded for the transformer.

### 3.4. Adding a New Environment to Online Fine-Tuning

1. Implement a wrapper that adds positional encoding, similar to `InferenceSmacWrapper` in `envs/smac_env/wrapper.py`.  
2. Implement a `make_env` function similar to `sf_make_gpt_smac_v2` in `envs/smac_env/create_env.py`.

---

Fine-tuning is started via:

```bash
python finetune_sf.py \
  --env=GRF \
  --experiment=academy_single_goal_versus_lazy \
  --save_best_metric=goal_diff \
  --num_workers=8 \
  --gpt_model_path=out/ckpt.pt
```

## 4. Evaluation Configuration

Evaluation is handled by a combination of environment-specific wrappers and global evaluation scripts.
The main evaluation script is `all_env_bench.py`.

To add a new environment:
1. Add an inference `wrapper` for the target environment.  
2. Create or modify an evaluation config, e.g. similar to `eval_configs/smac/terran_v2.yaml`.  
3. Add the environment entry to the main evaluation file `all_env_bench.py`.
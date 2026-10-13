import json
import time
from pathlib import Path

import numpy as np
import torch

from envs.smac_env.config import SMACGPTInferenceConfig
from envs.smac_env.create_env import make_smac_marl_gpt_v2
from gpt.inference import strip_prefix_from_state_dict
from gpt.model_ac import CriticGPTConfig, CriticWithLoss
from utils.env_observation import reshape_obs


SCENARIO = "terran_5_vs_5"
MAX_STEPS = 128
MODEL_PATH = Path("out/marl-gpt-main.pt")

# Seed для NumPy и PyTorch.
# Метод seed() в обёртке авторов SMACv2 пустой,
# поэтому это не гарантирует повторение того же боя.
np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(2)

print("1. Загружаем MARL-GPT...", flush=True)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

model_config = CriticGPTConfig(**checkpoint["model_args"])
model = CriticWithLoss(model_config)

model.load_state_dict(
    strip_prefix_from_state_dict(checkpoint["model"]),
    strict=True,
)
model.eval()

print("Модель загружена.", flush=True)

env_config = SMACGPTInferenceConfig(
    device="cpu",
    smac_vers=2,
    map_name=SCENARIO,
    map_type="terran",
    num_agents=5,
    history_len=6,
    history_step=1,
    obs_with_race=True,
    obs_per_agent=False,
    max_episode_steps=MAX_STEPS,
)

print("2. Создаём среду:", SCENARIO, flush=True)
env = make_smac_marl_gpt_v2(env_config)

steps = 0
total_reward = 0.0
episode_finished = False
final_stats = {}

try:
    observations, _ = env.reset()
    print("Размер входных наблюдений:", observations["obs"].shape)
    print("3. Начинаем бой...", flush=True)

    started = time.perf_counter()

    with torch.inference_mode():
        for step in range(MAX_STEPS):
            batch = reshape_obs(
                observations,
                model_config.block_size,
                model_config.action_size,
            )

            # Последний токен, как в режиме last_token=True.
            batch["attr_pos"][:, -1] = model_config.n_attr - 1
            batch["obs"][:, -1] = 0

            actions = model.act_actor(
                batch,
                do_sample=False,
            ).reshape(-1).tolist()

            observations, rewards, terminated, truncated, infos = env.step(
                actions
            )

            steps = step + 1
            total_reward += float(np.sum(rewards))
            final_stats = infos[0].get("episode_stats", {})

            if steps == 1 or steps % 16 == 0:
                print(
                    f"Шаг {steps}/{MAX_STEPS}; действия: {actions}",
                    flush=True,
                )

            if all(terminated) or all(truncated):
                episode_finished = True
                break

    elapsed = time.perf_counter() - started

finally:
    env.close()

print("\nПроверка завершена.")
print("Выполнено шагов:", steps)
print("Время боя, секунд:", round(elapsed, 2))
print("Суммарная награда команды:", round(total_reward, 4))
print("Статистика последнего шага:", final_stats)

if episode_finished:
    if "battle_won" in final_stats:
        print("Победа:", bool(final_stats["battle_won"]))
    else:
        print("Бой завершён; признак победы в статистике отсутствует.")
else:
    print("Достигнут лимит проверки. Бой не завершён.")

result = {
    "scenario": SCENARIO,
    "model": str(MODEL_PATH),
    "device": "cpu",
    "num_agents": 5,
    "history_len": 6,
    "sample_actions": False,
    "last_token": True,
    "max_steps": MAX_STEPS,
    "executed_steps": steps,
    "episode_finished": episode_finished,
    "runtime_seconds": elapsed,
    "team_reward": total_reward,
    "episode_stats": final_stats,
    "seed_note": (
        "NumPy и PyTorch: 0; seed() обёртки среды не реализован."
    ),
}

output = Path("results/smac_terran5vs5_demo.json")
output.parent.mkdir(parents=True, exist_ok=True)

def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Неподдерживаемый тип: {type(value)}")

output.write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=json_default,
    ),
    encoding="utf-8",
)

print("Результат сохранён:", output.resolve())

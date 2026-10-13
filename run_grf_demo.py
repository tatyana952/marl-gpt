import json
import time
from pathlib import Path

import numpy as np
import torch

from envs.grf_env.config import GRFConfig, GRFInferenceConfig
from envs.grf_env.grf import GRFv0
from envs.grf_env.wrapper import InferenceGrfWrapper
from gpt.inference import strip_prefix_from_state_dict
from gpt.model_ac import CriticGPTConfig, CriticWithLoss
from utils.env_observation import reshape_obs


SCENARIO = "academy_pass_and_shoot_with_keeper"
MAX_STEPS = 256
MODEL_PATH = Path("out/marl-gpt-main.pt")

torch.manual_seed(0)
np.random.seed(0)
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
print("2. Создаём футбольную среду...", flush=True)

game_config = GRFConfig.get_scenario_config(SCENARIO)

inference_config = GRFInferenceConfig(
    device="cpu",
    map_name=SCENARIO,
    num_agents=2,
    history_len=6,
    history_step=1,
    obs_per_agent=False,
    max_episode_steps=MAX_STEPS,
)

# Обёртка обработки наблюдений из репозитория авторов.
# Автоматический запуск следующего эпизода здесь не нужен.
env = InferenceGrfWrapper(
    GRFv0(game_config.model_dump()),
    inference_config,
)

steps = 0
episode_finished = False
final_stats = {}
player_returns = np.zeros(2, dtype=float)

try:
    observations, _ = env.reset()

    print("Размер входных наблюдений:", observations["obs"].shape)
    print("3. Начинаем эпизод...", flush=True)

    started = time.perf_counter()

    with torch.inference_mode():
        for step in range(MAX_STEPS):
            batch = reshape_obs(
                observations,
                model_config.block_size,
                model_config.action_size,
            )

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
            player_returns += np.asarray(rewards, dtype=float)
            final_stats = infos[0].get("episode_stats", {})

            if steps == 1 or steps % 32 == 0:
                print(
                    f"Шаг {steps}/{MAX_STEPS}; действия: {actions}; "
                    f"награды: {np.asarray(rewards).tolist()}",
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
print("Время эпизода, секунд:", round(elapsed, 2))
print("Накопленные награды игроков:", player_returns.tolist())
print("Статистика последнего шага:", final_stats)

if episode_finished:
    print("Эпизод завершён средой.")
else:
    print("Достигнут лимит проверки. Эпизод не завершён.")

result = {
    "scenario": SCENARIO,
    "model": str(MODEL_PATH),
    "device": "cpu",
    "num_agents": 2,
    "history_len": 6,
    "sample_actions": False,
    "last_token": True,
    "rewards": game_config.rewards,
    "max_steps": MAX_STEPS,
    "executed_steps": steps,
    "episode_finished": episode_finished,
    "runtime_seconds": elapsed,
    "player_returns": player_returns.tolist(),
    "episode_stats": final_stats,
    "seed_note": (
        "NumPy и PyTorch: 0; seed движка явно не задан."
    ),
}

output = Path("results/grf_pass_and_shoot_demo.json")
output.parent.mkdir(parents=True, exist_ok=True)

def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
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

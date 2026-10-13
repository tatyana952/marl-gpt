import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from pogema import GridConfig, pogema_v0

from envs.pogema_env.config import PogemaInferenceConfig
from envs.pogema_env.wrapper import InferencePogemaWrapper
from gpt.inference import strip_prefix_from_state_dict
from gpt.model_ac import CriticGPTConfig, CriticWithLoss
from utils.env_observation import reshape_obs


# Параметры проверочного эксперимента.
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--map-name",
    default="validation-random-seed-000",
)
args = parser.parse_args()

SEED = args.seed
NUM_AGENTS = 8
MAX_STEPS = 256
MODEL_PATH = Path("out/marl-gpt-main.pt")

torch.manual_seed(SEED)
torch.set_num_threads(2)

# Загружаем архитектуру и веса.
print("1. Загружаем модель...", flush=True)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

model_config = CriticGPTConfig(**checkpoint["model_args"])
model = CriticWithLoss(model_config)

state_dict = strip_prefix_from_state_dict(checkpoint["model"])
model.load_state_dict(state_dict, strict=True)
model.eval()

print("Модель загружена.", flush=True)
print("Длина контекста:", model_config.block_size)
print("Количество выходов действий:", model_config.action_size)

# Создаём lifelong-среду.
print("2. Создаём среду...", flush=True)

MAP_NAME = args.map_name
MAP_FILE = Path("eval_configs/01-random/maps.yaml")

maps = yaml.safe_load(MAP_FILE.read_text(encoding="utf-8"))
benchmark_map = maps[MAP_NAME]
map_rows = benchmark_map.strip().splitlines()

print("Карта:", MAP_NAME, flush=True)
print("Размер карты:", len(map_rows), "x", len(map_rows[0]))

grid_config = GridConfig(
    num_agents=NUM_AGENTS,
    map=benchmark_map,
    collision_system="soft",
    obs_radius=5,
    max_episode_steps=MAX_STEPS,
    observation_type="MAPF",
    on_target="restart",
    seed=SEED,
)

base_env = pogema_v0(grid_config=grid_config)

env = InferencePogemaWrapper(
    base_env,
    PogemaInferenceConfig(device="cpu"),
)

observations, infos = env.reset()

print("Размер наблюдений:", tuple(observations["obs"].shape))
print("3. Начинаем эпизод...", flush=True)

started = time.perf_counter()
steps = 0

try:
    with torch.inference_mode():
        for step in range(MAX_STEPS):
            # Дополняем вход до размерностей обученной модели.
            batch = reshape_obs(
                observations,
                model_config.block_size,
                model_config.action_size,
            )

            # Последний токен, как в основном режиме inference.
            batch["attr_pos"][:, -1] = model_config.n_attr - 1
            batch["obs"][:, -1] = 0

            # Actor выбирает действия с вероятностной выборкой.
            actions = model.act_actor(
                batch,
                do_sample=True,
            ).reshape(-1).tolist()

            observations, rewards, terminated, truncated, infos = env.step(
                actions
            )
            steps = step + 1

            if steps == 1 or steps % 32 == 0:
                print(
                    f"Шаг {steps}/{MAX_STEPS}; действия: {actions}",
                    flush=True,
                )

            if all(terminated) or all(truncated):
                break
finally:
    env.close()

elapsed = time.perf_counter() - started
metrics = infos[0].get("metrics", {})
throughput = metrics.get("avg_throughput")

print("\nЭпизод завершён.")
print("Выполнено шагов:", steps)
print("Время эпизода, секунд:", round(elapsed, 2))
print("Метрики среды:", metrics)

if throughput is not None:
    print("Throughput:", throughput)
    print("Достигнуто целей всей группой:", round(throughput * steps))

# Сохраняем результат первого запуска.
result = {
    "model": str(MODEL_PATH),
    "device": "cpu",
    "seed": SEED,
    "num_agents": NUM_AGENTS,
    "map_name": MAP_NAME,
    "map_source": str(MAP_FILE),
    "map_height": len(map_rows),
    "map_width": len(map_rows[0]),
    "collision_system": "soft",
    "max_episode_steps": MAX_STEPS,
    "executed_steps": steps,
    "runtime_seconds": elapsed,
    "metrics": metrics,
}

output = Path(f"results/pogema_{MAP_NAME}_agents{NUM_AGENTS}_seed{SEED}.json")
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

# Мои эксперименты MARL-GPT

## Цель проверки

Проверить работу предобученной MARL-GPT в POGEMA, SMACv2
и Google Research Football на CPU.

Использовались одни и те же предобученные веса
out/marl-gpt-main.pt без дообучения.
Для каждой среды использовалась её обработка наблюдений
из репозитория авторов.

Исходный commit сохранён в results/source_commit.txt.
Контрольная сумма весов сохранена в results/model_sha256.txt.

## Файлы запуска

| Файл | Назначение |
|---|---|
| run_pogema_demo.py | Первый проверочный запуск POGEMA |
| run_pogema_benchmark_one.py | Проверка карты random-000 с seed=0 |
| run_pogema_benchmark_seed.py | Повторные запуски random-000 с изменением seed |
| run_pogema_benchmark.py | Запуски POGEMA с выбором карты |
| run_smac_demo.py | Один бой SMACv2 Terran 5 против 5 |
| run_grf_demo.py | Один эпизод Google Research Football |

## Эксперимент 01. POGEMA

### Первый проверочный запуск

Одна карта, 8 агентов, один процесс, 64 шага.

Достигнуто 39 целей.
Throughput = 39 / 64 = 0,609375.
Время выполнения: 60,24 секунды.

Результат: results/pogema_demo_seed0.json.

### Карта validation-random-seed-000

Использовались параметры из
eval_configs/01-random/01-random.yaml:
8 агентов, 256 шагов, разрешение конфликтов soft.

Проведены три запуска с seed=0, seed=1 и seed=2.

| Seed | Достигнуто целей | Throughput | Время, с |
|---|---:|---:|---:|
| 0 | 135 | 0,52734375 | 252,54 |
| 1 | 146 | 0,57031250 | 257,54 |
| 2 | 139 | 0,54296875 | 266,31 |

Средний throughput: 0,546875.
Выборочное стандартное отклонение: 0,021749.
Среднее число достигнутых целей: 140.
Среднее время выполнения: 258,80 секунды.

Throughput вычисляется как суммарное число целей,
достигнутых группой, делённое на число шагов среды.

Файлы результатов:
- results/pogema_random000_agents8_seed0.json
- results/pogema_random000_agents8_seed1.json
- results/pogema_random000_agents8_seed2.json
- results/pogema_random000_agents8_summary.csv
- results/pogema_random000_agents8_summary.json

### Карта validation-random-seed-001

Один запуск, 8 агентов, 256 шагов, seed=0.

Достигнуто 472 цели.
Throughput = 472 / 256 = 1,84375.

Результат:
results/pogema_validation-random-seed-001_agents8_seed0.json.

Дополнительная таблица результатов POGEMA:
results/pogema_evaluation.csv.

## Эксперимент 02. SMACv2

Проведена техническая проверка создания среды и одного шага.
Затем выполнен один бой с предобученной MARL-GPT.

Сценарий: Terran 5 против 5.
Использовались параметры из eval_configs/smac/terran_v2.yaml:
история из 6 наблюдений, признак расы,
выбор наиболее вероятного действия.
Лимит проверки: 128 шагов.

Бой завершился победой за 92 шага.
Уничтожены все 5 противников, потерян 1 союзник.

Результат: results/smac_terran5vs5_demo.json.

## Эксперимент 03. Google Research Football

Проведена техническая проверка создания среды и одного шага.
Затем выполнен один эпизод с предобученной MARL-GPT.

Сценарий: academy_pass_and_shoot_with_keeper.
Количество управляемых игроков: 2.
Размер наблюдения каждого игрока: 115.
Количество действий: 19.

Использовались параметры из eval_configs/grf/grf.yaml:
история из 6 наблюдений, выбор наиболее вероятного действия,
последний служебный токен.
Лимит проверки: 256 шагов.

Эпизод завершился забитым голом за 31 шаг.
Итоговый счёт: 1:0.
Время выполнения эпизода: 5,94 секунды.
Накопленные награды игроков: [1.0, 1.0].

Обе награды относятся к одному командному голу.

Результат: results/grf_pass_and_shoot_demo.json.

Проверенные версии для GRF:
- PyTorch: 2.6.0+cpu
- Google Research Football: 2.10.2
- Gym: 0.23.1
- NumPy: 1.26.4
- OpenCV: пакет opencv-python 4.10.0.84
- six: 1.17.0

## Команды повторного запуска

Команды выполняются из корня репозитория при наличии
настроенных окружений, установленных сред и весов модели.

### POGEMA, окружение marl_gpt

    conda activate marl_gpt
    python -u run_pogema_demo.py
    python -u run_pogema_benchmark_one.py
    python -u run_pogema_benchmark_seed.py --seed 1
    python -u run_pogema_benchmark_seed.py --seed 2
    python -u run_pogema_benchmark.py --map-name validation-random-seed-001 --seed 0

### SMACv2, окружение marl_smac

    conda activate marl_smac
    python -u run_smac_demo.py

### GRF, окружение marl_grf

    conda activate marl_grf
    python -u run_grf_demo.py

При повторном запуске скриптов файлы результатов
с теми же именами могут быть перезаписаны.

## Выводы и ограничения

Запуски подтвердили работоспособность предобученной MARL-GPT
в трёх разных средах на CPU без дообучения.

В POGEMA модель достигала целей, в SMACv2 выиграла
проверочный бой, в GRF завершила сценарий забитым голом.

Это предварительная экспериментальная проверка.
Количество запусков недостаточно для воспроизведения
средних результатов статьи или оценки надёжности политики.

Знак ± для random-000 обозначает выборочное стандартное
отклонение трёх запусков, а не доверительный интервал.
Результат random-001 получен в одном запуске.

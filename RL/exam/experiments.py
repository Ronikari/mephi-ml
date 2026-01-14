"""
Скрипт для проведения контролируемых экспериментов.
Эксперимент 1: Сравнение алгоритмов PPO и A2C
Эксперимент 2: Влияние гиперпараметра gamma на обучение PPO
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import pandas as pd
from train import make_env, SEED, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES

# Создаём директории для результатов экспериментов
os.makedirs("experiments", exist_ok=True)
os.makedirs("experiments/logs", exist_ok=True)
os.makedirs("experiments/models", exist_ok=True)
os.makedirs("experiments/plots", exist_ok=True)


def train_experiment(algorithm, name, **kwargs):
    """
    Обучает агента с заданными параметрами для эксперимента.
    
    Args:
        algorithm: Класс алгоритма (PPO или A2C)
        name: Имя эксперимента для сохранения
        **kwargs: Гиперпараметры алгоритма
    
    Returns:
        Обученная модель и путь к логам
    """
    print(f"\n{'='*60}")
    print(f"Эксперимент: {name}")
    print(f"{'='*60}\n")
    
    # Создаём среду
    env = DummyVecEnv([make_env(seed=SEED)])
    eval_env = DummyVecEnv([make_env(seed=SEED)])
    
    # Базовые параметры
    base_params = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "seed": SEED,
    }
    
    # Добавляем tensorboard_log только если tensorboard установлен
    try:
        import tensorboard
        base_params["tensorboard_log"] = f"./experiments/logs/tensorboard_{name}/"
    except ImportError:
        pass  # Tensorboard не установлен, пропускаем логирование
    
    # Параметры по умолчанию для PPO
    if algorithm == PPO:
        default_params = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    # Параметры по умолчанию для A2C
    elif algorithm == A2C:
        default_params = {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    else:
        default_params = {}
    
    # Объединяем параметры
    params = {**base_params, **default_params, **kwargs}
    
    # Создаём модель
    model = algorithm(**params)
    
    # Callback для оценки
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./experiments/models/{name}_best/",
        log_path=f"./experiments/logs/{name}/",
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=N_EVAL_EPISODES,
    )
    
    # Обучаем
    print(f"Начинаем обучение на {TOTAL_TIMESTEPS} шагов...")
    
    # Проверяем наличие rich для progress bar
    try:
        import rich
        use_progress_bar = True
    except ImportError:
        print("⚠️  Rich не установлен. Progress bar отключен.")
        print("   Для установки: pip install rich")
        use_progress_bar = False
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=use_progress_bar,
    )
    
    # Сохраняем модель
    model_path = f"./experiments/models/{name}_final.zip"
    model.save(model_path)
    print(f"\nМодель сохранена: {model_path}")
    
    return model, f"./experiments/logs/{name}/"


def experiment_1_compare_algorithms():
    """
    Эксперимент 1: Сравнение алгоритмов PPO и A2C
    
    Гипотеза: PPO должен показать более стабильное обучение и лучшие 
    результаты благодаря механизму clipping, который ограничивает 
    слишком большие изменения политики.
    """
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 1: Сравнение алгоритмов PPO и A2C")
    print("="*60)
    print("\nГипотеза: PPO должен показать более стабильное обучение")
    print("и лучшие результаты благодаря механизму clipping.\n")
    
    # Обучаем PPO
    model_ppo, log_dir_ppo = train_experiment(
        PPO,
        "exp1_ppo",
        gamma=0.99,
    )
    
    # Обучаем A2C
    model_a2c, log_dir_a2c = train_experiment(
        A2C,
        "exp1_a2c",
        gamma=0.99,
    )
    
    # Строим сравнительный график
    plot_comparison(
        [log_dir_ppo, log_dir_a2c],
        ["PPO", "A2C"],
        "Сравнение алгоритмов PPO и A2C",
        "experiments/plots/exp1_comparison.png"
    )
    
    return model_ppo, model_a2c


def experiment_2_gamma_influence():
    """
    Эксперимент 2: Влияние гиперпараметра gamma на обучение
    
    Гипотеза: При увеличении gamma (коэффициент дисконтирования) 
    агент будет больше учитывать долгосрочные награды и лучше 
    планировать свои действия, что должно улучшить финальные результаты.
    Однако слишком высокий gamma может замедлить обучение.
    """
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 2: Влияние гиперпараметра gamma на обучение")
    print("="*60)
    print("\nГипотеза: При увеличении gamma агент будет больше")
    print("учитывать долгосрочные награды и лучше планировать.\n")
    
    # Тестируем разные значения gamma
    gamma_values = [0.90, 0.95, 0.99, 0.999]
    models = []
    log_dirs = []
    names = []
    
    for gamma in gamma_values:
        name = f"exp2_gamma_{gamma}"
        model, log_dir = train_experiment(
            PPO,
            name,
            gamma=gamma,
        )
        models.append(model)
        log_dirs.append(log_dir)
        names.append(f"PPO (γ={gamma})")
    
    # Строим сравнительный график
    plot_comparison(
        log_dirs,
        names,
        "Влияние гиперпараметра gamma на обучение",
        "experiments/plots/exp2_gamma_comparison.png"
    )
    
    return models


def plot_comparison(log_dirs, labels, title, save_path):
    """
    Строит сравнительный график обучения нескольких агентов.
    
    Args:
        log_dirs: Список путей к директориям с логами
        labels: Список меток для каждого агента
        title: Заголовок графика
        save_path: Путь для сохранения графика
    """
    plt.figure(figsize=(14, 8))
    
    for log_dir, label in zip(log_dirs, labels):
        try:
            # Читаем логи оценки
            eval_log_file = os.path.join(log_dir, "evaluations.npz")
            if os.path.exists(eval_log_file):
                data = np.load(eval_log_file)
                timesteps = data["timesteps"]
                results = data["results"]
                
                # Вычисляем среднее и стандартное отклонение
                mean_rewards = results.mean(axis=1)
                std_rewards = results.std(axis=1)
                
                # Строим график с доверительным интервалом
                plt.plot(timesteps, mean_rewards, label=label, linewidth=2)
                plt.fill_between(
                    timesteps,
                    mean_rewards - std_rewards,
                    mean_rewards + std_rewards,
                    alpha=0.2
                )
            else:
                print(f"Файл логов не найден: {eval_log_file}")
        except Exception as e:
            print(f"Ошибка при чтении логов для {label}: {e}")
    
    plt.xlabel("Количество шагов", fontsize=12)
    plt.ylabel("Средняя награда за эпизод", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nГрафик сохранён: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("Проведение контролируемых экспериментов")
    print("="*60)
    print(f"Seed: {SEED}")
    print(f"Общее количество шагов на эксперимент: {TOTAL_TIMESTEPS}")
    
    # Проводим эксперименты
    print("\nЗапуск эксперимента 1...")
    exp1_models = experiment_1_compare_algorithms()
    
    print("\nЗапуск эксперимента 2...")
    exp2_models = experiment_2_gamma_influence()
    
    print("\n" + "="*60)
    print("Все эксперименты завершены!")
    print("="*60)


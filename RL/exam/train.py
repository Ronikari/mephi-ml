"""
Основной скрипт для обучения RL-агента на среде LunarLander-v2.
Использует stable-baselines3 для обучения агента алгоритмом PPO.
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from datetime import datetime

# Фиксируем seed для воспроизводимости
SEED = 42
np.random.seed(SEED)

# Параметры обучения
TOTAL_TIMESTEPS = 500000  # Достаточно для сходимости на CPU за ~30 минут
EVAL_FREQ = 10000
N_EVAL_EPISODES = 10

# Создаём директории для результатов
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)


def make_env(seed=SEED):
    """Создаёт и оборачивает среду Gymnasium."""
    def _init():
        env = gym.make("LunarLander-v2")
        env = Monitor(env, filename=None, allow_early_resets=True)
        env.reset(seed=seed)
        return env
    return _init


def train_agent(algorithm="PPO", **kwargs):
    """
    Обучает агента на среде LunarLander-v2.
    
    Args:
        algorithm: Название алгоритма ("PPO", "A2C")
        **kwargs: Дополнительные гиперпараметры для алгоритма
    
    Returns:
        Обученная модель
    """
    print(f"\n{'='*60}")
    print(f"Обучение агента: {algorithm}")
    print(f"{'='*60}\n")
    
    # Создаём среду
    env = DummyVecEnv([make_env(seed=SEED)])
    eval_env = DummyVecEnv([make_env(seed=SEED)])
    
    # Параметры по умолчанию для PPO
    default_params = {
        "policy": "MlpPolicy",
        "env": env,
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
        "verbose": 1,
        "seed": SEED,
    }
    
    # Добавляем tensorboard_log только если tensorboard установлен
    try:
        import tensorboard
        default_params["tensorboard_log"] = "./logs/tensorboard/"
    except ImportError:
        print("⚠️  Tensorboard не установлен. Логирование в tensorboard отключено.")
        print("   Для установки: pip install tensorboard")
    
    # Обновляем параметры переданными значениями
    default_params.update(kwargs)
    
    # Создаём модель
    if algorithm == "PPO":
        model = PPO(**default_params)
    elif algorithm == "A2C":
        from stable_baselines3 import A2C
        # Убираем параметры, специфичные для PPO
        a2c_params = {k: v for k, v in default_params.items() 
                     if k not in ["n_steps", "n_epochs", "clip_range"]}
        a2c_params.update({
            "n_steps": 5,
            "learning_rate": 7e-4,
        })
        a2c_params.update({k: v for k, v in kwargs.items() if k not in ["n_steps", "n_epochs", "clip_range"]})
        model = A2C(**a2c_params)
    else:
        raise ValueError(f"Неподдерживаемый алгоритм: {algorithm}")
    
    # Callback для оценки во время обучения
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{algorithm.lower()}_best/",
        log_path=f"./logs/{algorithm.lower()}/",
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=N_EVAL_EPISODES,
    )
    
    # Callback для сохранения чекпоинтов
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"./models/{algorithm.lower()}_checkpoints/",
        name_prefix=f"{algorithm.lower()}_model",
    )
    
    # Обучаем модель
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
        callback=[eval_callback, checkpoint_callback],
        progress_bar=use_progress_bar,
    )
    
    # Сохраняем финальную модель
    model_path = f"./models/{algorithm.lower()}_final.zip"
    model.save(model_path)
    print(f"\nМодель сохранена: {model_path}")
    
    return model


def plot_training_progress(log_dir, algorithm_name, save_path=None):
    """
    Строит график обучения на основе логов.
    
    Args:
        log_dir: Путь к директории с логами
        algorithm_name: Название алгоритма для подписи
        save_path: Путь для сохранения графика
    """
    try:
        import pandas as pd
        
        # Читаем логи
        log_file = os.path.join(log_dir, "monitor.csv")
        if not os.path.exists(log_file):
            print(f"Файл логов не найден: {log_file}")
            return
        
        df = pd.read_csv(log_file, skiprows=1)
        
        # Строим график
        plt.figure(figsize=(12, 6))
        plt.plot(df["r"].rolling(window=100).mean(), label="Средняя награда (окно 100)", alpha=0.7)
        plt.plot(df["r"], label="Награда за эпизод", alpha=0.3)
        plt.xlabel("Эпизод")
        plt.ylabel("Награда")
        plt.title(f"Обучение агента {algorithm_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"График сохранён: {save_path}")
        
        plt.close()
    except Exception as e:
        print(f"Ошибка при построении графика: {e}")


if __name__ == "__main__":
    print("="*60)
    print("Обучение RL-агента на среде LunarLander-v2")
    print("="*60)
    print(f"Seed: {SEED}")
    print(f"Общее количество шагов: {TOTAL_TIMESTEPS}")
    print(f"Частота оценки: каждые {EVAL_FREQ} шагов")
    print(f"Количество эпизодов для оценки: {N_EVAL_EPISODES}")
    
    # Обучаем базового агента PPO
    model = train_agent("PPO")
    
    print("\n" + "="*60)
    print("Обучение завершено!")
    print("="*60)


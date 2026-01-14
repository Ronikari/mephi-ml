"""
Скрипт для визуализации результатов обучения и создания анимаций.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO, A2C
import gymnasium as gym
from train import SEED

# Настройка стиля графиков
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_training_curves(log_dirs, labels, save_path="results/training_curves.png"):
    """
    Строит графики обучения на основе логов оценки.
    
    Args:
        log_dirs: Список путей к директориям с логами
        labels: Список меток для каждого агента
        save_path: Путь для сохранения графика
    """
    plt.figure(figsize=(14, 8))
    
    for log_dir, label in zip(log_dirs, labels):
        try:
            eval_log_file = os.path.join(log_dir, "evaluations.npz")
            if os.path.exists(eval_log_file):
                data = np.load(eval_log_file)
                timesteps = data["timesteps"]
                results = data["results"]
                
                mean_rewards = results.mean(axis=1)
                std_rewards = results.std(axis=1)
                
                plt.plot(timesteps, mean_rewards, label=label, linewidth=2.5, marker="o", markersize=4)
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
    
    plt.xlabel("Количество шагов обучения", fontsize=13)
    plt.ylabel("Средняя награда за эпизод", fontsize=13)
    plt.title("Кривые обучения агентов", fontsize=15, fontweight="bold")
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"График сохранён: {save_path}")
    plt.close()


def create_episode_animation(model_path, n_episodes=3, save_path="videos/episode_animation.mp4"):
    """
    Создаёт анимацию нескольких эпизодов работы агента.
    
    Args:
        model_path: Путь к модели
        n_episodes: Количество эпизодов для записи
        save_path: Путь для сохранения видео
    """
    print(f"\nСоздание анимации для модели: {model_path}")
    
    # Определяем тип алгоритма
    if "a2c" in model_path.lower():
        model_class = A2C
    else:
        model_class = PPO
    
    # Загружаем модель
    try:
        model = model_class.load(model_path)
        print("Модель загружена успешно")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return
    
    # Создаём среду с записью видео
    # Проверяем наличие moviepy и ffmpeg
    try:
        import moviepy
        # Проверяем наличие ffmpeg
        try:
            import imageio_ffmpeg
            imageio_ffmpeg.get_ffmpeg_exe()
            env = gym.make("LunarLander-v2", render_mode="rgb_array")
            
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                os.path.dirname(save_path),
                name_prefix=os.path.basename(save_path).replace(".mp4", ""),
                episode_trigger=lambda x: True,
            )
        except (RuntimeError, Exception) as e:
            print("⚠️  FFmpeg не найден. Запись видео невозможна.")
            print("   Для установки ffmpeg:")
            print("     macOS: brew install ffmpeg")
            print("     Linux: sudo apt-get install ffmpeg")
            print("   Анимация будет создана без записи видео.")
            env = gym.make("LunarLander-v2", render_mode="human")
    except ImportError:
        print("⚠️  MoviePy не установлен. Запись видео невозможна.")
        print("   Для установки: pip install moviepy")
        print("   Анимация будет создана без записи видео.")
        env = gym.make("LunarLander-v2", render_mode="human")
    
    # Запускаем эпизоды
    for episode in range(n_episodes):
        obs, info = env.reset(seed=SEED + episode)
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        print(f"Эпизод {episode + 1}: Награда = {total_reward:.2f}")
    
    env.close()
    print(f"Анимация сохранена в директории: {os.path.dirname(save_path)}")


def plot_evaluation_comparison(evaluation_stats_paths, labels, save_path="results/evaluation_comparison.png"):
    """
    Строит сравнительный график результатов оценки агентов.
    
    Args:
        evaluation_stats_paths: Список путей к CSV файлам со статистикой
        labels: Список меток для каждого агента
        save_path: Путь для сохранения графика
    """
    plt.figure(figsize=(14, 8))
    
    for stats_path, label in zip(evaluation_stats_paths, labels):
        try:
            df = pd.read_csv(stats_path)
            plt.plot(df["episode"], df["reward"], label=label, marker="o", linewidth=2, markersize=6)
        except Exception as e:
            print(f"Ошибка при чтении {stats_path}: {e}")
    
    plt.xlabel("Номер эпизода", fontsize=13)
    plt.ylabel("Награда за эпизод", fontsize=13)
    plt.title("Сравнение результатов оценки агентов", fontsize=15, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"График сохранён: {save_path}")
    plt.close()


def plot_reward_distribution(evaluation_stats_paths, labels, save_path="results/reward_distribution.png"):
    """
    Строит график распределения наград для сравнения агентов.
    
    Args:
        evaluation_stats_paths: Список путей к CSV файлам со статистикой
        labels: Список меток для каждого агента
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # График распределения
    ax1 = axes[0]
    for stats_path, label in zip(evaluation_stats_paths, labels):
        try:
            df = pd.read_csv(stats_path)
            ax1.hist(df["reward"], alpha=0.6, label=label, bins=15, edgecolor="black")
        except Exception as e:
            print(f"Ошибка при чтении {stats_path}: {e}")
    
    ax1.set_xlabel("Награда за эпизод", fontsize=12)
    ax1.set_ylabel("Частота", fontsize=12)
    ax1.set_title("Распределение наград", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Box plot для сравнения
    ax2 = axes[1]
    data_for_box = []
    labels_for_box = []
    for stats_path, label in zip(evaluation_stats_paths, labels):
        try:
            df = pd.read_csv(stats_path)
            data_for_box.append(df["reward"].values)
            labels_for_box.append(label)
        except Exception as e:
            print(f"Ошибка при чтении {stats_path}: {e}")
    
    if data_for_box:
        bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)
        ax2.set_ylabel("Награда за эпизод", fontsize=12)
        ax2.set_title("Сравнение распределений наград", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"График сохранён: {save_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("Визуализация результатов обучения")
    print("="*60)
    
    # Пример использования
    if len(sys.argv) > 1 and sys.argv[1] == "animate":
        # Создаём анимацию
        model_path = sys.argv[2] if len(sys.argv) > 2 else "models/ppo_final.zip"
        create_episode_animation(model_path, n_episodes=3)
    else:
        print("\nИспользование:")
        print("  python visualize.py animate [путь_к_модели]")
        print("\nДля создания графиков используйте функции из experiments.py")


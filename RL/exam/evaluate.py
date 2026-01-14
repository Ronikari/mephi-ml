"""
Скрипт для оценки обученных агентов.
Выполняет оценку на нескольких эпизодах и собирает статистику.
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
from train import make_env, SEED


def evaluate_agent(model_path, n_episodes=20, render=False, save_video=False, video_path=None):
    """
    Оценивает обученного агента на нескольких эпизодах.
    
    Args:
        model_path: Путь к сохранённой модели
        n_episodes: Количество эпизодов для оценки
        render: Показывать ли визуализацию
        save_video: Сохранять ли видео эпизодов
        video_path: Путь для сохранения видео
    
    Returns:
        Словарь со статистикой (средняя награда, std, min, max)
    """
    print(f"\n{'='*60}")
    print(f"Оценка агента: {model_path}")
    print(f"{'='*60}\n")
    
    # Определяем тип алгоритма по пути
    if "a2c" in model_path.lower():
        model_class = A2C
    else:
        model_class = PPO
    
    # Загружаем модель
    try:
        model = model_class.load(model_path)
        print(f"Модель успешно загружена: {model_path}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None
    
    # Создаём среду
    # Проверяем наличие moviepy и ffmpeg для записи видео
    if save_video:
        try:
            import moviepy
            # Проверяем наличие ffmpeg
            try:
                import imageio_ffmpeg
                imageio_ffmpeg.get_ffmpeg_exe()
                env = gym.make("LunarLander-v2", render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda x: True)
            except (RuntimeError, Exception) as e:
                print("⚠️  FFmpeg не найден. Запись видео отключена.")
                print("   Для установки ffmpeg:")
                print("     macOS: brew install ffmpeg")
                print("     Linux: sudo apt-get install ffmpeg")
                save_video = False
                if render:
                    env = gym.make("LunarLander-v2", render_mode="human")
                else:
                    env = gym.make("LunarLander-v2")
        except ImportError:
            print("⚠️  MoviePy не установлен. Запись видео отключена.")
            print("   Для установки: pip install moviepy")
            save_video = False
            if render:
                env = gym.make("LunarLander-v2", render_mode="human")
            else:
                env = gym.make("LunarLander-v2")
    elif render:
        env = gym.make("LunarLander-v2", render_mode="human")
    else:
        env = gym.make("LunarLander-v2")
    
    env = Monitor(env, filename=None, allow_early_resets=True)
    env.reset(seed=SEED)
    
    # Оцениваем агента
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=SEED + episode)
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 5 == 0:
            print(f"Эпизод {episode + 1}/{n_episodes}: Награда = {total_reward:.2f}, Шагов = {steps}")
    
    env.close()
    
    # Вычисляем статистику
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }
    
    # Выводим результаты
    print(f"\n{'='*60}")
    print("Результаты оценки:")
    print(f"{'='*60}")
    print(f"Количество эпизодов: {n_episodes}")
    print(f"Средняя награда: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Минимальная награда: {stats['min_reward']:.2f}")
    print(f"Максимальная награда: {stats['max_reward']:.2f}")
    print(f"Средняя длина эпизода: {stats['mean_length']:.2f} шагов")
    print(f"{'='*60}\n")
    
    return stats


def compare_agents(model_paths, labels, n_episodes=20):
    """
    Сравнивает несколько агентов и строит сравнительный график.
    
    Args:
        model_paths: Список путей к моделям
        labels: Список меток для каждой модели
        n_episodes: Количество эпизодов для оценки каждого агента
    
    Returns:
        DataFrame со статистикой всех агентов
    """
    print("\n" + "="*60)
    print("Сравнение агентов")
    print("="*60 + "\n")
    
    all_stats = []
    
    for model_path, label in zip(model_paths, labels):
        stats = evaluate_agent(model_path, n_episodes=n_episodes)
        if stats:
            stats["model"] = label
            all_stats.append(stats)
    
    # Создаём DataFrame для сравнения
    comparison_data = []
    for stats in all_stats:
        comparison_data.append({
            "Модель": stats["model"],
            "Средняя награда": f"{stats['mean_reward']:.2f}",
            "Std": f"{stats['std_reward']:.2f}",
            "Min": f"{stats['min_reward']:.2f}",
            "Max": f"{stats['max_reward']:.2f}",
        })
    
    df = pd.DataFrame(comparison_data)
    print("\nСравнительная таблица:")
    print(df.to_string(index=False))
    print()
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python evaluate.py <путь_к_модели> [количество_эпизодов]")
        print("\nПримеры:")
        print("  python evaluate.py models/ppo_final.zip 20")
        print("  python evaluate.py experiments/models/exp1_ppo_best/best_model.zip 20")
        sys.exit(1)
    
    model_path = sys.argv[1]
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    # Создаём директорию для видео
    os.makedirs("videos", exist_ok=True)
    
    # Оцениваем агента
    stats = evaluate_agent(
        model_path,
        n_episodes=n_episodes,
        render=False,
        save_video=True,
        video_path="videos/evaluation"
    )
    
    if stats:
        # Сохраняем статистику
        stats_df = pd.DataFrame({
            "episode": range(1, len(stats["episode_rewards"]) + 1),
            "reward": stats["episode_rewards"],
            "length": stats["episode_lengths"],
        })
        stats_path = "results/evaluation_stats.csv"
        os.makedirs("results", exist_ok=True)
        stats_df.to_csv(stats_path, index=False)
        print(f"Статистика сохранена: {stats_path}")


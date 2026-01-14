"""
Основной скрипт для запуска всего проекта:
1. Базовое обучение агента
2. Проведение экспериментов
3. Оценка всех агентов
4. Создание визуализаций и анимаций
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Выполняет команду и выводит описание."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    print(f"Выполняется: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n⚠️  Ошибка при выполнении: {description}")
        return False
    else:
        print(f"\n✅ Успешно завершено: {description}")
        return True


def main():
    """Основная функция для запуска всех этапов проекта."""
    print("="*60)
    print("ЗАПУСК ПОЛНОГО ЦИКЛА ПРОЕКТА RL")
    print("="*60)
    
    # Создаём необходимые директории
    directories = [
        "models", "logs", "results", "videos",
        "experiments/models", "experiments/logs", "experiments/plots"
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    
    steps = []
    
    # Шаг 1: Базовое обучение
    if len(sys.argv) > 1 and "skip-train" in sys.argv:
        print("\n⏭️  Пропуск базового обучения (используйте существующие модели)")
    else:
        steps.append(("python train.py", "Шаг 1: Базовое обучение PPO агента"))
    
    # Шаг 2: Эксперименты
    if len(sys.argv) > 1 and "skip-experiments" in sys.argv:
        print("\n⏭️  Пропуск экспериментов")
    else:
        steps.append(("python experiments.py", "Шаг 2: Проведение контролируемых экспериментов"))
    
    # Выполняем шаги 1 и 2 (обучение и эксперименты)
    for cmd, description in steps:
        success = run_command(cmd, description)
        if not success:
            print(f"\n❌ Произошла ошибка на этапе: {description}")
            print("Проект можно продолжить вручную, выполнив оставшиеся команды.")
            return
    
    # Шаг 3: Оценка агентов (выполняется ПОСЛЕ обучения)
    print("\n" + "="*60)
    print("Шаг 3: Оценка обученных агентов")
    print("="*60)
    
    # Список всех возможных моделей для оценки
    all_possible_models = [
        ("models/ppo_final.zip", "Базовый PPO (финальная модель)"),
        ("models/ppo_best/best_model.zip", "Базовый PPO (лучшая модель)"),
        ("experiments/models/exp1_ppo_final.zip", "Эксперимент 1: PPO (финальная)"),
        ("experiments/models/exp1_ppo_best/best_model.zip", "Эксперимент 1: PPO (лучшая)"),
        ("experiments/models/exp1_a2c_final.zip", "Эксперимент 1: A2C (финальная)"),
        ("experiments/models/exp1_a2c_best/best_model.zip", "Эксперимент 1: A2C (лучшая)"),
    ]
    
    # Добавляем модели из эксперимента 2 (gamma)
    for gamma in [0.90, 0.95, 0.99, 0.999]:
        all_possible_models.extend([
            (f"experiments/models/exp2_gamma_{gamma}_final.zip", f"Эксперимент 2: PPO (γ={gamma}, финальная)"),
            (f"experiments/models/exp2_gamma_{gamma}_best/best_model.zip", f"Эксперимент 2: PPO (γ={gamma}, лучшая)"),
        ])
    
    # Фильтруем только существующие модели
    models_to_evaluate = []
    for model_path, description in all_possible_models:
        if os.path.exists(model_path):
            models_to_evaluate.append((model_path, description))
    
    # Оцениваем каждую найденную модель
    evaluation_success = False
    if models_to_evaluate:
        print(f"\nНайдено {len(models_to_evaluate)} обученных моделей для оценки:\n")
        for model_path, description in models_to_evaluate:
            cmd = f"python evaluate.py {model_path} 20"
            if run_command(cmd, f"Оценка: {description}"):
                evaluation_success = True
    else:
        print("\n⚠️  Не найдено обученных моделей для оценки.")
        print("   Убедитесь, что шаги 1 и 2 выполнены успешно.")
        print("   Ожидаемые пути к моделям:")
        print("     - models/ppo_final.zip")
        print("     - experiments/models/exp1_ppo_best/best_model.zip")
        print("     - experiments/models/exp1_a2c_best/best_model.zip")
    
    # Шаг 4: Создание анимаций (выполняется ПОСЛЕ оценки)
    print("\n" + "="*60)
    print("Шаг 4: Создание анимаций")
    print("="*60)
    
    # Создаём анимацию для лучшей модели (приоритет лучшим моделям из экспериментов)
    best_model = None
    
    # Сначала ищем лучшие модели из экспериментов
    priority_models = [
        "experiments/models/exp1_ppo_best/best_model.zip",
        "experiments/models/exp1_a2c_best/best_model.zip",
        "models/ppo_best/best_model.zip",
        "models/ppo_final.zip",
    ]
    
    for model_path in priority_models:
        if os.path.exists(model_path):
            best_model = model_path
            break
    
    # Если не нашли приоритетные, используем любую найденную модель
    if not best_model and models_to_evaluate:
        best_model = models_to_evaluate[0][0]
    
    if best_model:
        cmd = f"python visualize.py animate {best_model}"
        run_command(cmd, f"Создание анимации работы агента ({os.path.basename(best_model)})")
    else:
        print("\n⚠️  Не найдено моделей для создания анимации")
        print("   Убедитесь, что шаги 1 и 2 выполнены успешно.")
    
    print("\n" + "="*60)
    print("✅ ВСЕ ЭТАПЫ ПРОЕКТА ЗАВЕРШЕНЫ!")
    print("="*60)
    print("\nРезультаты сохранены в:")
    print("  - models/          : Обученные модели")
    print("  - experiments/     : Результаты экспериментов")
    print("  - results/         : Статистика оценки")
    print("  - videos/          : Видео эпизодов")
    print("  - experiments/plots/: Графики сравнения")
    print("\nДля просмотра результатов проверьте файлы в этих директориях.")


if __name__ == "__main__":
    main()


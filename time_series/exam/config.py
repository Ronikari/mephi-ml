"""
Конфигурационный файл проекта.

Содержит все настройки и параметры для работы бота и моделей машинного обучения.
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv(find_dotenv())

# ===== ПУТИ К ДИРЕКТОРИЯМ =====
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "tmp"

# Создание директорий, если они не существуют
LOGS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Путь к файлу логов запросов
LOG_FILE = LOGS_DIR / "requests_log.csv"

# ===== ПАРАМЕТРЫ ДАННЫХ =====
# Период загрузки исторических данных
START_DATE = '2022-01-01'
END_DATE = '2024-01-01'

# Доля тестовых данных для оценки моделей
TEST_SIZE = 0.2  # 20% данных

# ===== ПАРАМЕТРЫ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ =====

# Random Forest: размер окна для лаговых признаков
LAG_WINDOW = 30

# ===== НАСТРОЙКИ TELEGRAM БОТА =====
# Токен бота (загружается из переменных окружения)
BOT_TOKEN = os.getenv("BOT_TOKEN")

# ===== ИНИЦИАЛИЗАЦИЯ =====
# Создание файла логов с заголовками, если он не существует
if not LOG_FILE.exists():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("user_id,timestamp,ticker,amount,forecast_days,best_model,metric_value,profit\n")
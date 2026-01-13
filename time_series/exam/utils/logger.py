"""
Модуль для логирования запросов пользователей.

Записывает информацию о каждом анализе в CSV файл для последующего анализа.
"""

import pandas as pd
from datetime import datetime
import logging

from config import LOG_FILE

logger = logging.getLogger(__name__)


def log_request(
    user_id: int,
    timestamp: datetime,
    ticker: str,
    amount: float,
    forecast_days: int,
    best_model: str,
    metric_value: float,
    profit: float
) -> None:
    """
    Записывает информацию о запросе пользователя в лог-файл.
    
    Args:
        user_id: ID пользователя в Telegram
        timestamp: Время запроса
        ticker: Тикер акции
        amount: Сумма инвестиции
        forecast_days: Период прогноза в днях
        best_model: Название выбранной модели
        metric_value: Значение метрики RMSE
        profit: Процент прибыли
    """
    try:
        log_entry = {
            'user_id': user_id,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': ticker,
            'amount': amount,
            'forecast_days': forecast_days,
            'best_model': best_model,
            'metric_value': metric_value,
            'profit': profit
        }
        
        # Создаем DataFrame с записью
        df = pd.DataFrame([log_entry])
        
        # Определяем правильный порядок столбцов
        expected_columns = ['user_id', 'timestamp', 'ticker', 'amount', 'forecast_days', 
                          'best_model', 'metric_value', 'profit']
        
        # Убеждаемся, что столбцы в правильном порядке
        df = df[expected_columns]
        
        # Добавляем в CSV файл
        if LOG_FILE.exists():
            # Проверяем заголовок существующего файла
            try:
                existing_df = pd.read_csv(LOG_FILE, nrows=0)
                existing_columns = list(existing_df.columns)
                
                # Если заголовок не совпадает, пересоздаем файл с правильным заголовком
                if existing_columns != expected_columns:
                    logger.warning(f"Несоответствие заголовков CSV. Ожидалось: {expected_columns}, "
                                 f"найдено: {existing_columns}. Пересоздаю файл с правильным заголовком.")
                    # Читаем все существующие данные без заголовка, чтобы правильно интерпретировать столбцы
                    try:
                        # Пытаемся прочитать с правильными именами столбцов
                        all_data = pd.read_csv(LOG_FILE, names=expected_columns, skiprows=1)
                        # Пересоздаем файл с правильным заголовком
                        all_data[expected_columns].to_csv(LOG_FILE, mode='w', header=True, index=False)
                    except Exception as read_error:
                        logger.error(f"Ошибка при чтении существующих данных: {read_error}. "
                                   f"Создаю новый файл с правильным заголовком.")
                        # Если не удалось прочитать, создаем новый файл
                        df.to_csv(LOG_FILE, mode='w', header=True, index=False)
                        return
                    # Добавляем новую запись
                    df.to_csv(LOG_FILE, mode='a', header=False, index=False)
                else:
                    # Заголовок правильный, просто добавляем запись
                    df.to_csv(LOG_FILE, mode='a', header=False, index=False)
            except Exception as e:
                logger.error(f"Ошибка при проверке заголовка CSV: {e}. Пытаюсь добавить запись напрямую.")
                df.to_csv(LOG_FILE, mode='a', header=False, index=False)
        else:
            # Файл не существует, создаем с правильным заголовком
            df.to_csv(LOG_FILE, mode='w', header=True, index=False)
        
        logger.info(f"Запрос пользователя {user_id} записан в лог: {ticker}")
    
    except Exception as e:
        logger.error(f"Ошибка при записи в лог: {e}")
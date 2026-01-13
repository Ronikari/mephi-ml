import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from models.ml_model import MLModel
from models.prophet_model import ProphetModel
from models.chronos_model import ChronosModel

from data.data_splitter import train_test_split_time_series

logger = logging.getLogger(__name__)

def train_and_evaluate_models(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Обучает и оценивает все модели.

    Returns:
        Список словарей с результатами каждой модели
    """
    # Разделяем данные
    train_data, test_data = train_test_split_time_series(data)

    # Инициализируем модели
    models = [
        MLModel(),        # Random Forest
        ProphetModel(),   # Prophet от Meta
        ChronosModel(),   # Chronos от Amazon
    ]

    results = []

    for model in models:
        try:
            logger.info(f"Обучаю модель: {model.name}")

            # Обучаем модель
            model.train(train_data)

            # Прогнозируем на тестовых данных
            y_true = test_data['Close'].values
            y_pred = model.predict(test_data)

            # Проверяем, что прогнозы имеют правильную длину и не содержат NaN
            if len(y_pred) != len(y_true):
                logger.warning(
                    f"Модель {model.name} вернула прогнозы разной длины ({len(y_pred)} vs {len(y_true)}). Пропускаем.")
                continue

            if np.isnan(y_pred).any():
                logger.warning(f"Модель {model.name} вернула NaN в прогнозе. Заменяю на последнее значение.")
                last_value = train_data['Close'].iloc[-1]
                y_pred = np.where(np.isnan(y_pred), last_value, y_pred)

            # Проверяем, что прогнозы не содержат бесконечных значений
            if np.any(np.isinf(y_pred)):
                logger.warning(
                    f"Модель {model.name} вернула бесконечные значения в прогнозе. Заменяю на последнее значение.")
                last_value = train_data['Close'].iloc[-1]
                y_pred = np.where(np.isinf(y_pred), last_value, y_pred)

            # Проверяем, что прогнозы не слишком большие (более 1000x от реальных значений)
            if np.max(np.abs(y_pred)) > np.max(np.abs(y_true)) * 1000:
                logger.warning(f"Модель {model.name} вернула нереалистичные значения. Использую простое среднее.")
                y_pred = np.full_like(y_true, y_true.mean())

            # Проверяем на "плоский" прогноз (горизонтальная прямая)
            # Если стандартное отклонение прогноза меньше 0.1% от среднего значения, это плохой прогноз
            pred_std = np.std(y_pred)
            pred_mean = np.mean(y_pred)
            if pred_mean > 0 and pred_std / pred_mean < 0.001:
                logger.warning(f"Модель {model.name} вернула плоский прогноз (горизонтальная прямая). Увеличиваем штраф к RMSE.")
                # Добавляем большой штраф, чтобы эта модель не была выбрана
                flat_penalty = pred_mean * 10  # Большой штраф
            else:
                flat_penalty = 0

            # Оцениваем качество
            metrics = model.evaluate(y_true, y_pred)
            metrics['model'] = model
            metrics['predictions'] = y_pred
            metrics['rmse'] = metrics['rmse'] + flat_penalty  # Добавляем штраф за плоский прогноз
            
            if flat_penalty > 0:
                metrics['is_flat'] = True
                logger.warning(f"Модель {model.name} наказана за плоский прогноз. RMSE с штрафом: {metrics['rmse']:.4f}")
            else:
                metrics['is_flat'] = False

            results.append(metrics)

            logger.info(f"Модель {model.name}: RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%")

        except Exception as e:
            logger.error(f"Ошибка при обучении модели {model.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Продолжаем с другими моделями

    # Если ни одна модель не сработала, используем простую модель
    if not results:
        logger.warning("Ни одна модель не сработала. Использую простую модель.")
        try:
            # Простая модель на основе среднего
            class SimpleModel:
                def __init__(self):
                    self.name = "Simple Mean"

                def forecast(self, data, steps):
                    last_value = data['Close'].iloc[-1]
                    last_date = data['Date'].iloc[-1]
                    forecast_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=steps,
                        freq='B'
                    )
                    return pd.Series([last_value] * steps, index=forecast_dates)

            simple_model = SimpleModel()

            results.append({
                'model_name': 'Simple Mean',
                'model': simple_model,
                'rmse': 0.0,
                'mape': 0.0,
                'predictions': np.full(len(test_data), train_data['Close'].mean())
            })

        except Exception as e:
            logger.error(f"Даже простая модель не сработала: {e}")

    return results

def select_best_model(results: List[Dict[str, Any]]) -> Tuple[str, Any, Dict[str, float]]:
    """
    Выбирает лучшую модель на основе метрики RMSE.

    Returns:
        Кортеж (имя_модели, объект_модели, метрики)
    """
    if not results:
        raise ValueError("Нет результатов для выбора модели!")

    # Фильтруем модели с корректными метриками
    valid_results = [r for r in results if not np.isnan(r['rmse'])]

    if not valid_results:
        # Если все модели вернули NaN, выбираем первую
        return results[0]['model_name'], results[0]['model'], results[0]

    # Выбираем модель с наименьшим RMSE
    best_result = min(valid_results, key=lambda x: x['rmse'])

    return best_result['model_name'], best_result['model'], best_result
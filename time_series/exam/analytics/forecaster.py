import pandas as pd
from typing import Any
import logging

logger = logging.getLogger(__name__)

def make_forecast(model: Any, data: pd.DataFrame, model_name: str, steps: int = 30) -> pd.Series:
    """
    Строит прогноз на заданное количество дней вперед.

    Args:
        model: Обученная модель
        data: Исторические данные
        model_name: Название модели (для логирования)
        steps: Количество дней для прогноза

    Returns:
        Прогноз на steps дней
    """
    try:
        # Строим прогноз
        forecast = model.forecast(data, steps)

        # Проверяем, что прогноз не содержит NaN
        if forecast.isna().any():
            logger.warning(f"Модель {model_name} вернула NaN в прогнозе. Заменяю на последнее значение.")
            last_price = data['Close'].iloc[-1]
            forecast = forecast.fillna(last_price)

        # Проверяем, что прогноз не пустой
        if len(forecast) == 0:
            logger.warning(f"Модель {model_name} вернула пустой прогноз. Создаю постоянный прогноз.")
            last_price = data['Close'].iloc[-1]
            last_date = data['Date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq='B'
            )
            forecast = pd.Series([last_price] * steps, index=forecast_dates)

        return forecast

    except Exception as e:
        logger.error(f"Ошибка при построении прогноза: {e}")
        # Создаем простой прогноз на основе последнего значения
        last_price = data['Close'].iloc[-1]
        last_date = data['Date'].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='B'
        )
        return pd.Series([last_price] * steps, index=forecast_dates)
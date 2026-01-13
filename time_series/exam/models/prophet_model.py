"""
Модуль с реализацией модели Facebook Prophet для прогнозирования временных рядов.

Prophet - это библиотека от Meta (Facebook) для прогнозирования временных рядов,
которая хорошо работает с данными, имеющими сезонность и тренды.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from models.base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet не установлен. Установите: pip install prophet")

class ProphetModel(BaseModel):
    """
    Модель Facebook Prophet для прогнозирования временных рядов.
    
    Prophet автоматически обнаруживает тренды, сезонность и праздники
    в данных, что делает её идеальной для финансовых временных рядов.
    """
    
    def __init__(self):
        super().__init__("Prophet")

    def prepare_data_for_prophet(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготавливает данные для Prophet модели.
        
        Prophet требует, чтобы данные имели колонки 'ds' (дата) и 'y' (значение).
        Также необходимо удалить информацию о часовом поясе.
        
        Args:
            data: DataFrame с колонками 'Date' и 'Close'
        
        Returns:
            DataFrame с колонками 'ds' и 'y' для Prophet
        """
        df_prophet = data[['Date', 'Close']].copy()
        df_prophet.columns = ['ds', 'y']
        
        # Удаляем часовой пояс, если он есть
        if hasattr(df_prophet['ds'].dt, 'tz') and df_prophet['ds'].dt.tz is not None:
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
        
        # Преобразуем в datetime без часового пояса
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
        
        return df_prophet

    def train(self, train_data: pd.DataFrame) -> None:
        """Обучает Prophet модель на исторических данных."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet не установлен.")

        try:
            # Подготавливаем данные для Prophet
            df_prophet = self.prepare_data_for_prophet(train_data)

            # Проверяем, что данные корректны
            if df_prophet['y'].isna().any():
                logger.warning("Обнаружены NaN в целевой переменной. Заполняю средним.")
                df_prophet['y'] = df_prophet['y'].fillna(df_prophet['y'].mean())

            # Создаем и обучаем модель
            self.model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                changepoint_range=0.8
            )

            # Добавляем дополнительные регрессоры если есть
            if 'Volume' in train_data.columns:
                df_prophet['volume'] = train_data['Volume'].values
                self.model.add_regressor('volume')

            self.model.fit(df_prophet)
            self.is_trained = True

            logger.info("Prophet модель успешно обучена")

        except Exception as e:
            logger.error(f"Ошибка при обучении Prophet: {e}")
            raise

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Прогнозирует на тестовых данных."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        try:
            # Создаем DataFrame с датами тестового периода
            future_dates = test_data[['Date']].copy()
            future_dates.columns = ['ds']

            # Удаляем часовой пояс
            future_dates['ds'] = pd.to_datetime(future_dates['ds']).dt.tz_localize(None)

            # Добавляем дополнительные регрессоры если есть
            if 'volume' in self.model.extra_regressors:
                future_dates['volume'] = test_data['Volume'].values

            # Делаем прогноз
            forecast = self.model.predict(future_dates)

            # Проверяем, что прогноз не содержит NaN
            if forecast['yhat'].isna().any():
                logger.warning("Prophet вернул NaN в прогнозе. Заменяю на последнее значение.")
                last_value = test_data['Close'].iloc[-1]
                forecast['yhat'] = forecast['yhat'].fillna(last_value)

            return forecast['yhat'].values

        except Exception as e:
            logger.error(f"Ошибка при прогнозе Prophet: {e}")
            # Возвращаем последнее известное значение в качестве прогноза
            return np.full(len(test_data), test_data['Close'].iloc[-1])

    def forecast(self, data: pd.DataFrame, steps: int) -> pd.Series:
        """Строит прогноз на steps шагов вперед."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        try:
            # Подготавливаем все данные для переобучения
            df_prophet = self.prepare_data_for_prophet(data)

            # Создаем новую модель на всех данных
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05
            )

            # Добавляем дополнительные регрессоры если есть
            if 'Volume' in data.columns:
                df_prophet['volume'] = data['Volume'].values
                model.add_regressor('volume')

            model.fit(df_prophet)

            # Создаем даты для прогноза
            last_date = data['Date'].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq='B'
            )

            future_df = pd.DataFrame({'ds': future_dates})

            # Добавляем дополнительные регрессоры для будущих дат (используем среднее значение)
            if 'volume' in model.extra_regressors:
                avg_volume = data['Volume'].mean()
                future_df['volume'] = avg_volume

            # Делаем прогноз
            forecast = model.predict(future_df)

            # Проверяем на NaN
            if forecast['yhat'].isna().any():
                logger.warning("Prophet вернул NaN в долгосрочном прогнозе. Использую линейную экстраполяцию.")
                last_value = data['Close'].iloc[-1]
                forecast['yhat'] = forecast['yhat'].fillna(last_value)

            # Создаем Series с правильным индексом
            forecast_series = pd.Series(forecast['yhat'].values, index=future_dates)

            return forecast_series

        except Exception as e:
            logger.error(f"Ошибка при построении долгосрочного прогноза Prophet: {e}")
            # Создаем простой прогноз на основе последнего значения
            last_value = data['Close'].iloc[-1]
            last_date = data['Date'].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq='B'
            )
            return pd.Series([last_value] * steps, index=future_dates)
import pandas as pd
import numpy as np
import logging
import torch
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class ChronosModel(BaseModel):
    """Модель на основе Chronos для прогнозирования временных рядов."""

    def __init__(self):
        super().__init__("Chronos")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context_length = 64  # Количество точек для контекста
        
    def train(self, train_data: pd.DataFrame) -> None:
        """
        Инициализирует предобученную модель Chronos.
        Chronos не требует обучения, используется zero-shot прогнозирование.
        """
        try:
            from chronos import ChronosPipeline
            
            # Загружаем предобученную модель (используем tiny версию для скорости)
            self.model = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-tiny",
                device_map=self.device,
                torch_dtype=torch.float32,
            )
            
            # Сохраняем исторические данные для контекста
            self.train_series = train_data['Close'].values
            self.is_trained = True
            
            logger.info(f"Модель Chronos загружена на устройство: {self.device}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели Chronos: {e}")
            raise

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Прогнозирует на тестовых данных."""
        if not self.is_trained:
            raise ValueError("Модель не загружена!")
        
        try:
            # Берем последние context_length точек из train_series
            context = torch.tensor(
                self.train_series[-self.context_length:],
                dtype=torch.float32
            ).unsqueeze(0)  # Добавляем batch dimension
            
            # Прогнозируем
            prediction_length = len(test_data)
            forecast = self.model.predict(
                context,
                prediction_length,
                num_samples=20,  # Количество сэмплов для усреднения
            )
            
            # Усредняем прогнозы по всем сэмплам
            predictions = np.median(forecast[0].numpy(), axis=0)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Ошибка при прогнозировании с Chronos: {e}")
            # Возвращаем простой прогноз на основе последнего значения
            last_value = self.train_series[-1]
            return np.full(len(test_data), last_value)

    def forecast(self, data: pd.DataFrame, steps: int) -> pd.Series:
        """Строит прогноз на steps шагов вперед."""
        if not self.is_trained:
            raise ValueError("Модель не загружена!")
        
        try:
            # Используем последние context_length точек
            series = data['Close'].values
            context = torch.tensor(
                series[-self.context_length:],
                dtype=torch.float32
            ).unsqueeze(0)  # Добавляем batch dimension
            
            # Делаем прогноз
            forecast = self.model.predict(
                context,
                steps,
                num_samples=20,
            )
            
            # Усредняем прогнозы
            predictions = np.median(forecast[0].numpy(), axis=0)
            
            # Создаем индекс дат для прогноза
            last_date = data['Date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq='B'
            )
            
            return pd.Series(predictions, index=forecast_dates)
            
        except Exception as e:
            logger.error(f"Ошибка при построении прогноза с Chronos: {e}")
            # Возвращаем простой прогноз
            last_value = data['Close'].iloc[-1]
            last_date = data['Date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq='B'
            )
            return pd.Series([last_value] * steps, index=forecast_dates)
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict
import numpy as np

class BaseModel(ABC):
    """Абстрактный базовый класс для моделей прогнозирования."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, train_data: pd.DataFrame) -> None:
        """Обучает модель на данных."""
        pass

    @abstractmethod
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Делает прогноз на тестовых данных."""
        pass

    @abstractmethod
    def forecast(self, data: pd.DataFrame, steps: int) -> pd.Series:
        """Строит прогноз на steps шагов вперед."""
        pass

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Вычисляет метрики качества прогноза."""
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        return {
            'rmse': rmse,
            'mape': mape,
            'model_name': self.name
        }
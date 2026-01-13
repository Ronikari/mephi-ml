import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel
from data.data_splitter import prepare_ml_data
from config import LAG_WINDOW

class MLModel(BaseModel):
    """Упрощенная ML модель на основе Random Forest."""

    def __init__(self):
        super().__init__("Random Forest")
        self.scaler = StandardScaler()
        self.window = LAG_WINDOW

    def train(self, train_data: pd.DataFrame) -> None:
        """Обучает модель на исторических данных."""
        # Используем только цену закрытия
        series = train_data['Close'].values

        # Подготавливаем данные в формате окон
        X, y = prepare_ml_data(series, self.window)

        # Масштабируем признаки
        X_scaled = self.scaler.fit_transform(X)

        # Обучаем модель
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Сохраняем последние значения для прогноза
        self.last_values = series[-self.window:].copy()

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Прогнозирует на тестовых данных."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        # Используем итеративный прогноз для тестовых данных
        predictions = []
        current_window = self.last_values.copy()

        for _ in range(len(test_data)):
            # Масштабируем текущее окно
            window_scaled = self.scaler.transform(current_window.reshape(1, -1))

            # Прогнозируем
            pred = self.model.predict(window_scaled)[0]
            predictions.append(pred)

            # Обновляем окно: удаляем самый старый, добавляем прогноз
            current_window = np.roll(current_window, -1)
            current_window[-1] = pred

        return np.array(predictions)

    def forecast(self, data: pd.DataFrame, steps: int) -> pd.Series:
        """Строит прогноз на steps шагов вперед."""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        # Получаем последние значения
        series = data['Close'].values
        current_window = series[-self.window:].copy()

        forecasts = []

        for _ in range(steps):
            # Масштабируем текущее окно
            window_scaled = self.scaler.transform(current_window.reshape(1, -1))

            # Прогнозируем следующий шаг
            next_pred = self.model.predict(window_scaled)[0]
            forecasts.append(next_pred)

            # Обновляем окно для следующей итерации
            current_window = np.roll(current_window, -1)
            current_window[-1] = next_pred

        # Создаем индекс дат для прогноза
        last_date = data['Date'].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='B'
        )

        return pd.Series(forecasts, index=forecast_dates)
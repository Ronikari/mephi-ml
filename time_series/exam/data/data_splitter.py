"""Модуль для разделения данных на обучающую и тестовую выборки."""

import pandas as pd
import numpy as np
from typing import Tuple
from config import TEST_SIZE

def train_test_split_time_series(
    data: pd.DataFrame,
    target_col: str = 'Close',
    test_size: float = TEST_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделяет временной ряд на обучающую и тестовую выборки.
    
    Важно: разделение происходит без перемешивания, так как порядок
    временных рядов критически важен.
    
    Args:
        data: DataFrame с временным рядом
        target_col: Название целевой колонки (по умолчанию 'Close')
        test_size: Доля тестовых данных (по умолчанию из config)
    
    Returns:
        Кортеж (train_data, test_data)
    """
    # Определяем индекс разделения
    split_idx = int(len(data) * (1 - test_size))
    
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    return train_data, test_data

def prepare_ml_data(data: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подготавливает данные для моделей машинного обучения.
    
    Создает скользящие окна фиксированного размера для обучения моделей.
    
    Args:
        data: Временной ряд (pd.Series или np.ndarray)
        window: Размер скользящего окна
    
    Returns:
        Кортеж (X, y) где:
        - X: массив признаков (окна)
        - y: массив целевых значений
    """
    X, y = [], []
    
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    
    return np.array(X), np.array(y)
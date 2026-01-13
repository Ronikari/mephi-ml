"""
Модуль для генерации торговых сигналов и расчета потенциальной прибыли.

Использует анализ локальных экстремумов для определения точек покупки и продажи.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from scipy.signal import argrelextrema

def find_local_extrema(series: pd.Series, order: int = 5) -> Dict[str, List]:
    """
    Находит локальные минимумы и максимумы в ряде.

    Args:
        series: Временной ряд
        order: Количество соседей для сравнения

    Returns:
        Словарь с датами и значениями экстремумов
    """
    # Находим индексы локальных минимумов и максимумов
    min_indices = argrelextrema(series.values, np.less_equal, order=order)[0]
    max_indices = argrelextrema(series.values, np.greater_equal, order=order)[0]

    # Фильтруем крайние точки
    if len(series) > 0:
        min_indices = min_indices[min_indices > 0]
        min_indices = min_indices[min_indices < len(series) - 1]

        max_indices = max_indices[max_indices > 0]
        max_indices = max_indices[max_indices < len(series) - 1]

    return {
        'minima': [(series.index[i], series.iloc[i]) for i in min_indices],
        'maxima': [(series.index[i], series.iloc[i]) for i in max_indices]
    }

def generate_trading_signals(forecast: pd.Series) -> List[Dict[str, Any]]:
    """
    Генерирует торговые сигналы на основе прогноза.

    Returns:
        Список сигналов с действием, датой и ценой
    """
    # Находим экстремумы
    extrema = find_local_extrema(forecast, order=3)

    signals = []

    # Добавляем сигналы на покупку (минимумы)
    for date, price in extrema['minima']:
        signals.append({
            'action': 'BUY',
            'date': date,
            'price': price
        })

    # Добавляем сигналы на продажу (максимумы)
    for date, price in extrema['maxima']:
        signals.append({
            'action': 'SELL',
            'date': date,
            'price': price
        })

    # Сортируем по дате
    signals.sort(key=lambda x: x['date'])

    return signals

def calculate_profit(
        initial_amount: float,
        forecast: pd.Series,
        signals: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Рассчитывает потенциальную прибыль от следования сигналам.

    Returns:
        Словарь с результатами расчета
    """
    if not signals:
        return {
            'final_amount': initial_amount,
            'profit_abs': 0.0,
            'profit_pct': 0.0
        }

    # Симулируем стратегию: покупаем на минимумах, продаем на максимумах
    cash = initial_amount
    shares = 0.0

    # Сортируем сигналы по дате
    sorted_signals = sorted(signals, key=lambda x: x['date'])

    for signal in sorted_signals:
        if signal['action'] == 'BUY' and cash > 0:
            # Покупаем максимально возможное количество акций
            shares_to_buy = cash / signal['price']
            shares += shares_to_buy
            cash = 0

        elif signal['action'] == 'SELL' and shares > 0:
            # Продаем все акции
            cash = shares * signal['price']
            shares = 0

    # Если в конце у нас остались акции, продаем их по последней цене
    if shares > 0:
        cash = shares * forecast.iloc[-1]

    final_amount = cash

    return {
        'final_amount': final_amount,
        'profit_abs': final_amount - initial_amount,
        'profit_pct': ((final_amount - initial_amount) / initial_amount * 100)
        if initial_amount > 0 else 0.0
    }
"""
Модуль для создания визуализаций прогнозов и торговых сигналов.

Использует matplotlib для создания графиков с историческими данными (свечные графики),
прогнозами и торговыми сигналами.
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib

matplotlib.use('Agg')  # Для работы без GUI

from config import TEMP_DIR

def plot_candlesticks(ax, data):
    """
    Рисует свечной график на указанной оси.
    
    Args:
        ax: Ось matplotlib для рисования
        data: DataFrame с колонками Date, Open, High, Low, Close
    """
    for idx, (i, row) in enumerate(data.iterrows()):
        date = row['Date']
        open_price = row['Open']
        close_price = row['Close']
        high_price = row['High']
        low_price = row['Low']
        
        # Определяем цвет свечи
        color = 'green' if close_price >= open_price else 'red'
        
        # Рисуем фитиль (высокая-низкая линия)
        ax.plot([date, date], [low_price, high_price], color='black', linewidth=0.8, zorder=1)
        
        # Рисуем тело свечи как линию с толщиной
        body_height = abs(close_price - open_price)
        if body_height < 0.01:  # Если цена почти не изменилась, рисуем горизонтальную линию
            ax.plot([date, date], [open_price - 0.1, open_price + 0.1], 
                   color='black', linewidth=2, zorder=2)
        else:
            # Рисуем толстую линию для тела свечи
            ax.plot([date, date], [open_price, close_price], 
                   color=color, linewidth=4, solid_capstyle='butt', zorder=2, alpha=0.9)

def create_forecast_plot(
        historical_data: pd.DataFrame,
        forecast: pd.Series,
        signals: list,
        ticker: str,
        forecast_days: int = 30
) -> Path:
    """
    Создает график с историческими данными (свечи) и прогнозом.

    Args:
        historical_data: Исторические данные
        forecast: Прогноз
        signals: Список торговых сигналов
        ticker: Тикер акции
        forecast_days: Количество дней прогнозирования

    Returns:
        Путь к сохраненному изображению
    """
    # Создаем график
    fig, ax = plt.subplots(figsize=(16, 9))

    # Рисуем свечной график для исторических данных
    plot_candlesticks(ax, historical_data)

    # Соединяем последнюю цену закрытия с первой точкой прогноза
    last_date = historical_data['Date'].iloc[-1]
    last_price = historical_data['Close'].iloc[-1]
    first_forecast_date = forecast.index[0]
    first_forecast_price = forecast.values[0]
    
    ax.plot(
        [last_date, first_forecast_date],
        [last_price, first_forecast_price],
        color='black',
        linewidth=2,
        linestyle='--',
        alpha=0.7
    )

    # Прогноз
    ax.plot(
        forecast.index,
        forecast.values,
        label=f'Прогноз на {forecast_days} дней',
        color='black',
        linewidth=2,
        linestyle='--',
        alpha=0.9
    )

    # Торговые сигналы
    buy_dates = [s['date'] for s in signals if s['action'] == 'BUY']
    buy_prices = [s['price'] for s in signals if s['action'] == 'BUY']

    sell_dates = [s['date'] for s in signals if s['action'] == 'SELL']
    sell_prices = [s['price'] for s in signals if s['action'] == 'SELL']

    if buy_dates:
        ax.scatter(
            buy_dates, buy_prices,
            color='green', s=80, marker='^',
            label='Сигнал ПОКУПКИ', zorder=5,
            edgecolors='white', linewidths=1.5
        )

    if sell_dates:
        ax.scatter(
            sell_dates, sell_prices,
            color='orange', s=80, marker='v',
            label='Сигнал ПРОДАЖИ', zorder=5,
            edgecolors='white', linewidths=1.5
        )

    # Настройки графика
    ax.set_title(f'Прогноз цен акций {ticker} на {forecast_days} дней', fontsize=16, fontweight='bold')
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Цена (USD)', fontsize=12)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Форматирование оси X
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Сохраняем график
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_{timestamp}.png"
    filepath = TEMP_DIR / filename

    # Сохраняем с хорошим качеством
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath
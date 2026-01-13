"""Модуль для загрузки исторических данных акций через Yahoo Finance."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from config import START_DATE, END_DATE

logger = logging.getLogger(__name__)

def load_stock_data(ticker: str) -> pd.DataFrame:
    """
    Загружает исторические данные акций за указанный период.
    
    Args:
        ticker: Тикер акции (например, 'AAPL', 'GOOGL')
    
    Returns:
        DataFrame с колонками ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    Raises:
        ValueError: Если данные для указанного тикера не найдены
    """
    try:
        logger.info(f"Загрузка данных для {ticker} с {START_DATE} по {END_DATE}")
        
        # Загружаем данные через yfinance
        stock = yf.Ticker(ticker)
        df = stock.history(start=START_DATE, end=END_DATE)
        
        if df.empty:
            # Пробуем загрузить за последний год
            logger.warning(
                f"Нет данных для {ticker} за указанный период. "
                f"Загружаю данные за последний год..."
            )
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"Данные для тикера {ticker} не найдены")
        
        # Преобразуем индекс в колонку
        df = df.reset_index()
        
        # Оставляем только необходимые колонки
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Удаляем строки с пропущенными значениями
        df = df.dropna(subset=['Close'])
        
        # Убираем часовой пояс из дат
        if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        # Преобразуем в datetime без часового пояса
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        logger.info(f"Загружено {len(df)} записей для {ticker}")
        
        return df
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных для {ticker}: {e}")
        raise
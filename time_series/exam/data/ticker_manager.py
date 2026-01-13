"""Модуль для управления списком тикеров акций."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class TickerManager:
    """
    Менеджер для работы со списком тикеров акций.
    
    Предоставляет функциональность для:
    - Загрузки списка тикеров из JSON файла
    - Поиска тикеров по букве или части названия
    - Получения полного списка доступных тикеров
    """
    
    def __init__(self, tickers_file: Optional[Path] = None):
        """
        Инициализация менеджера тикеров.
        
        Args:
            tickers_file: Путь к JSON файлу с тикерами.
                         Если не указан, используется tickers.json из текущей папки.
        """
        self.tickers_file = tickers_file or Path(__file__).parent / "tickers.json"
        self.tickers_data = self._load_tickers()
    
    def _load_tickers(self) -> Dict[str, List[str]]:
        """
        Загружает список тикеров из JSON файла.
        
        Returns:
            Словарь, где ключ - буква, значение - список тикеров на эту букву
        """
        try:
            with open(self.tickers_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Загружено {sum(len(v) for v in data.values())} тикеров")
            return data
        except FileNotFoundError:
            logger.error(f"Файл тикеров не найден: {self.tickers_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка при парсинге JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"Неизвестная ошибка при загрузке тикеров: {e}")
            return {}
    
    def get_tickers_by_letter(self, letter: str, limit: int = 10) -> List[str]:
        """
        Возвращает тикеры, начинающиеся с указанной буквы.
        
        Args:
            letter: Буква для поиска (будет приведена к верхнему регистру)
            limit: Максимальное количество возвращаемых тикеров
        
        Returns:
            Список тикеров на указанную букву
        """
        letter = letter.upper()
        if letter in self.tickers_data:
            return self.tickers_data[letter][:limit]
        return []
    
    def get_all_tickers(self) -> List[str]:
        """
        Возвращает все доступные тикеры.
        
        Returns:
            Отсортированный список всех тикеров
        """
        all_tickers = []
        for tickers in self.tickers_data.values():
            all_tickers.extend(tickers)
        return sorted(all_tickers)
    
    def search_tickers(self, query: str, limit: int = 10) -> List[str]:
        """
        Ищет тикеры по частичному совпадению.
        
        Args:
            query: Строка для поиска (будет приведена к верхнему регистру)
            limit: Максимальное количество результатов
        
        Returns:
            Отсортированный список найденных тикеров
        """
        query = query.upper()
        results = []
        
        for letter, tickers in self.tickers_data.items():
            for ticker in tickers:
                if query in ticker:
                    results.append(ticker)
        
        return sorted(results)[:limit]

# Создаем глобальный экземпляр менеджера
ticker_manager = TickerManager()
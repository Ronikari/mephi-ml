import logging
import traceback
import asyncio
from datetime import datetime
from typing import Dict, Any

from telegram import Update, ReplyKeyboardRemove
from telegram.ext import ContextTypes, ConversationHandler
from telegram.error import NetworkError, TimedOut

from core.states import TICKER, AMOUNT
from core.keyboards import (
    get_popular_tickers_keyboard, 
    get_ticker_categories_keyboard,
    get_main_menu_keyboard,
    get_tech_tickers_keyboard,
    get_finance_tickers_keyboard,
    get_health_tickers_keyboard,
    get_retail_tickers_keyboard,
    get_energy_tickers_keyboard,
    get_industry_tickers_keyboard
)
from data.stock_loader import load_stock_data
from data.ticker_manager import ticker_manager
from models.model_selector import train_and_evaluate_models, select_best_model
from analytics.forecaster import make_forecast
from analytics.visualizer import create_forecast_plot
from analytics.strategy import generate_trading_signals, calculate_profit
from utils.logger import log_request
from utils.formatters import format_currency, format_percentage

logger = logging.getLogger(__name__)

# Глобальный словарь для хранения данных пользователя между состояниями
user_sessions: Dict[int, Dict[str, Any]] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало диалога, запрос тикера."""
    user = update.message.from_user
    logger.info(f"Пользователь {user.first_name} начал диалог.")

    await update.message.reply_text(
        "📈 Привет! Я бот для анализа и прогнозирования акций.\n\n"
        "Я помогу вам:\n"
        "• Проанализировать исторические данные\n"
        "• Построить прогноз на выбранный период\n"
        "• Дать торговые рекомендации\n"
        "• Рассчитать потенциальную прибыль\n\n"
        "Выберите акцию из списка популярных или введите тикер вручную:",
        reply_markup=get_popular_tickers_keyboard()
    )

    return TICKER

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик нажатий на inline кнопки."""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    data = query.data
    
    # Обработка выбора тикера
    if data.startswith("ticker_"):
        ticker = data.replace("ticker_", "")
        
        if ticker == "custom":
            # Пользователь хочет ввести свой тикер
            await query.edit_message_text(
                "✍️ Введите тикер компании вручную (например, AAPL, GOOGL, TSLA):"
            )
            return TICKER
        
        # Пользователь выбрал тикер из кнопок
        user_sessions[user.id] = {'ticker': ticker}
        
        await query.edit_message_text(
            f"✅ Выбран тикер: {ticker}\n"
            f"⏳ Загружаю исторические данные..."
        )
        
        # Загружаем данные
        try:
            data_df = load_stock_data(ticker)
            if data_df.empty:
                await query.message.reply_text(
                    f"❌ Не удалось загрузить данные для тикера {ticker}.\n"
                    f"Попробуйте другой тикер.",
                    reply_markup=get_popular_tickers_keyboard()
                )
                return TICKER
            
            user_sessions[user.id]['data'] = data_df
            
            await query.message.reply_text(
                f"✅ Данные успешно загружены! Период: {len(data_df)} дней.\n\n"
                f"💰 Введите сумму для условной инвестиции в USD (например, 1000):",
                reply_markup=get_main_menu_keyboard()
            )
            
            return AMOUNT
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            await query.message.reply_text(
                f"❌ Произошла ошибка при загрузке данных: {str(e)[:100]}\n"
                f"Попробуйте другой тикер.",
                reply_markup=get_popular_tickers_keyboard()
            )
            return TICKER
    
    # Обработка навигации по категориям
    elif data == "back_to_popular":
        await query.edit_message_text(
            "📊 Выберите акцию из популярных или введите тикер вручную:",
            reply_markup=get_popular_tickers_keyboard()
        )
        return TICKER
    
    elif data == "back_to_categories":
        await query.edit_message_text(
            "📂 Выберите категорию акций:",
            reply_markup=get_ticker_categories_keyboard()
        )
        return TICKER
    
    # Обработка категорий
    elif data == "category_tech":
        await query.edit_message_text(
            "💻 Технологические компании:",
            reply_markup=get_tech_tickers_keyboard()
        )
        return TICKER
    
    elif data == "category_finance":
        await query.edit_message_text(
            "🏦 Финансовые компании:",
            reply_markup=get_finance_tickers_keyboard()
        )
        return TICKER
    
    elif data == "category_health":
        await query.edit_message_text(
            "🏥 Медицинские компании:",
            reply_markup=get_health_tickers_keyboard()
        )
        return TICKER
    
    elif data == "category_retail":
        await query.edit_message_text(
            "🛒 Ритейл компании:",
            reply_markup=get_retail_tickers_keyboard()
        )
        return TICKER
    
    elif data == "category_energy":
        await query.edit_message_text(
            "⚡ Энергетические компании:",
            reply_markup=get_energy_tickers_keyboard()
        )
        return TICKER
    
    elif data == "category_industry":
        await query.edit_message_text(
            "🏭 Промышленные компании:",
            reply_markup=get_industry_tickers_keyboard()
        )
        return TICKER
    
    return TICKER

async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик текстовых сообщений с кнопок главного меню."""
    text = update.message.text
    user = update.message.from_user
    
    if text == "📊 Новый анализ":
        # Очищаем сессию пользователя
        if user.id in user_sessions:
            del user_sessions[user.id]
        return await start(update, context)
    
    elif text == "📋 Помощь":
        return await help_command(update, context)
    
    elif text == "🔍 Поиск тикеров":
        await update.message.reply_text(
            "🔍 Для поиска тикеров используйте команду:\n"
            "/get_tickers [буква или часть тикера]\n\n"
            "Примеры:\n"
            "• /get_tickers A - все тикеры на букву A\n"
            "• /get_tickers AAPL - поиск по части названия",
            reply_markup=get_main_menu_keyboard()
        )
        return ConversationHandler.END
    
    elif text == "❌ Отмена":
        return await cancel(update, context)
    
    # Если это не кнопка меню, передаем дальше
    return None

async def get_tickers_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /get_tickers."""
    if not context.args:
        await update.message.reply_text(
            "Пожалуйста, укажите букву для поиска тикеров.\n"
            "Например: /get_tickers A\n"
            "Или: /get_tickers AAPL (для поиска по части тикера)"
        )
        return

    query = context.args[0].upper()

    # Если запрос - одна буква
    if len(query) == 1 and query.isalpha():
        tickers = ticker_manager.get_tickers_by_letter(query)
        if tickers:
            tickers_list = "\n".join([f"• {ticker}" for ticker in tickers])
            await update.message.reply_text(
                f"📊 Тикеры на букву '{query}':\n\n{tickers_list}\n\n"
                f"Всего найдено: {len(tickers)} тикеров\n"
                f"Для анализа напишите название тикера"
            )
        else:
            await update.message.reply_text(
                f"Не найдено тикеров на букву '{query}'.\n"
                f"Попробуйте другую букву."
            )
    else:
        # Если запрос - часть тикера
        tickers = ticker_manager.search_tickers(query)
        if tickers:
            tickers_list = "\n".join([f"• {ticker}" for ticker in tickers])
            await update.message.reply_text(
                f"🔍 Результаты поиска для '{query}':\n\n{tickers_list}\n\n"
                f"Всего найдено: {len(tickers)} тикеров\n"
                f"Для анализа напишите название тикера"
            )
        else:
            await update.message.reply_text(
                f"Не найдено тикеров по запросу '{query}'.\n"
                f"Попробуйте другой запрос."
            )

async def process_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка введенного тикера, запрос суммы инвестиции."""
    ticker = update.message.text.upper().strip()
    user = update.message.from_user

    # Сохраняем тикер в сессии пользователя
    user_sessions[user.id] = {'ticker': ticker}

    await update.message.reply_text(
        f"✅ Тикер: {ticker}\n"
        f"⏳ Загружаю исторические данные...",
        reply_markup=ReplyKeyboardRemove()
    )

    # Загружаем данные
    try:
        data = load_stock_data(ticker)
        if data.empty:
            await update.message.reply_text(
                f"❌ Не удалось загрузить данные для тикера {ticker}.\n"
                f"Проверьте правильность тикера и попробуйте снова.",
                reply_markup=get_popular_tickers_keyboard()
            )
            return TICKER

        user_sessions[user.id]['data'] = data

        await update.message.reply_text(
            f"✅ Данные успешно загружены! Период: {len(data)} дней.\n\n"
            f"💰 Введите сумму для условной инвестиции в USD (например, 1000):",
            reply_markup=get_main_menu_keyboard()
        )

        return AMOUNT

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        await update.message.reply_text(
            f"❌ Произошла ошибка при загрузке данных: {str(e)[:100]}\n"
            f"Попробуйте другой тикер.",
            reply_markup=get_popular_tickers_keyboard()
        )
        return TICKER

async def process_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка суммы инвестиции, запуск анализа с фиксированным периодом 30 дней."""
    user = update.message.from_user

    try:
        amount = float(update.message.text.replace(',', '.').replace(' ', ''))
        if amount <= 0:
            raise ValueError("Сумма должна быть положительной")
    except ValueError:
        await update.message.reply_text(
            "⚠️ Пожалуйста, введите корректную сумму (число больше 0).",
            reply_markup=get_main_menu_keyboard()
        )
        return AMOUNT

    # Сохраняем сумму и фиксированный период 30 дней в сессии
    user_sessions[user.id]['amount'] = amount
    user_sessions[user.id]['forecast_days'] = 30

    await update.message.reply_text(
        f"✅ Сумма инвестиции: ${amount:,.2f}\n"
        f"📅 Период прогноза: 30 дней\n"
        f"⏳ Начинаю анализ..."
    )

    # Запускаем анализ
    return await run_analysis(update.message, context, user)

async def run_analysis(message, context: ContextTypes.DEFAULT_TYPE, user) -> int:
    """Запуск анализа акций и построение прогноза."""
    forecast_days = user_sessions[user.id]['forecast_days']
    
    # Отправляем новое сообщение о начале анализа
    processing_msg = await message.reply_text(
        f"🔍 Начинаю анализ...\n"
        f"• Тикер: {user_sessions[user.id]['ticker']}\n"
        f"• Сумма: ${user_sessions[user.id]['amount']:,.2f}\n"
        f"• Период прогноза: {forecast_days} дней\n\n"
        f"⏳ Это может занять несколько минут...",
        reply_markup=ReplyKeyboardRemove()
    )

    try:
        # Получаем данные из сессии
        ticker = user_sessions[user.id]['ticker']
        data = user_sessions[user.id]['data']
        amount = user_sessions[user.id]['amount']

        # Шаг 1: Обучаем и сравниваем модели
        try:
            await processing_msg.edit_text("🔍 Анализ в процессе...\n1️⃣ 📊 Обучаю модели...")
        except Exception:
            # Если не удалось отредактировать, отправляем новое сообщение
            await processing_msg.delete()
            processing_msg = await message.reply_text("🔍 Анализ в процессе...\n1️⃣ 📊 Обучаю модели...")
        models_results = train_and_evaluate_models(data)

        # Шаг 2: Выбираем лучшую модель
        try:
            await processing_msg.edit_text("🔍 Анализ в процессе...\n2️⃣ ⚖️ Сравниваю метрики...")
        except Exception:
            pass
        best_model_name, best_model, metrics = select_best_model(models_results)

        # Шаг 3: Делаем прогноз на указанный период
        try:
            await processing_msg.edit_text(f"🔍 Анализ в процессе...\n3️⃣ 🔮 Строю прогноз на {forecast_days} дней...")
        except Exception:
            pass
        forecast = make_forecast(best_model, data, model_name=best_model_name, steps=forecast_days)

        # Шаг 4: Генерируем торговые сигналы
        try:
            await processing_msg.edit_text("🔍 Анализ в процессе...\n4️⃣ 📈 Анализирую торговые сигналы...")
        except Exception:
            pass
        signals = generate_trading_signals(forecast)

        # Шаг 5: Рассчитываем прибыль
        profit = calculate_profit(amount, forecast, signals)

        # Шаг 6: Создаем визуализацию
        try:
            await processing_msg.edit_text("🔍 Анализ в процессе...\n5️⃣ 🎨 Создаю график...")
        except Exception:
            pass
        plot_path = create_forecast_plot(data, forecast, signals, ticker, forecast_days)

        # Шаг 7: Формируем финальный отчет
        last_price = data['Close'].iloc[-1]
        forecast_price = forecast.iloc[-1]
        price_change = (forecast_price - last_price) / last_price * 100

        # Формируем списки дат покупок и продаж
        buy_signals_list = [s for s in signals if s['action'] == 'BUY']
        sell_signals_list = [s for s in signals if s['action'] == 'SELL']
        
        # Форматируем даты для вывода
        buy_dates_str = ""
        if buy_signals_list:
            buy_dates_str = "\n".join([
                f"  • {s['date'].strftime('%d.%m.%Y')} - {format_currency(s['price'])}"
                for s in buy_signals_list
            ])
        else:
            buy_dates_str = "  • Нет сигналов на покупку"
        
        sell_dates_str = ""
        if sell_signals_list:
            sell_dates_str = "\n".join([
                f"  • {s['date'].strftime('%d.%m.%Y')} - {format_currency(s['price'])}"
                for s in sell_signals_list
            ])
        else:
            sell_dates_str = "  • Нет сигналов на продажу"

        report = f"""
📊 **ОТЧЕТ ПО АКЦИЯМ {ticker}**

📈 **Прогноз на {forecast_days} дней:**
• Текущая цена: {format_currency(last_price)}
• Прогноз через {forecast_days} дней: {format_currency(forecast_price)}
• Изменение: {format_percentage(price_change)}

🏆 **Лучшая модель:** {best_model_name}
• Метрика RMSE: {metrics['rmse']:.4f}
• Метрика MAPE: {metrics['mape']:.2f}%

🎯 **Торговые рекомендации:**

🟢 **ПОКУПКА** (всего: {len(buy_signals_list)}):
{buy_dates_str}

🔴 **ПРОДАЖА** (всего: {len(sell_signals_list)}):
{sell_dates_str}

💰 **Симуляция стратегии:**
• Начальный капитал: {format_currency(amount)}
• Конечный капитал: {format_currency(profit['final_amount'])}
• Прибыль: {format_currency(profit['profit_abs'])} ({format_percentage(profit['profit_pct'])})
"""

        # Отправляем график с обработкой ошибок
        photo_sent = False
        max_retries = 3
        max_file_size = 10 * 1024 * 1024  # 10 MB - лимит Telegram
        
        if plot_path.exists():
            file_size = plot_path.stat().st_size
            if file_size > 0 and file_size <= max_file_size:
                for attempt in range(max_retries):
                    try:
                        with open(plot_path, 'rb') as photo:
                            await message.reply_photo(
                                photo=photo,
                                caption=report,
                                parse_mode='Markdown',
                                reply_markup=get_main_menu_keyboard()
                            )
                        photo_sent = True
                        break
                    except (NetworkError, TimedOut) as e:
                        logger.warning(f"Ошибка сети при отправке фото (попытка {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                        else:
                            logger.error(f"Не удалось отправить фото после {max_retries} попыток")
                    except Exception as e:
                        logger.error(f"Неожиданная ошибка при отправке фото: {e}")
                        break
            elif file_size > max_file_size:
                logger.warning(f"Файл слишком большой ({file_size / 1024 / 1024:.2f} MB), отправляю только текст")
        
        # Если фото не удалось отправить, отправляем только текст
        if not photo_sent:
            await message.reply_text(
                report,
                parse_mode='Markdown',
                reply_markup=get_main_menu_keyboard()
            )
            if plot_path.exists():
                logger.info(f"График сохранен по пути: {plot_path}, но не отправлен")
        
        # Очищаем временный файл после отправки
        try:
            if plot_path.exists():
                plot_path.unlink()
                logger.debug(f"Временный файл {plot_path} удален")
        except Exception as e:
            logger.warning(f"Не удалось удалить временный файл {plot_path}: {e}")

        # Логируем запрос
        log_request(
            user_id=user.id,
            timestamp=datetime.now(),
            ticker=ticker,
            amount=amount,
            forecast_days=forecast_days,
            best_model=best_model_name,
            metric_value=metrics['rmse'],
            profit=profit['profit_pct']
        )

        try:
            await processing_msg.delete()
        except Exception:
            pass  # Если сообщение уже удалено или недоступно

        # Предлагаем новый анализ
        await message.reply_text(
            "✅ Анализ завершен!\n\n"
            "Используйте кнопки меню для продолжения работы 👇",
            reply_markup=get_main_menu_keyboard()
        )

        # Очищаем сессию пользователя
        if user.id in user_sessions:
            del user_sessions[user.id]

        return ConversationHandler.END

    except Exception as e:
        logger.error(f"Ошибка при анализе: {e}. Traceback: {traceback.format_exc()}")
        try:
            await processing_msg.delete()
        except Exception:
            pass  # Если сообщение уже удалено
        
        await message.reply_text(
            f"❌ Произошла ошибка при анализе: {str(e)[:200]}\n"
            f"Попробуйте снова с другим тикером.",
            reply_markup=get_main_menu_keyboard()
        )
        
        # Очищаем сессию при ошибке
        if user.id in user_sessions:
            del user_sessions[user.id]
        
        return ConversationHandler.END

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправка справки по команде /help."""
    await update.message.reply_text(
        "📈 **Бот для анализа акций**\n\n"
        "🔹 **Доступные команды:**\n"
        "• /start - начать новый анализ\n"
        "• /help - показать справку\n"
        "• /get_tickers [буква] - поиск тикеров\n"
        "• /cancel - отменить текущий анализ\n\n"
        "🔹 **Как использовать:**\n"
        "1. Выберите акцию из списка или введите тикер\n"
        "2. Укажите сумму для инвестиции\n"
        "3. Выберите период прогноза\n"
        "4. Получите детальный анализ и рекомендации\n\n"
        "🔹 **Примеры популярных тикеров:**\n"
        "• 🍎 AAPL - Apple\n"
        "• 💻 MSFT - Microsoft\n"
        "• 🔍 GOOGL - Google\n"
        "• 🚗 TSLA - Tesla\n"
        "• 📦 AMZN - Amazon\n\n"
        "💡 Используйте кнопки меню для быстрого доступа!",
        parse_mode='Markdown',
        reply_markup=get_main_menu_keyboard()
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена диалога."""
    user = update.message.from_user
    logger.info(f"Пользователь {user.first_name} отменил диалог.")

    # Очищаем сессию пользователя
    if user.id in user_sessions:
        del user_sessions[user.id]

    await update.message.reply_text(
        "❌ Анализ отменен.\n\n"
        "Чтобы начать заново, используйте кнопку '📊 Новый анализ' или команду /start",
        reply_markup=get_main_menu_keyboard()
    )

    return ConversationHandler.END
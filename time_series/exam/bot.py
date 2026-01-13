import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackQueryHandler
from core import bot_handlers
from core.bot_handlers import help_command, cancel, get_tickers_command, button_callback
from config import BOT_TOKEN

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Запуск бота."""
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN не найден в переменных окружения!")

    # Создаем Application
    application = Application.builder().token(BOT_TOKEN).build()

    # Создаем ConversationHandler для диалога
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', bot_handlers.start),
            MessageHandler(filters.Regex('^📊 Новый анализ$'), bot_handlers.start)
        ],
        states={
            bot_handlers.TICKER: [
                CallbackQueryHandler(button_callback),
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot_handlers.process_ticker)
            ],
            bot_handlers.AMOUNT: [
                MessageHandler(filters.Regex('^(📊 Новый анализ|🔍 Поиск тикеров)$'), 
                              bot_handlers.text_message_handler),
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot_handlers.process_amount)
            ],
        },
        fallbacks=[
            CommandHandler('cancel', bot_handlers.cancel)
        ],
    )

    # Регистрируем обработчики
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("get_tickers", get_tickers_command))
    application.add_handler(MessageHandler(filters.Regex('^🔍 Поиск тикеров$'), 
                                          lambda update, context: update.message.reply_text(
                                              "🔍 Используйте команду /get_tickers [буква]")))

    # Запускаем бота
    print("Бот запущен и готов к работе!")
    application.run_polling(allowed_updates=["message", "callback_query"])

if __name__ == '__main__':
    main()
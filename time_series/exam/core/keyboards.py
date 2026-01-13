from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup

def get_popular_tickers_keyboard() -> InlineKeyboardMarkup:
    """Создает inline клавиатуру с популярными тикерами."""
    keyboard = [
        [
            InlineKeyboardButton("🍎 Apple (AAPL)", callback_data="ticker_AAPL"),
            InlineKeyboardButton("🚗 Tesla (TSLA)", callback_data="ticker_TSLA"),
        ],
        [
            InlineKeyboardButton("💻 Microsoft (MSFT)", callback_data="ticker_MSFT"),
            InlineKeyboardButton("🔍 Google (GOOGL)", callback_data="ticker_GOOGL"),
        ],
        [
            InlineKeyboardButton("📦 Amazon (AMZN)", callback_data="ticker_AMZN"),
            InlineKeyboardButton("🔴 AMD (AMD)", callback_data="ticker_AMD"),
        ],
        [
            InlineKeyboardButton("📱 Meta (META)", callback_data="ticker_META"),
            InlineKeyboardButton("💳 Visa (V)", callback_data="ticker_V"),
        ],
        [
            InlineKeyboardButton("☕ Starbucks (SBUX)", callback_data="ticker_SBUX"),
            InlineKeyboardButton("🎮 Netflix (NFLX)", callback_data="ticker_NFLX"),
        ],
        [
            InlineKeyboardButton("✍️ Ввести свой тикер", callback_data="ticker_custom"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_ticker_categories_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру с категориями тикеров."""
    keyboard = [
        [
            InlineKeyboardButton("💻 Технологии", callback_data="category_tech"),
            InlineKeyboardButton("🏦 Финансы", callback_data="category_finance"),
        ],
        [
            InlineKeyboardButton("🏥 Здравоохранение", callback_data="category_health"),
            InlineKeyboardButton("🛒 Ритейл", callback_data="category_retail"),
        ],
        [
            InlineKeyboardButton("⚡ Энергетика", callback_data="category_energy"),
            InlineKeyboardButton("🏭 Промышленность", callback_data="category_industry"),
        ],
        [
            InlineKeyboardButton("« Назад к популярным", callback_data="back_to_popular"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_tech_tickers_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура с технологическими компаниями."""
    keyboard = [
        [
            InlineKeyboardButton("🍎 Apple (AAPL)", callback_data="ticker_AAPL"),
            InlineKeyboardButton("💻 Microsoft (MSFT)", callback_data="ticker_MSFT"),
        ],
        [
            InlineKeyboardButton("🔍 Google (GOOGL)", callback_data="ticker_GOOGL"),
            InlineKeyboardButton("📱 Meta (META)", callback_data="ticker_META"),
        ],
        [
            InlineKeyboardButton("🔴 AMD (AMD)", callback_data="ticker_AMD"),
            InlineKeyboardButton("🎮 Netflix (NFLX)", callback_data="ticker_NFLX"),
        ],
        [
            InlineKeyboardButton("🐦 Twitter/X (X)", callback_data="ticker_X"),
            InlineKeyboardButton("💿 Intel (INTC)", callback_data="ticker_INTC"),
        ],
        [
            InlineKeyboardButton("« Назад к категориям", callback_data="back_to_categories"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_finance_tickers_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура с финансовыми компаниями."""
    keyboard = [
        [
            InlineKeyboardButton("🏦 JPMorgan (JPM)", callback_data="ticker_JPM"),
            InlineKeyboardButton("💰 Bank of America (BAC)", callback_data="ticker_BAC"),
        ],
        [
            InlineKeyboardButton("💳 Visa (V)", callback_data="ticker_V"),
            InlineKeyboardButton("💳 Mastercard (MA)", callback_data="ticker_MA"),
        ],
        [
            InlineKeyboardButton("🏦 Goldman Sachs (GS)", callback_data="ticker_GS"),
            InlineKeyboardButton("💵 PayPal (PYPL)", callback_data="ticker_PYPL"),
        ],
        [
            InlineKeyboardButton("« Назад к категориям", callback_data="back_to_categories"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_health_tickers_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура с медицинскими компаниями."""
    keyboard = [
        [
            InlineKeyboardButton("💊 Pfizer (PFE)", callback_data="ticker_PFE"),
            InlineKeyboardButton("💉 Johnson & Johnson (JNJ)", callback_data="ticker_JNJ"),
        ],
        [
            InlineKeyboardButton("🧬 Moderna (MRNA)", callback_data="ticker_MRNA"),
            InlineKeyboardButton("💊 Merck (MRK)", callback_data="ticker_MRK"),
        ],
        [
            InlineKeyboardButton("« Назад к категориям", callback_data="back_to_categories"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_retail_tickers_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура с ритейл компаниями."""
    keyboard = [
        [
            InlineKeyboardButton("📦 Amazon (AMZN)", callback_data="ticker_AMZN"),
            InlineKeyboardButton("🛒 Walmart (WMT)", callback_data="ticker_WMT"),
        ],
        [
            InlineKeyboardButton("☕ Starbucks (SBUX)", callback_data="ticker_SBUX"),
            InlineKeyboardButton("🍔 McDonald's (MCD)", callback_data="ticker_MCD"),
        ],
        [
            InlineKeyboardButton("🏪 Costco (COST)", callback_data="ticker_COST"),
            InlineKeyboardButton("🏬 Target (TGT)", callback_data="ticker_TGT"),
        ],
        [
            InlineKeyboardButton("« Назад к категориям", callback_data="back_to_categories"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_energy_tickers_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура с энергетическими компаниями."""
    keyboard = [
        [
            InlineKeyboardButton("⛽ ExxonMobil (XOM)", callback_data="ticker_XOM"),
            InlineKeyboardButton("⛽ Chevron (CVX)", callback_data="ticker_CVX"),
        ],
        [
            InlineKeyboardButton("⚡ NextEra Energy (NEE)", callback_data="ticker_NEE"),
            InlineKeyboardButton("🔋 Tesla Energy (TSLA)", callback_data="ticker_TSLA"),
        ],
        [
            InlineKeyboardButton("« Назад к категориям", callback_data="back_to_categories"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_industry_tickers_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура с промышленными компаниями."""
    keyboard = [
        [
            InlineKeyboardButton("✈️ Boeing (BA)", callback_data="ticker_BA"),
            InlineKeyboardButton("🚗 General Motors (GM)", callback_data="ticker_GM"),
        ],
        [
            InlineKeyboardButton("🚛 Caterpillar (CAT)", callback_data="ticker_CAT"),
            InlineKeyboardButton("⚙️ 3M (MMM)", callback_data="ticker_MMM"),
        ],
        [
            InlineKeyboardButton("« Назад к категориям", callback_data="back_to_categories"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_main_menu_keyboard() -> ReplyKeyboardMarkup:
    """Создает основное меню с базовыми кнопками."""
    keyboard = [
        ["📊 Новый анализ"],
        ["🔍 Поиск тикеров"]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
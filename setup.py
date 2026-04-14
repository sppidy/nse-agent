from setuptools import setup, find_packages

setup(
    name="ai_trading_agent",
    version="0.1.0",
    py_modules=[
        "ai_strategy", "autopilot", "backtester", "chat", "config",
        "data_fetcher", "learner", "logger", "main", "market_calendar",
        "news_sentiment", "paper_trader", "persistence", "predictor", "strategy"
    ],
    install_requires=[
        "pandas",
        "yfinance",
        "ta",
        "google-genai",
        "python-dotenv",
        "pydantic",
        "tabulate",
        "scikit-learn",
        "xgboost",
        "sqlalchemy",
    ]
)

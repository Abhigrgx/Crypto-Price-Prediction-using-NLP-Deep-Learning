"""
Application configuration loaded from environment / .env file.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "development"
    secret_key: str = "change_me_in_production"
    allowed_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "crypto_db"
    postgres_user: str = "crypto_user"
    postgres_password: str = "changeme"

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017/crypto_sentiment"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # APIs
    binance_api_key: str = ""
    binance_api_secret: str = ""
    coingecko_api_key: str = ""
    news_api_key: str = ""
    cryptopanic_api_key: str = ""
    twitter_bearer_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "CryptoPredictBot/1.0"

    # ML
    model_dir: str = "ml/saved_models"
    huggingface_model: str = "ProsusAI/finbert"
    sequence_length: int = 60
    prediction_horizon: int = 1

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = Settings()

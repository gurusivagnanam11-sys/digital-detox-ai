from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):

    APP_NAME: str = "Digital Detox AI"
    APP_VERSION: str = "1.0"

    ALLOWED_ORIGINS: List[str] = ["*"]

    DATABASE_URL: str = "sqlite:///./digital_detox.db"

    # Enable SQLAlchemy echo for debug logging (useful during development)
    DEBUG: bool = False

    LATE_NIGHT_START_HOUR: int = 23
    LATE_NIGHT_END_HOUR: int = 6
    
    # Embedding model for sentence-transformers
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Directory where ChromaDB persists vectors (relative to repo root)
    CHROMA_PERSIST_DIR: str = "./data/chroma"

    # OpenAI / LLM settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    LLM_MAX_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7


def get_settings():
    return Settings()
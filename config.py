"""
app/config.py

Central configuration — now includes feedback + memory settings.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ── Llama 3.2 (Critic) ──
    LLAMA_API_KEY: str
    LLAMA_BASE_URL: str = "https://api.together.xyz/v1"
    LLAMA_MODEL: str = "meta-llama/Llama-3.2-3B-Instruct"
    LLAMA_MAX_TOKENS: int = 2048
    LLAMA_TEMPERATURE: float = 0.0
    LLAMA_TIMEOUT: int = 30

    # ── Claude (Judge) ──
    CLAUDE_API_KEY: str
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
    CLAUDE_MAX_TOKENS: int = 2048
    CLAUDE_TEMPERATURE: float = 0.0
    CLAUDE_TIMEOUT: int = 60

    # ── Retry / Resilience ──
    MAX_RETRIES: int = 3
    RETRY_BASE_DELAY: float = 1.0

    # ── Feedback Storage ──
    FEEDBACK_DB_PATH: str = "data/feedback.db"

    # ── Vector Memory ──
    CHROMA_PERSIST_DIR: str = "data/chroma"
    CHROMA_COLLECTION: str = "forensic_cases"
    SIMILAR_CASES_K: int = 5
    SIMILAR_CASES_MIN_SCORE: float = 0.3

    # ── Learning ──
    USE_DYNAMIC_FEWSHOTS: bool = True
    MIN_LABELED_CASES_FOR_LEARNING: int = 3

    # ── Logging ──
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

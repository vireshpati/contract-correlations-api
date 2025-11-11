"""Configuration management using pydantic-settings."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    pghost: str
    pgport: int = 5432
    pgdatabase: str
    pguser: str
    pgpassword: str

    # Hugging Face
    hf_token: str

    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_cache_dir: str = "./model_cache"
    use_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7

    # RAG
    rag_top_k: int = 5
    similarity_threshold: float = 0.6

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.pguser}:{self.pgpassword}"
            f"@{self.pghost}:{self.pgport}/{self.pgdatabase}?sslmode=require"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()

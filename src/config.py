"""Configuration."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Database
    pghost: str
    pgport: int = 5432
    pgdatabase: str
    pguser: str
    pgpassword: str

    # HuggingFace
    hf_token: str

    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_cache_dir: Optional[str] = None  # Set to custom path if disk space limited

    # Index building
    skip_index_build: bool = False  # Set to True to skip auto-building index on startup
    index_path: str = "./faiss_index"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.pguser}:{self.pgpassword}@{self.pghost}:{self.pgport}/{self.pgdatabase}?sslmode=require"


def get_settings() -> Settings:
    return Settings()

"""Configuration."""
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.pguser}:{self.pgpassword}@{self.pghost}:{self.pgport}/{self.pgdatabase}?sslmode=require"


def get_settings() -> Settings:
    return Settings()

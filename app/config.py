"""
Configuration management using pydantic-settings.

WHY pydantic-settings?
- Automatically reads from environment variables and .env files
- Provides type validation (if NEO4J_URI is missing, app fails FAST at startup, not at runtime)
- Single source of truth for all configuration

INTERVIEW TIP: This follows the "Fail Fast" principle.
If a required config is missing, the app crashes immediately on startup
rather than failing at 3 AM when a user hits the Neo4j endpoint.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")

    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key for LLM calls")

    # App Configuration
    app_env: str = Field(default="development", description="Application environment")
    app_debug: bool = Field(default=True, description="Enable debug mode")

    model_config = {
        "env_file": ".env",          # Read from .env file
        "env_file_encoding": "utf-8",
        "case_sensitive": False,      # NEO4J_URI and neo4j_uri both work
    }


# Singleton pattern — create ONE instance, reuse everywhere
# This avoids reading .env file multiple times
settings = Settings()

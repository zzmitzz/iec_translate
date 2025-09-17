
import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Server Configuration
    app_name: str = Field(default="Realtime Translate API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    app_description: str = Field(
        default="A real-time translation service with WebSocket support",
        env="APP_DESCRIPTION"
    )
    api_key: str = Field(default="1234567890", env="API_KEY")
    
    # Server Host and Port
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=3456, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    
    # CORS Configuration
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(default=["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")
    cors_credentials: bool = Field(default=True, env="CORS_CREDENTIALS")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="app.log", env="LOG_FILE")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Database Configuration (for future use)
    # database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # # Translation Service Configuration
    # default_source_language: str = Field(default="auto", env="DEFAULT_SOURCE_LANGUAGE")
    # default_target_language: str = Field(default="en", env="DEFAULT_TARGET_LANGUAGE")
    
    # Whisper Configuration
    whisper_model: str = Field(default="base", env="WHISPER_MODEL")
    whisper_device: str = Field(default="cpu", env="WHISPER_DEVICE")
    
    # WebSocket Configuration
    websocket_timeout: int = Field(default=300, env="WEBSOCKET_TIMEOUT")
    max_connections: int = Field(default=100, env="MAX_CONNECTIONS")
    
    # Audio Processing Configuration
    # audio_sample_rate: int = Field(default=16000, env="AUDIO_SAMPLE_RATE")
    # audio_chunk_duration: float = Field(default=1.0, env="AUDIO_CHUNK_DURATION")
    
    # Security Configuration
    secret_key: Optional[str] = Field(default=None, env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Development/Production Mode
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment variables"""
    global settings
    settings = Settings()
    return settings 
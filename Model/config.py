from pydantic_settings import BaseSettings
from pydantic import validator

class Settings(BaseSettings):
    RETELL_API_KEY: str = "key_30218ea0c0e650dba03254e0d8e1"
    OPENAI_API_KEY: str 
    DEEPSEEK_API_KEY: str = "sk-edbd132cdd4c48378d384cdb023269ff"
    DEEPGRAM_API_KEY: str ="493763d17bce44f96521b6691f59ddf2a2ceb707"
    ELEVENLABS_API_KEY: str = "sk_dd8365f5cf1c46822124d3abf54fd2ea628df9f399cdb074"
    TALKSCRIBER_API_KEY: str = "LJrCo9ZD5UxxoHUPE-QK-4GEfp0G62RpNtbLLmF5_X0"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

    @validator("RETELL_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY", "TALKSCRIBER_API_KEY")
    def validate_api_keys(cls, v):
        if not v or len(v) < 10:  # Basic validation
            raise ValueError("Invalid API key")
        return v

settings = Settings()
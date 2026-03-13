from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:123456@localhost:5432/aimedicalDb"
    SECRET_KEY: str = "a3f8k2m9x7p4q1r6w5n0j8h3d2b5c7e9"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

settings = Settings()
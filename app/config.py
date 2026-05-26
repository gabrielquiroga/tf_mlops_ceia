from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Se leen/configuran las variables de entorno, y el resto las lee de esta clase.
    """
    mlflow_tracking_uri: str
    model_name: str
    model_stage: str
    data_dir: str = "./data"
    log_level: str = "INFO"
    data_source_url: str

    class Config:
        env_file = ".env"

settings = Settings()
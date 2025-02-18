from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mlflow_uri: str = "https://platform.clapeyron.digital/mlflow/"
    mlflow_username: str = "deployer"
    mlflow_password: str = "U0KjTPPDJE"
    mlflow_experiment_id: int = 16
    mlflow_run_prefix: str = "opensource"
    git_base_url: str = "https://github.com/dkuryakin/mot"


settings = Settings()

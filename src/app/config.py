from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # default hf repo id for the model if not specified in .env
    huggingface_repo_id: str = "JayRay5/convnext-tiny-224-cyprus-fish-cls"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()

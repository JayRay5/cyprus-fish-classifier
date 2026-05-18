from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # default hf repo id for the model if not specified in .env
    huggingface_repo_id: str = "JayRay5/convnext-tiny-224-cyprus-fish-cls"
    github_repo_id: str = "JayRay5/reconnaissance_poisson_chypre.git"
    image_samples_path: str = "./src/app/assets/samples"
    allowed_cors_origins: str = "http://localhost:7860"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()

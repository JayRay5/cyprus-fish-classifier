from pydantic_settings import BaseSettings


class UISettings(BaseSettings):
    # L'URL de l'API Azure (Infectée via les Secrets HF, défaut sur local)
    api_url: str = "http://127.0.0.1:8000"

    api_secret_token: str = ""

    api_key_name: str = "X-API-Key"

    github_repo_url: str = "https://github.com/JayRay5/cyprus-fish-classifier/tree/main"
    huggingface_repo_url: str = (
        "https://huggingface.co/JayRay5/convnext-tiny-224-cyprus-fish-cls"
    )
    ui_sample_path: str = "./assets/samples"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = UISettings()

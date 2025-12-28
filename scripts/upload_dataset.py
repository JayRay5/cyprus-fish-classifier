# Script to upload the splited dataset to the hub
import os
import hydra
from datasets import load_dataset
from omegaconf import DictConfig


def push_dataset(data_cfg: DictConfig, hf_token: str):
    print(f"Load data from {data_cfg.processed_path}...")

    dataset = load_dataset("imagefolder", data_dir=str(data_cfg.processed_path))  # nosec B615

    print(f"Send to Hugging Face Hub Repo : {data_cfg.hf_repo_id}...")

    dataset.push_to_hub(
        data_cfg.hf_repo_id, token=hf_token, private=data_cfg.hf_private
    )

    print(
        f"Success: Dataset sent to https://huggingface.co/datasets/{data_cfg.hf_repo_id}"
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("Error: HF_Token not found! Please add it to your conda env.")

    push_dataset(cfg.data, hf_token)


if __name__ == "__main__":
    main()

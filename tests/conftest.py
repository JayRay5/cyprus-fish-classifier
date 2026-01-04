import os
import torch
import pytest
from PIL import Image
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="session")
def hf_token():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        pytest.fail(
            "Error, the huggingface token has not been find in the HF_TOKEN env variable!"
        )
    return hf_token


@pytest.fixture
def mock_config(tmp_path):
    cfg = OmegaConf.create(
        {
            "prepare_data": {
                "raw_path": str(tmp_path / "raw"),
                "processed_path": str(tmp_path / "processed"),
                "split_ratio": [0.8, 0.2],
                "seed": 42,
                "valid_extensions": [".jpg", ".jpeg", ".png"],
            },
            "data": {
                "repo_id": "fake/repo",
                "repo_revision": "hffake01",
                "num_classes": 3,
                "class_names": ["fish_A", "fish_B", "fish_C"],
            },
            "model": {
                "hf_repo_id": "fake/vit-model",
                "revision": "hffake02",
                "training_output_dir": "convnext-cyprus-fish-cls",
                "target_hf_repo_id": "fake2/convnext",
            },
            "train": {
                "k_folds": 2,
                "batch_size": 1,
                "grad_acc": 1,
                "epochs": 1,
                "lr": 3e-4,
                "warmup_steps": 0,
                "weight_decay": 0.01,
                "scheduler": "constant",
                "num_workers": 2,
                "seed": 42,
                "device": "cpu",
                "fp16": False,
                "freeze_backbone": True,
                "push_to_hub": False,
            },
        }
    )
    return cfg


@pytest.fixture(scope="session")
def mock_hf_item():
    """Simule un dictionnaire renvoyé par Hugging Face Dataset"""
    return {"image": Image.new("RGB", (100, 100), color="red"), "label": 1}


@pytest.fixture(scope="session")
def mock_external_deps(mock_hf_item):
    with (
        patch("src.cyprus_fish.data.load_dataset") as mock_load_dataset,
        patch(
            "src.cyprus_fish.data.AutoImageProcessor.from_pretrained"
        ) as mock_processor_cls,
    ):
        # Dataset
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 10
        mock_dataset_instance.__getitem__.return_value = mock_hf_item
        mock_load_dataset.return_value = mock_dataset_instance

        # Processor
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224)
        }
        mock_processor_cls.return_value = mock_processor_instance

        yield {"load_dataset": mock_load_dataset, "processor": mock_processor_instance}

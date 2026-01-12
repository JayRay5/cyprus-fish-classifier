import os
import torch
import pytest
from PIL import Image
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch
from transformers.image_processing_base import BatchFeature


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
                "name": "fake_model",
                "hf_repo_id": "fake/vit-model",
                "revision": "hffake02",
                "training_output_dir": "convnext-cyprus-fish-cls",
                "target_hf_repo_id": "fake2/convnext",
            },
            "train": {
                "k_folds": 2,
                "batch_size": 2,
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
                "experiment_name": "fake_name",
            },
            "space_id": "fake_space_id",
        }
    )
    return cfg


@pytest.fixture(scope="session")
def mock_hf_item():
    """Mimick a dict return by an hf dataset"""
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
        mock_processor_instance.return_value = BatchFeature(
            {"pixel_values": torch.randn(1, 3, 224, 224)}
        )
        mock_processor_cls.return_value = mock_processor_instance

        # Model
        model = MagicMock()
        model.device = "cpu"
        model.dtype = torch.float32

        model.config.id2label = {0: "Fish_A", 1: "Fish_B", 2: "Fish_C"}

        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.0, 0.0, 0.5]])

        model.return_value = mock_outputs

        yield {
            "load_dataset": mock_load_dataset,
            "processor": mock_processor_instance,
            "model": model,
        }

import torch
from src.cyprus_fish.data import CyprusFishDataset


def test_dataset_initialization(mock_config, mock_external_deps):
    """
    Test CyprusFishDataset
    """

    ds = CyprusFishDataset(
        repo_id=mock_config.data.repo_id,
        repo_revision=mock_config.data.repo_revision,
        model_name=mock_config.model.hf_repo_id,
        model_revision=mock_config.model.revision,
        split="train",
        num_classes=mock_config.data.num_classes,
    )

    assert len(ds) == 10

    # Check __getitem__
    pixel_values = ds[0]["pixel_values"]
    labels = ds[0]["labels"]

    # Check shape
    print(pixel_values.shape)
    assert pixel_values.shape == (3, 224, 224)
    assert labels.shape == (3,)
    # Check one hot encoding
    assert torch.equal(labels, torch.tensor([0.0, 1.0, 0.0]))

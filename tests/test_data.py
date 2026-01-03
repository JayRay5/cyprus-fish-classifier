import torch
from torch.utils.data import DataLoader
from src.cyprus_fish.data import CyprusFishDataset


def test_dataset_initialization(mock_config, mock_external_deps):
    """
    Test CyprusFishDataset
    """
    ds = CyprusFishDataset(
        repo_id=mock_config.data.repo_id,
        repo_revision=mock_config.data.repo_revision,
        model_name=mock_config.processor.model_name,
        processor_revision=mock_config.processor.revision,
        split="train",
        num_classes=mock_config.data.num_classes,
    )

    assert len(ds) == 10

    # Check __getitem__
    pixel_values, label_one_hot = ds[0]

    # Check shape
    assert pixel_values.shape == (3, 224, 224)
    assert label_one_hot.shape == (3,)
    # Check one hot encoding
    assert torch.equal(label_one_hot, torch.tensor([0.0, 1.0, 0.0]))


def test_dataloader_batching(mock_config, mock_external_deps):
    """Integration Test with the DataLoader"""

    ds = CyprusFishDataset(
        repo_id=mock_config.data.repo_id,
        repo_revision=mock_config.data.repo_revision,
        model_name=mock_config.processor.model_name,
        processor_revision=mock_config.processor.revision,
        split="train",
        num_classes=mock_config.data.num_classes,
    )

    dl = DataLoader(
        ds,
        batch_size=mock_config.train.batch_size,
        shuffle=False,
    )

    batch_images, batch_labels = next(iter(dl))

    # Check batch shape
    assert batch_images.shape == (2, 3, 224, 224)
    assert batch_labels.shape == (2, 3)

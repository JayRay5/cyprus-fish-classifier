from pathlib import Path
from scripts.prepare_dataset import split_data


def test_split_data_ratio(mock_config):
    """
    Check split preparation
    """

    # Create temporary fake folders
    raw_dir = Path(mock_config.prepare_data.raw_path)
    processed_dir = Path(mock_config.prepare_data.processed_path)

    # Create fake images
    class_dir = raw_dir / "fish_a"
    class_dir.mkdir(parents=True)

    for i in range(10):
        (class_dir / f"img_{i}.jpg").touch()  # Create an empty file

    split_data(mock_config.prepare_data)

    # 3. ASSERTIONS

    # Check images in split dir
    assert (processed_dir / "train" / "fish_a").exists()
    assert (processed_dir / "test" / "fish_a").exists()

    # Check image numbers
    train_files = list((processed_dir / "train" / "fish_a").glob("*.jpg"))
    test_files = list((processed_dir / "test" / "fish_a").glob("*.jpg"))

    # Check ratio
    assert len(train_files) == 8
    assert len(test_files) == 2

    # Check data leak in train
    train_names = {f.name for f in train_files}
    test_names = {f.name for f in test_files}

    assert train_names.isdisjoint(test_names), (
        "Warning ! Test data detected in train split!"
    )

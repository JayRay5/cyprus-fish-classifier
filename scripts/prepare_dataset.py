import hydra
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig


def setup_directories(classes, processed_path):
    # If the directory already exists, delete it
    if processed_path.exists():
        shutil.rmtree(processed_path)

    for split in ["train", "test"]:
        for class_name in classes:
            (processed_path / split / class_name).mkdir(parents=True, exist_ok=True)


def split_data(cfg: DictConfig):
    raw_path = Path(cfg.raw_path)
    processed_path = Path(cfg.processed_path)
    split_ratio = cfg.split_ratio
    valid_extensions = cfg.valid_extensions
    seed = cfg.seed

    random.seed(seed)

    if not raw_path.exists():
        raise FileNotFoundError(f"The folder {raw_path} has not been found!")

    classes = [d.name for d in raw_path.iterdir() if d.is_dir()]

    setup_directories(classes, processed_path)
    for class_name in classes:
        images = [
            f
            for f in (raw_path / class_name).glob("*.*")
            if f.suffix.lower() in valid_extensions
        ]
        random.shuffle(images)

        # Generate image idx for each split
        split_idx = int(len(images) * split_ratio[0])
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        # Copy to train
        for img in tqdm(train_imgs, desc=f"Train {class_name}", leave=False):
            shutil.copy2(img, processed_path / "train" / class_name / img.name)

        # Copy to test
        for img in tqdm(test_imgs, desc=f"Test  {class_name}", leave=False):
            shutil.copy2(img, processed_path / "test" / class_name / img.name)

    print(f"\n Processing over ! Data saved in {processed_path}")
    print(f"   - Train : {split_ratio[0] * 100}%")
    print(f"   - Test  : {split_ratio[1] * 100}%")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    split_data(cfg.prepare_data)


if __name__ == "__main__":
    main()

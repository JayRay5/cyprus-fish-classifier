import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoImageProcessor


class CyprusFishDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        repo_revision: str,
        model_name: str,
        model_revision,
        split: str,
        num_classes: int = 5,
    ):
        self.num_classes = num_classes
        self.hf_dataset = load_dataset(repo_id, split=split, revision=repo_revision)  # nosec B615

        self.processor = AutoImageProcessor.from_pretrained(
            model_name, revision=model_revision
        )  # nosec B615

        if split == "train":
            self.augment = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ]
            )
        else:
            self.augment = None

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        label_idx = item["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.augment:
            image = self.augment(image)

        inputs = self.processor(images=image, return_tensors="pt")

        pixel_values = inputs["pixel_values"].squeeze(0)

        label_one_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        label_one_hot[label_idx] = 1.0

        return pixel_values, label_one_hot

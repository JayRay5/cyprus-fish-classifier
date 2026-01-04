import hydra
import torch
import numpy as np
import os
import json
from datetime import datetime
from huggingface_hub import create_repo, upload_folder
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

from src.cyprus_fish.data import CyprusFishDataset
from src.cyprus_fish.utils import (
    compute_metrics,
    CalculateTrainMetricsCallback,
    extract_clean_history,
)


def model_loading(cfg: DictConfig):
    class_names = list(cfg.data.class_names)
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}

    model = AutoModelForImageClassification.from_pretrained(
        cfg.model.hf_repo_id,
        revision=cfg.model.revision,
        num_labels=cfg.data.num_classes,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
    ).to(cfg.train.device)  # nosec B615

    if cfg.train.get("freeze_backbone", True):
        for name, param in model.named_parameters():
            if "classifier" not in name and "head" not in name:
                param.requires_grad = False
        print("Backbone frozen.")

    return model


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train_k_fold(cfg: DictConfig):
    save_path = f"experiments/{cfg.model.training_output_dir}/k_fold_validation"
    if os.path.exists(save_path):
        n_dir = len(os.listdir(save_path))
        save_path = f"{save_path}/m_{n_dir}"
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = f"{save_path}/m_0"
        os.makedirs(save_path)

    dataset = CyprusFishDataset(
        repo_id=cfg.data.hf_repo_id,
        repo_revision=cfg.data.revision,
        model_name=cfg.model.hf_repo_id,
        model_revision=cfg.model.revision,
        split="train",
        num_classes=cfg.data.num_classes,
    )

    try:
        labels = dataset.labels
    except AttributeError:
        labels = [dataset[i]["labels"] for i in range(len(dataset))]

    y_array = np.array(
        [
            label.cpu().numpy() if isinstance(label, torch.Tensor) else label
            for label in labels
        ]
    )
    if y_array.ndim > 1:
        labels_indices = np.argmax(y_array, axis=1)
    else:
        labels_indices = y_array

    k_folds = cfg.train.get("k_folds", 5)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=cfg.train.seed)

    global_results = {}

    dummy_X = np.zeros(len(labels_indices))

    for fold, (train_idx, val_idx) in enumerate(skf.split(dummy_X, labels_indices)):
        print(f"\n FOLD {fold + 1}/{k_folds}")

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        model = model_loading(cfg)

        training_args = TrainingArguments(
            output_dir=f"{save_path}/fold_{fold}",
            remove_unused_columns=False,
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=cfg.train.lr,
            lr_scheduler_type=cfg.train.scheduler,
            per_device_train_batch_size=cfg.train.batch_size,
            gradient_accumulation_steps=cfg.train.grad_acc,
            per_device_eval_batch_size=cfg.train.batch_size,
            num_train_epochs=cfg.train.epochs,
            warmup_steps=cfg.train.warmup_steps,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=cfg.train.fp16,
            push_to_hub=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.add_callback(CalculateTrainMetricsCallback(trainer))

        trainer.train()

        history = trainer.state.log_history
        history = extract_clean_history(history)
        global_results[f"fold_{fold + 1}"] = {"history": history}
    print(global_results)

    accuracies = [res["history"][-1]["val_accuracy"] for res in global_results.values()]
    if None in accuracies:
        global_results["global_val"] = {"mean": None, "std": None}
    else:
        global_results["global_val"] = {
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
        }

    output_file = f"{save_path}/kfold_results.json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(global_results, f, indent=4)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    model_path = f"experiments/{cfg.model.training_output_dir}"
    save_path = f"{model_path}/full_training"
    if os.path.exists(save_path):
        n_dir = len(os.listdir(save_path))
        save_path = f"{save_path}/m_{n_dir}"
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = f"{save_path}/m_0"
        os.makedirs(save_path)

    train_dataset = CyprusFishDataset(
        repo_id=cfg.data.hf_repo_id,
        repo_revision=cfg.data.revision,
        model_name=cfg.model.hf_repo_id,
        model_revision=cfg.model.revision,
        split="train",
        num_classes=cfg.data.num_classes,
    )

    test_dataset = CyprusFishDataset(
        repo_id=cfg.data.hf_repo_id,
        repo_revision=cfg.data.revision,
        model_name=cfg.model.hf_repo_id,
        model_revision=cfg.model.revision,
        split="test",
        num_classes=cfg.data.num_classes,
    )

    model = model_loading(cfg)

    training_args = TrainingArguments(
        output_dir=save_path,
        remove_unused_columns=False,
        learning_rate=cfg.train.lr,
        lr_scheduler_type=cfg.train.scheduler,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_acc,
        per_device_eval_batch_size=cfg.train.batch_size,
        num_train_epochs=cfg.train.epochs,
        warmup_steps=cfg.train.warmup_steps,
        fp16=cfg.train.fp16,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    save_path = f"{save_path}/end_training_weights"
    trainer.save_model(save_path)

    metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    test_results = {"test_metrics": metrics}

    with open(f"{save_path}/results.json", "w") as f:
        json.dump(test_results, f, indent=4)

    tracker_file = f"{model_path}/best_model_tracker.json"
    if os.path.exists(tracker_file):
        with open(tracker_file, "r") as f:
            data = json.load(f)
            best_recorded_acc = data.get("test_accuracy", 0.0)
    else:
        best_recorded_acc = 0.0

    current_acc = metrics["test_accuracy"]

    if current_acc > best_recorded_acc:
        new_record = {
            "test_accuracy": current_acc,
            "local_model_path": save_path,
            "date": datetime.now().isoformat(),
        }
        with open(tracker_file, "w") as f:
            json.dump(new_record, f, indent=4)

        if hasattr(train_dataset, "processor"):
            train_dataset.processor.save_pretrained(save_path)
        try:
            create_repo(
                repo_id=cfg.model.target_hf_repo_id, exist_ok=True, private=False
            )

        except Exception as e:
            print(f"Warning: {e}")

        upload_folder(
            folder_path=save_path,
            repo_id=cfg.model.target_hf_repo_id,
            commit_message=f"Update weights, new test acc: {current_acc:.2%}",
            ignore_patterns=["README.md", ".git*"],
        )

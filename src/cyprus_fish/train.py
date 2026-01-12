import hydra
import torch
import numpy as np
import os
import json
import mlflow
from omegaconf import OmegaConf
from datetime import datetime
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from huggingface_hub import HfApi, create_repo, upload_folder

from src.cyprus_fish.data import CyprusFishDataset
from src.cyprus_fish.utils import (
    compute_metrics,
    CalculateTrainMetricsCallback,
    extract_clean_history,
    get_best_run_by_parameter,
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
    mlflow.set_experiment(cfg.train.experiment_name)
    remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
    if remote_server_uri:
        mlflow.set_tracking_uri(remote_server_uri)
    backbone_status = "Frozen" if cfg.train.freeze_backbone else "Unfrozen"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    run_name = (
        f"{cfg.model.name}_"
        f"KFold-cross-validation_"
        f"Ep{cfg.train.epochs}_"
        f"BS{cfg.train.batch_size}_"
        f"LR{cfg.train.lr}_"
        f"SCH{cfg.train.scheduler}_"
        f"WD{cfg.train.weight_decay}_"
        f"{backbone_status}_"
        f"Dt{timestamp}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_param("model_name", cfg.model.name)
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

        # Prepare the k-folds datasets for the cross validation
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
        skf = StratifiedKFold(
            n_splits=k_folds, shuffle=True, random_state=cfg.train.seed
        )

        global_results = {}

        dummy_X = np.zeros(len(labels_indices))

        # Cross-validation loop for each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(dummy_X, labels_indices)):
            print(f"\n FOLD {fold + 1}/{k_folds}")
            child_run_name = f"Fold_{fold + 1}"

            with mlflow.start_run(run_name=child_run_name, nested=True):
                mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
                train_dataset = Subset(dataset, train_idx)
                val_dataset = Subset(dataset, val_idx)

                model = model_loading(cfg)

                training_args = TrainingArguments(
                    output_dir=f"{save_path}/fold_{fold}",
                    remove_unused_columns=False,
                    logging_strategy="epoch",
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    report_to=["mlflow"],
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

        accuracies = [
            res["history"][-1]["val_accuracy"] for res in global_results.values()
        ]

        if None in accuracies:
            global_results["global_val"] = {"mean": None, "std": None}
        else:
            global_results["global_val"] = {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
            }

        mlflow.log_metric("mean_cv_accuracy", global_results["global_val"]["mean"])
        mlflow.log_metric("std_cv_accuracy", global_results["global_val"]["std"])

    output_file = f"{save_path}/kfold_results.json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(global_results, f, indent=4)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    hf_token = os.getenv("HF_TOKEN")

    if hf_token is None:
        raise Warning("Warning : HF_Token not found in the env.")

    mlflow.set_experiment(cfg.train.experiment_name)
    remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
    if remote_server_uri:
        mlflow.set_tracking_uri(remote_server_uri)
    backbone_status = "Frozen" if cfg.train.freeze_backbone else "Unfrozen"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = (
        f"{cfg.model.name}_"
        f"Global_train_"
        f"Ep{cfg.train.epochs}_"
        f"BS{cfg.train.batch_size}_"
        f"LR{cfg.train.lr}_"
        f"SCH{cfg.train.scheduler}_"
        f"WD{cfg.train.weight_decay}_"
        f"{backbone_status}_"
        f"Dt{timestamp}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_param("model_name", cfg.model.name)
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
            logging_strategy="epoch",
            eval_strategy="no",
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
            report_to=["mlflow"],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.add_callback(CalculateTrainMetricsCallback(trainer))

        trainer.train()

        metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

        # Save model and metrics locally
        save_path = f"{save_path}/end_training_weights"
        trainer.save_model(save_path)

        if hasattr(train_dataset, "processor"):
            train_dataset.processor.save_pretrained(save_path)

        test_results = {"test_metrics": metrics}

        with open(f"{save_path}/results.json", "w") as f:
            json.dump(test_results, f, indent=4)

        # Save metrics on ML-Flow
        mlflow.log_metrics(metrics)

        # Look for the best current model
        # Local experiment
        tracker_file = f"{model_path}/best_model_tracker.json"
        if os.path.exists(tracker_file):
            with open(tracker_file, "r") as f:
                data = json.load(f)
                best_recorded_acc = data.get("test_accuracy", 0.0)
        else:
            best_recorded_acc = 0.0

        # Dagshub/mlflow experiments
        _, best_acc_mlflow, _ = get_best_run_by_parameter(
            cfg.train.experiment_name,
            "metrics.test_accuracy",
            target_model_name=cfg.model.name,
        )
        if best_acc_mlflow is None:
            best_acc_mlflow = 0.0
        print(best_acc_mlflow)

        current_acc = metrics["test_accuracy"]

        # Push the model to the HuggingFace Hub / Dagshub and restart the Space
        if current_acc > best_recorded_acc and current_acc > best_acc_mlflow:
            new_record = {
                "test_accuracy": current_acc,
                "local_model_path": save_path,
                "date": datetime.now().isoformat(),
            }
            with open(tracker_file, "w") as f:
                json.dump(new_record, f, indent=4)
            mlflow.log_artifacts(save_path, artifact_path="model")

            if cfg.train.push_to_hub:
                if hf_token is not None:
                    try:
                        create_repo(
                            repo_id=cfg.model.target_hf_repo_id,
                            exist_ok=True,
                            private=False,
                            token=hf_token,
                        )

                    except Exception as e:
                        print(f"Warning: {e}")

                    upload_folder(
                        folder_path=save_path,
                        repo_id=cfg.model.target_hf_repo_id,
                        commit_message=f"Update weights, new test acc: {current_acc:.2%}",
                        ignore_patterns=["README.md", ".git*"],
                        token=hf_token,
                    )

                    # Restart the hf space
                    api = HfApi()
                    space_id = cfg.space_id

                    is_space_existing = api.repo_exists(
                        repo_id=space_id, repo_type="space", token=hf_token
                    )

                    if is_space_existing:
                        try:
                            api.restart_space(repo_id=space_id, token=hf_token)
                            print("HF space restared with the new model")
                        except Exception as e:
                            print(f"Error when restarting the space : {e}")

                    else:
                        print(f"Error: Space '{space_id}' not found.")

                else:
                    raise Warning(
                        "Cannot push the model to the hub because no hf_token found in the env."
                    )

import numpy as np
import evaluate
import mlflow
from mlflow.tracking import MlflowClient
from transformers import TrainerCallback

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """
    Evaluation function use by Hugging Face Trainer
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    labels = np.argmax(labels, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


class CalculateTrainMetricsCallback(TrainerCallback):
    """
    Callback to compute accuracy at each epoch
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer.evaluate(
            eval_dataset=self.trainer.train_dataset, metric_key_prefix="train"
        )
        if args.eval_strategy.value != "no":
            self.trainer.evaluate(
                eval_dataset=self.trainer.eval_dataset, metric_key_prefix="val"
            )


def extract_clean_history(log_history):
    """
    Function use to clean the log of the Hugging face Trainer after the training.
    """

    data_by_epoch = {}

    for entry in log_history:
        if "epoch" not in entry:
            continue

        epoch = round(entry["epoch"])

        if epoch not in data_by_epoch:
            data_by_epoch[epoch] = {
                "epoch": epoch,
                "train_loss": None,
                "train_accuracy": None,
                "val_loss": None,
                "val_accuracy": None,
            }

        if "train_accuracy" in entry:
            data_by_epoch[epoch]["train_accuracy"] = entry["train_accuracy"]
            if "train_loss" in entry:
                data_by_epoch[epoch]["train_loss"] = entry["train_loss"]

        elif "loss" in entry:
            data_by_epoch[epoch]["train_loss"] = entry["loss"]

        if "val_accuracy" in entry:
            data_by_epoch[epoch]["val_accuracy"] = entry["val_accuracy"]
            data_by_epoch[epoch]["val_loss"] = entry["val_loss"]

    return sorted(data_by_epoch.values(), key=lambda x: x["epoch"])


def get_best_run_by_parameter(
    experiment_name, metric_name, target_model_name=None, mode="max"
):
    """
    Look for the best results of the model among registered experiments in ML-Flow

    Args:
        experiment_name (str): experiment name DagsHub/MLflow
        metric_name (str): metric use to evaluate the actual best model
        target_model_name (str): name of the model as registered in the yaml config
        mode (str): "max" (Accuracy) ou "min" (Loss)
    """
    print("[INFO] start pulling ml flow runs")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    print(experiment)
    if experiment is None:
        print(f"Experiment {experiment_name} not found.")
        return None, None, None

    # Filter only among finished run
    filter_parts = ["status = 'FINISHED'"]

    if target_model_name:
        filter_parts.append(f"params.model_name = '{target_model_name}'")

    filter_string = " AND ".join(filter_parts)

    # Search for runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string=filter_string
    )

    if runs.empty:
        print(f"Run not found for the model '{target_model_name}'.")
        return None, None, None

    # Filter
    if metric_name not in runs.columns:
        print(f"Metric '{metric_name}' not found")
        return None, None, None

    runs = runs.dropna(subset=[metric_name])
    ascending = True if mode == "min" else False
    best_run = runs.sort_values(by=metric_name, ascending=ascending).iloc[0]

    return (
        best_run["run_id"],
        best_run[metric_name],
        best_run.get("tags.mlflow.runName", "No Name"),
    )

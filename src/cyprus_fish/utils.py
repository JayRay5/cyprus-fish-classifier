import numpy as np
import evaluate
from transformers import TrainerCallback

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
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
        self.trainer.evaluate(
            eval_dataset=self.trainer.eval_dataset, metric_key_prefix="val"
        )


def extract_clean_history(log_history):
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

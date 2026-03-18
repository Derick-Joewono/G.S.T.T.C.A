"""
Logger Utility

Mencatat hasil training ke file CSV agar mudah dianalisis.
"""

import os
import csv
from datetime import datetime


class Logger:

    def __init__(self, config):

        self.model_name = config.model_name

        self.log_dir = os.path.join("logs", self.model_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, "log.csv")

        self.start_time = datetime.now()

        self._init_log_file()

    # =====================================================
    # INIT CSV
    # =====================================================

    def _init_log_file(self):

        if not os.path.exists(self.log_path):

            with open(self.log_path, mode="w", newline="") as f:

                writer = csv.writer(f)

                header = [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "iou",
                    "f1",
                    "precision",
                    "recall",
                    "accuracy",
                    "kappa",
                    "time_elapsed"
                ]

                writer.writerow(header)

    # =====================================================
    # LOG PER EPOCH
    # =====================================================

    def log(self, epoch, train_loss, val_loss, metrics):

        time_elapsed = (datetime.now() - self.start_time).total_seconds()

        row = [
            epoch,
            train_loss,
            val_loss,
            metrics.get("iou", 0),
            metrics.get("f1", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("accuracy", 0),
            metrics.get("kappa", 0),
            time_elapsed
        ]

        with open(self.log_path, mode="a", newline="") as f:

            writer = csv.writer(f)
            writer.writerow(row)

    # =====================================================
    # PRINT LOG KE CONSOLE
    # =====================================================

    def print(self, epoch, train_loss, val_loss, metrics):

        print(f"\nEpoch {epoch}")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")

        for k, v in metrics.items():
            print(f"{k.upper():<10}: {v:.4f}")

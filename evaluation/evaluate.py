import torch
from torch.utils.data import DataLoader

from datasets.dataLoader import ChangeDetectionDataset
from evaluation.metrics import (
    compute_confusion_matrix,
    evaluate_all
)
from configs.train_config import TrainConfig

from utils.forward_model import forward_model


def evaluate_model(
    model,
    loader,
    model_path,
    num_classes,
    model_type,
    batch_size=TrainConfig.batch_size,
    device =TrainConfig.device
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    conf_matrix = torch.zeros((num_classes, num_classes)).to(device)

    with torch.no_grad():

        for t1, t2, mask in loader:

            t1 = t1.to(device)
            t2 = t2.to(device)
            mask = mask.to(device)

            pred = forward_model(
                model,
                t1,
                t2,
                model_type
            )

            pred_label = torch.argmax(pred, dim=1)

            conf_matrix += compute_confusion_matrix(
                pred_label,
                mask,
                num_classes
            )

    results = evaluate_all(conf_matrix)

    print("\n===== Evaluation Results =====\n")

    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix:\n")
    print(conf_matrix.cpu().numpy())

    return results

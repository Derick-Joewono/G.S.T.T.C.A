import torch
import os
from evaluation.metrics import compute_confusion_matrix, evaluate_all
from utils.forward_model import forward_model

def evaluate_model(model, loader, model_path, num_classes, model_type, device="cuda"):
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} tidak ditemukan!")
        return None

    # Load bobot model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    conf_matrix = torch.zeros((num_classes, num_classes)).to(device)

    print(f"Sedang mengevaluasi model {model_type}...")
    with torch.no_grad():
        for t1, t2, mask, _ in loader:
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)

            pred = forward_model(model, t1, t2, model_type)
            pred_label = torch.argmax(pred, dim=1)

            conf_matrix += compute_confusion_matrix(pred_label, mask, num_classes)

    # Hitung semua metrik
    results = evaluate_all(conf_matrix)

    print("\n" + "="*40)
    print("         HASIL EVALUASI FINAL         ")
    print("="*40)
    
    # Print metrik utama
    main_metrics = ["accuracy", "precision", "recall", "f1", "iou", "kappa"]
    for m in main_metrics:
        if m in results:
            print(f"{m.capitalize():12}: {results[m]:.4f}")
    
    # Print detail IoU per kelas (Sangat penting untuk Change Detection)
    if "_iou_per_class" in results:
        iou_arr = results["_iou_per_class"]
        print("-" * 40)
        print(f"IoU Class 0 (No Change)  : {iou_arr[0]:.4f}")
        print(f"IoU Class 1 (Deforest)   : {iou_arr[1]:.4f}")
        print(f"IoU Class 2 (Forestation): {iou_arr[2]:.4f}")

    print("-" * 40)
    print("Confusion Matrix:")
    print(conf_matrix.cpu().numpy().astype(int))
    print("="*40 + "\n")

    return results
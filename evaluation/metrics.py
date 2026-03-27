import torch

def compute_confusion_matrix(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    # Masking untuk mengabaikan ignore_index -1
    mask = (target >= 0) & (target < num_classes)
    
    # Menghitung histogram distribusi prediksi vs target
    hist = torch.bincount(
        num_classes * target[mask].long() + pred[mask].long(),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def evaluate_all(conf_matrix):
    results = {}
    
    # Tambahkan epsilon untuk menghindari pembagian dengan nol
    eps = 1e-7
    
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp

    # 1. Overall Accuracy (Sesuai key 'accuracy' di train_model)
    total_pixels = conf_matrix.sum()
    results["accuracy"] = (tp.sum() / (total_pixels + eps)).item()

    # 2. Per Class Metrics (Precision, Recall, F1, IoU)
    precision_cls = tp / (tp + fp + eps)
    recall_cls = tp / (tp + fn + eps)
    f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls + eps)
    iou_cls = tp / (tp + fp + fn + eps)

    # 3. Mean Metrics (Nama key sinkron dengan train_model metrics dict)
    results["precision"] = precision_cls.mean().item()
    results["recall"] = recall_cls.mean().item()
    results["f1"] = f1_cls.mean().item()
    results["iou"] = iou_cls.mean().item() # Ini adalah Mean IoU
    
    # 4. Kappa Coefficient
    po = tp.sum() / (total_pixels + eps)
    pe = (conf_matrix.sum(dim=0) * conf_matrix.sum(dim=1)).sum() / (total_pixels ** 2 + eps)
    results["kappa"] = ((po - pe) / (1 - pe + eps)).item()

    # Simpan array mentah untuk detail per kelas jika dibutuhkan
    results["_iou_per_class"] = iou_cls.cpu().numpy()
    
    return results
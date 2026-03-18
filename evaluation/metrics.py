import torch


def compute_confusion_matrix(pred, target, num_classes):

    pred = pred.view(-1)
    target = target.view(-1)

    mask = (target >= 0) & (target < num_classes)

    hist = torch.bincount(
        num_classes * target[mask] + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return hist


def overall_accuracy(conf_matrix):

    correct = torch.diag(conf_matrix).sum()
    total = conf_matrix.sum()

    return (correct / total).item()


def precision_per_class(conf_matrix):

    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp

    precision = tp / (tp + fp + 1e-6)

    return precision


def recall_per_class(conf_matrix):

    tp = torch.diag(conf_matrix)
    fn = conf_matrix.sum(dim=1) - tp

    recall = tp / (tp + fn + 1e-6)

    return recall


def f1_per_class(conf_matrix):

    precision = precision_per_class(conf_matrix)
    recall = recall_per_class(conf_matrix)

    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return f1


def iou_per_class(conf_matrix):

    tp = torch.diag(conf_matrix)

    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp

    iou = tp / (tp + fp + fn + 1e-7)

    return iou


def mean_iou(conf_matrix):

    return iou_per_class(conf_matrix).mean().item()


def mean_f1(conf_matrix):

    return f1_per_class(conf_matrix).mean().item()


def kappa(conf_matrix):

    total = conf_matrix.sum()

    po = torch.diag(conf_matrix).sum() / total

    pe = (
        conf_matrix.sum(dim=0) * conf_matrix.sum(dim=1)
    ).sum() / (total ** 2)

    kappa = (po - pe) / (1 - pe + 1e-7)

    return kappa.item()


def evaluate_all(conf_matrix):

    results = {}

    results["Overall Accuracy"] = overall_accuracy(conf_matrix)

    precision = precision_per_class(conf_matrix)
    recall = recall_per_class(conf_matrix)
    f1 = f1_per_class(conf_matrix)
    iou = iou_per_class(conf_matrix)

    results["Precision per class"] = precision.cpu().numpy()
    results["Recall per class"] = recall.cpu().numpy()
    results["F1 per class"] = f1.cpu().numpy()
    results["IoU per class"] = iou.cpu().numpy()

    results["Mean IoU"] = mean_iou(conf_matrix)
    results["Mean F1"] = mean_f1(conf_matrix)

    results["Kappa"] = kappa(conf_matrix)

    return results
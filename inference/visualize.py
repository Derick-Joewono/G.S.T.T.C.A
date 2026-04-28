import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# NORMALIZE IMAGE
# =========================================================

def normalize_image(img):
    """
    Normalisasi image agar bisa divisualisasikan.
    """

    img = img.astype(np.float32)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img


# =========================================================
# SELECT RGB CHANNELS
# =========================================================
def to_rgb(img, rgb_indices=(2, 1, 0)):
    rgb = img[list(rgb_indices), :, :] 
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = normalize_image(rgb)
    return rgb

# =========================================================
# VISUALIZE CHANGE DETECTION RESULT
# =========================================================

def visualize_change_detection(
    t1,
    t2,
    prediction,
    ground_truth=None,
    save_path=None
):


    t1_rgb = to_rgb(t1)
    t2_rgb = to_rgb(t2)

    n_cols = 4 if ground_truth is not None else 3

    plt.figure(figsize=(5 * n_cols, 5))

    # T1
    plt.subplot(1, n_cols, 1)
    plt.imshow(t1_rgb)
    plt.title("T1 Image")
    plt.axis("off")

    # T2
    plt.subplot(1, n_cols, 2)
    plt.imshow(t2_rgb)
    plt.title("T2 Image")
    plt.axis("off")

    # Prediction
    plt.subplot(1, n_cols, 3)
    plt.imshow(prediction, cmap="jet")
    plt.title("Prediction Map")
    plt.axis("off")

    # Ground Truth
    if ground_truth is not None:

        plt.subplot(1, n_cols, 4)
        plt.imshow(ground_truth, cmap="jet")
        plt.title("Ground Truth")
        plt.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

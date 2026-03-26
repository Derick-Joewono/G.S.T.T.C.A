import torch
import numpy as np
import cv2
import os
from utils.visualize import visualize_change_detection

from configs import train_config as config
from registry.model_registry import MODEL_REGISTRY

def load_model():

    model_info = MODEL_REGISTRY[config.TrainConfig.model_name]

    model = model_info["model"]()
    model_type = model_info["type"]

    checkpoint_path = os.path.join(
        config.TrainConfig.checkpoint_dir,
        f"{config.TrainConfig.model_name}_best.pth"
    )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=config.TrainConfig.device)
    )

    model.to(config.TrainConfig.device)
    model.eval()

    return model, model_type

def load_image(path):

    img = np.load(path)

    img = img.astype(np.float32)

    img = torch.from_numpy(img)

    return img

def predict(model, model_type, t1, t2):

    device = config.TrainConfig.device

    t1 = t1.unsqueeze(0).to(device)
    t2 = t2.unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "early":
            x = torch.cat([t1, t2], dim=1)
            output = model(x)

        elif model_type == "siamese":
            output = model(t1, t2)

        pred = torch.argmax(output, dim=1)
    return pred.squeeze().cpu().numpy()

def save_prediction(pred, save_path):

    pred = pred.astype(np.uint8)
    cv2.imwrite(save_path, pred)

def run_inference(t1_path, t2_path, save_path, show_vis=True): # Tambahkan flag visualisasi

    model, model_type = load_model()
    
    t1 = load_image(t1_path)
    t2 = load_image(t2_path)
    
    # 1. Lakukan prediksi
    prediction = predict(model, model_type, t1, t2)
    
    # 2. Simpan hasil mentah (sebagai gambar)
    save_prediction(prediction, save_path)
    
    # 3. Opsional: Tampilkan Visualisasi Side-by-Side
    if show_vis:
        # Kita perlu memindahkan tensor ke numpy dan menyesuaikan dimensi jika perlu
        # Visualize butuh (C, H, W) untuk T1/T2 dan (H, W) untuk prediction
        visualize_change_detection(
            t1=t1.numpy(), 
            t2=t2.numpy(), 
            prediction=prediction,
            ground_truth=None, # Saat inference biasanya kita tidak punya GT
            save_path=save_path.replace(".png", "_vis.png") # Simpan plotnya juga
        )

    print(f"Inference completed. Result saved to {save_path}")




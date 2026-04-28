import torch
import numpy as np
import cv2
import os
from inference.visualize import visualize_change_detection

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

def run_inference(npz_path, save_path, show_vis=True):
    
    model, model_type = load_model()
    
    # Muat data langsung dari satu file NPZ
    data = np.load(npz_path)
    t1_np = data["t1"]
    t2_np = data["t2"]
    
    # Konversi ke Tensor untuk prediksi
    t1_tensor = torch.from_numpy(t1_np).float()
    t2_tensor = torch.from_numpy(t2_np).float()
    
    prediction = predict(model, model_type, t1_tensor, t2_tensor)
    save_prediction(prediction, save_path)
    
    if show_vis:
        visualize_change_detection(
            t1=t1_np, 
            t2=t2_np, 
            prediction=prediction,
            ground_truth=data["mask"],
            save_path=save_path.replace(".png", "_vis.png") 
        )



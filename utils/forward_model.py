import torch 

def forward_model(model, t1, t2, model_type):

    if model_type == "early":

        x = torch.cat([t1, t2], dim=1)

        pred = model(x)

    elif model_type == "siamese":

        pred = model(t1, t2)

    else:

        raise ValueError("Unknown model_type")

    if isinstance(pred, dict):

        pred = pred["logits"]

    return pred

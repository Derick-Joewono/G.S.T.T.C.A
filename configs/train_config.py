"""
Training Configuration File

Semua parameter eksperimen disimpan di sini agar
train.py tidak memiliki parameter yang hardcoded.
"""


class TrainConfig:

    # ==========================
    # EXPERIMENT INFO
    # ==========================

    experiment_name = "gswin_tac_deforestation"

    # ==========================
    # DATASET
    # ==========================

    dataset_path = "data/deforestation_dataset"

    train_split = "train"
    val_split = "val"
    test_split = "test"

    num_classes = 3

    input_channels = 20

    image_size = 256

    # ==========================
    # TRAINING PARAMETER
    # ==========================

    batch_size = 8

    num_epochs = 30

    learning_rate = 1e-4

    weight_decay = 1e-4

    num_workers = 4

    pin_memory = True

    # ==========================
    # OPTIMIZER
    # ==========================

    optimizer = "AdamW"

    # ==========================
    # SCHEDULER
    # ==========================
    

    scheduler = "cosine"

    t_max = 30

    # ==========================
    # CHECKPOINT
    # ==========================

    save_model = True

    checkpoint_dir = "checkpoints/"

    save_best_only = True

    # ==========================
    # DEVICE
    # ==========================

    device = "cuda"

    # ==========================
    # METRICS
    # ==========================

    metrics = [
        "iou",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "kappa"
    ]
    # ==========================
    # model_name
    # ==========================

    model_name= "gswin_tac"

    #options ->
    # "unet_ef"
    # "deeplabv3_ef"
    # "swin_ef"
    # "segformer_ef"
    # "stanet"
    # "bit"
    # "gswin_tac"

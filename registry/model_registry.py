from models import segFormer,STANet,swin_earlyfusion,BIT,Gswin_tac,UNet,DeepLabV3EarlyFusion

#init params
MODEL_REGISTRY = {
    "unet_ef": {
        "model": lambda: UNet.UNetEarlyFusion(in_channels=20, num_classes=3),
        "type": "early",
        "optimizer": "Adam"
    },

    "deeplabv3_ef": {
        "model": lambda: DeepLabV3EarlyFusion(in_channels=20, num_classes=3),
        "type": "early",
        "optimizer": "Adam"
    },

    "swin_ef": {
        "model": lambda: swin_earlyfusion.SwinEarlyFusionFPN(in_channels=20, num_classes=3),
        "type": "early",
        "optimizer": "AdamW"
    },

    "segformer_ef": {
        "model": lambda: segFormer.SegFormerEarlyFusion(in_channels=20, num_classes=3),
        "type": "early",
        "optimizer": "AdamW"
    },

    "stanet": {
        "model": lambda: STANet.STANet(),
        "type": "siamese",
        "optimizer": "Adam"
    },

    "bit": {
        "model": lambda: BIT.BIT(),
        "type": "siamese",
        "optimizer": "AdamW"
    },

    "gswin_tac": {
        "model": lambda: Gswin_tac.GSWIN_TAC(num_classes=3),
        "type": "early",
        "optimizer": "AdamW"
    }
}
from models import segFormer,STANet,swin_earlyfusion,BIT,Gswin_tac,UNet,DeepLabV3EarlyFusion,CF_GCN,HCGNET

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
    ,
    "cf_gcn": {
        "model": lambda: CF_GCN.BASE_GCN(
            input_nc=10,         # 10 channel satelit
            output_nc=3,         # 3 kelas prediksi
            resnet_stages_num=4, 
            backbone='resnet50'
        ),
        "type": "siamese",       # Harus siamese karena menerima t1 dan t2 terpisah
        "optimizer": "Adam"      
    },
    "hcgnet": {
        "model": lambda: HCGNET.HCGNet(
            in_channels=10,      # 10 channel satelit (diproses per T1/T2 di forward)
            num_classes=3        # 3 kelas prediksi (No Change, Deforest, Forest)
        ),
        "type": "siamese",       # Menggunakan dual-branch encoder
        "optimizer": "AdamW"     # AdamW biasanya lebih optimal untuk arsitektur berbasis Graph/Transformer, namun bisa diganti "Adam" jika diinginkan
    }
}
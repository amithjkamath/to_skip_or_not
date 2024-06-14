
import pytorch_lightning

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
)
from monai.networks.nets import UNet, VNet, AttentionUnet
from toskipornot.models.NoSkipVnet import NoSkipVNet
from toskipornot.models.NoSkipUnet import NoSkipUNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader

import torch

import os
import glob


PATCH_SIZE = 128
DEVICE = "cpu" #"cuda"


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        """
        self._model = VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            act="ReLU",
            dropout_prob_down=0.0,
            dropout_prob_up=(0.0, 0.0),
        )
        """
        """
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        """
        """
        self._model = NoSkipUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=0,
            norm=Norm.BATCH,
            act="ReLU",
            bias=False,
        )
        """
        """
        self._model = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
        """
        self._model = NoSkipVNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            act="ReLU",
            dropout_prob=0.0,
        )
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType("tensor", device=DEVICE), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device=DEVICE), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.test_step_outputs = []

    def forward(self, x):
        return self._model(x)

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (PATCH_SIZE, PATCH_SIZE, 32)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        dice = self.dice_metric(y_pred=outputs, y=labels)
        d = {"test_dice": dice, "test_number": len(outputs)}
        self.test_step_outputs.append(d)
        return d


def inference(saved_path, data_root):

    model = Net.load_from_checkpoint(saved_path)

    # set up the correct data path
    images = sorted(glob.glob(os.path.join(data_root, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_root, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)
    ]
    # For Spleen:
    test_files = data_dicts[34:]

    # set deterministic training for reproducibility
    set_determinism(seed=0)

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    test_ds = CacheDataset(
        data=test_files,
        transform=test_transforms,
        cache_rate=1.0,
        num_workers=4,
    )

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

    trainer = pytorch_lightning.Trainer(
        accelerator=DEVICE,
    )

    trainer.test(model=model, dataloaders=test_loader)
    print(model.test_step_outputs)


if __name__ == "__main__":

    # Top DSC model:
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1013-4201-unet/model-epoch=370-val_loss=0.03-val_dice=0.94.ckpt"
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1109-1942-vnet/model-epoch=480-val_loss=0.04-val_dice=0.93.ckpt"
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1114-5907-noskipunet/model-epoch=390-val_loss=0.08-val_dice=0.85.ckpt"
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1211-4106-attentionunet/model-epoch=461-val_loss=0.04-val_dice=0.93.ckpt"
    saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1215-4821-noskipvnet/model-epoch=484-val_loss=0.04-val_dice=0.93.ckpt"

    # UNet:
    # [{'test_dice': metatensor([[0.9143]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9345]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9482]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9121]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9304]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8856]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8918]]), 'test_number': 1}]

    # VNet:
    # [{'test_dice': metatensor([[0.9610]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9663]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9089]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9211]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9499]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9297]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.6997]]), 'test_number': 1}]

    # NoSkipUNet:
    # [{'test_dice': metatensor([[0.7356]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8686]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.6429]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8333]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8693]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8530]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8577]]), 'test_number': 1}]

    # Attention UNet:
    # [{'test_dice': metatensor([[0.9466]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9536]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8965]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9453]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9398]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9320]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9371]]), 'test_number': 1}]

    # Second highest DSC model:
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1013-4201-unet/model-epoch=421-val_loss=0.03-val_dice=0.93.ckpt"
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1109-1942-vnet/model-epoch=467-val_loss=0.04-val_dice=0.93.ckpt"
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1114-5907-noskipunet/model-epoch=376-val_loss=0.08-val_dice=0.85.ckpt"
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1211-4106-attentionunet/model-epoch=460-val_loss=0.03-val_dice=0.93.ckpt"

    # UNet:
    # [{'test_dice': metatensor([[0.9258]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9525]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9248]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9182]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9378]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9113]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9393]]), 'test_number': 1}]

    # VNet:
    # [{'test_dice': metatensor([[0.9600]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9669]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9451]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9296]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9387]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9277]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8317]]), 'test_number': 1}]

    # NoSkipUNet:
    # [{'test_dice': metatensor([[0.6120]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.7875]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.4199]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8023]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.7031]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.7789]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8262]]), 'test_number': 1}]

    # Attention UNet:
    # [{'test_dice': metatensor([[0.9236]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9420]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8971]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9421]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9298]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9258]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9252]]), 'test_number': 1}]

    # Third highest DSC model:
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1013-4201-unet/model-epoch=223-val_loss=0.03-val_dice=0.93.ckpt"
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1109-1942-vnet/model-epoch=406-val_loss=0.04-val_dice=0.92.ckpt"
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1114-5907-noskipunet/model-epoch=371-val_loss=0.08-val_dice=0.85.ckpt"
    # saved_path = "/Users/amithkamath/repo/monai-from-matlab/logs-202406-1211-4106-attentionunet/model-epoch=459-val_loss=0.04-val_dice=0.93.ckpt"

    # UNet:
    # [{'test_dice': metatensor([[0.8593]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9441]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9098]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8243]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8078]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.7776]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8743]]), 'test_number': 1}]

    # VNet:
    # [{'test_dice': metatensor([[0.9606]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9560]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9294]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9238]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9293]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9402]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8149]]), 'test_number': 1}]

    # NoSkipUNet:
    # [{'test_dice': metatensor([[0.7922]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8587]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.6513]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8293]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8233]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8466]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8150]]), 'test_number': 1}]

    # Attention UNet:
    # [{'test_dice': metatensor([[0.8755]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9434]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.8943]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9355]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9278]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9176]]), 'test_number': 1},
    #  {'test_dice': metatensor([[0.9025]]), 'test_number': 1}]

    # LBP for 2.5 - run it per slice and average.
    # Compare to 2D results - we expect 3D to be better.

    data_root = "/Users/amithkamath/data/MSD/Task09_Spleen"
    inference(saved_path, data_root)
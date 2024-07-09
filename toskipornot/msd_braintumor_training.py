import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
    Spacingd,
    EnsureType,
    RandAffined,
)
from monai.networks.nets import UNet, VNet, BasicUNetPlusPlus, AttentionUnet
from toskipornot.models.NoSkipVnet import NoSkipVNet
from toskipornot.models.NoSkipUnet import NoSkipUNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.config import print_config

import torch
import numpy as np

import os
import glob
from datetime import datetime

PATCH_SIZE = 96
DEVICE = "cuda"  # "cuda"


class Net(pytorch_lightning.LightningModule):
    def __init__(self, data_dir):
        super().__init__()
        """
        self._model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=(64, 128, 256, 512, 1024, 1024),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            act="ReLU",
            bias=False,
        )
        """
        self._model = VNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            act="ReLU",
            dropout_prob_down=0.0,
            dropout_prob_up=(0.0, 0.0),
        )
        """
        self._model = BasicUNetPlusPlus(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            features=(16, 32, 64, 128, 256, 512),
            norm=Norm.BATCH,
            act="ReLU",
            bias=False,
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
        """
        self._model = NoSkipVNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            act="ReLU",
            dropout_prob=0.0,
        )
        """
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType("tensor", device=DEVICE), AsDiscrete(argmax=True, to_onehot=4)])
        self.post_label = Compose([EnsureType("tensor", device=DEVICE), AsDiscrete(to_onehot=4)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.data_dir = data_dir

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # set up the correct data path
        train_images = sorted(glob.glob(os.path.join(self.data_dir, "imagesTr", "*.nii.gz")))
        train_labels = sorted(glob.glob(os.path.join(self.data_dir, "labelsTr", "*.nii.gz")))
        data_dicts = [
            {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
        ]
        train_files, val_files = data_dicts[:334], data_dicts[334:384]

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityd(
                    keys=["image"],
                    minv=0.0,
                    maxv=1.0,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # randomly crop out patch samples from
                # big image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
                    pos=2,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                # user can also add other random transforms
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=0.5,
                    spatial_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
                    rotate_range=(0, 0, np.pi/4),
                    scale_range=(0.4, 0.4, 0.4)
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityd(
                    keys=["image"],
                    minv=0.0,
                    maxv=1.0,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=4,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=1.0,
            num_workers=4,
        )

    #         self.train_ds = monai.data.Dataset(
    #             data=train_files, transform=train_transforms)
    #         self.val_ds = monai.data.Dataset(
    #             data=val_files, transform=val_transforms)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=2,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=4)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        self.log("val_loss", loss)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        self.log("val_dice", mean_val_dice)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.validation_step_outputs.clear()  # free memory
        return {"log": tensorboard_logs}


def main():
    print_config()

    repo_root = "/home/akamath/Documents/to_skip_or_not"
    eventid = datetime.now().strftime('-%Y%m-%d%H-%M%S')
    log_dir = os.path.join(repo_root, "logs" + eventid)

    data_root = "/home/akamath/data/MSD"
    data_dir = os.path.join(data_root, "Task01_BrainTumour")

    # initialise the LightningModule
    net = Net(data_dir)

    # set up loggers and checkpoints
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir)
    checkpoint_callback = ModelCheckpoint(dirpath=log_dir,
                                          monitor="val_loss",
                                          save_top_k=3,
                                          filename="model-{epoch:02d}-{val_loss:.4f}-{val_dice:.4f}",
                                          )

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        accelerator=DEVICE,
        max_epochs=500,
        logger=tb_logger,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=16,
        callbacks=checkpoint_callback,
    )

    # train
    trainer.fit(net)

    print(f"train completed, best_metric: {net.best_val_dice:.4f} " f"at epoch {net.best_val_epoch}")


if __name__ == "__main__":
    main()

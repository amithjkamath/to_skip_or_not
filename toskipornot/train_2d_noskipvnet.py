# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from glob import glob
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid

import monai
from monai.data import (
    list_data_collate,
    decollate_batch,
    DataLoader,
)
from monai.utils import set_determinism
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    SaveImage,
    ScaleIntensityd,
    RandRotate90d,
)
import wandb
from toskipornot.models.NoSkipVnet import NoSkipVNet


def check_dataset(train_files, data_transforms):
    # define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=data_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(
        check_ds, batch_size=2, num_workers=4, collate_fn=list_data_collate
    )
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["seg"].shape)


def train_and_validate(
    model,
    device,
    optimizer,
    scheduler,
    loss_function,
    config,
    train_files,
    val_files,
    train_transforms,
    val_transforms,
    variation,
    seed_num,
):
    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        collate_fn=list_data_collate,
    )

    # start a typical PyTorch training
    max_epochs = config["max_epochs"]
    val_interval = config["val_interval"]
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            # 🐝 log train_loss for each step to wandb
            wandb.log({"train/loss": loss.item()})

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        # step scheduler after each epoch (cosine decay)
        # scheduler.step()

        # 🐝 log train_loss averaged over epoch to wandb
        wandb.log({"train/loss_epoch": epoch_loss})

        # 🐝 log learning rate after each epoch to wandb
        # wandb.log({"learning_rate": scheduler.get_lr()[0]})

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            best_metric, best_metric_epoch = validate_model(
                model,
                device,
                best_metric,
                best_metric_epoch,
                epoch,
                val_loader,
                variation,
                seed_num,
            )

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )

    # 🐝 log best score and epoch number to wandb
    wandb.log(
        {
            "best_dice_metric": best_metric,
            "best_metric_epoch": best_metric_epoch,
        }
    )
    return model


def validate_model(
    model,
    device,
    best_metric,
    best_metric_epoch,
    epoch,
    val_loader,
    variation,
    seed_num,
):
    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    model.eval()
    with torch.no_grad():
        val_images = None
        val_labels = None
        val_outputs = None
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(
                device
            )
            val_outputs = model(val_images)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

            image_caption = "Example validation output / label at epoch:" + str(epoch)
            val_image = torch.hstack((val_outputs[0], val_labels[0]))
            grid = make_grid(val_image[0])
            images = wandb.Image(grid[0, :, :], caption=image_caption)
            wandb.log({"Val output": images})

            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()
        output_folder = "./NoSkipVNet_dice_" + variation + "_seed_" + str(seed_num)
        os.makedirs(output_folder, exist_ok=True)
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(
                model.state_dict(),
                os.path.join(
                    output_folder, "best_metric_model_segmentation2d_dict.pth"
                ),
            )
            print("saved new best metric model")
        print(
            "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                epoch + 1, metric, best_metric, best_metric_epoch
            )
        )
        # 🐝 log validation dice score for each validation round
        wandb.log({"val/dice_metric": metric})
    return best_metric, best_metric_epoch


def test_model(model, device, config, test_files, data_transforms, variation, seed_num):
    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    model.eval()

    test_ds = monai.data.Dataset(data=test_files, transform=data_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        collate_fn=list_data_collate,
    )

    output_folder = "./NoSkipVNet_dice_" + variation + "_seed_" + str(seed_num)
    output_saver = SaveImage(
        output_dir=output_folder, output_ext=".png", output_postfix="out"
    )
    label_saver = SaveImage(
        output_dir=output_folder, output_ext=".png", output_postfix="seg"
    )

    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data["img"].to(device), test_data["seg"].to(
                device
            )
            test_outputs = model(test_images)
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            test_labels = decollate_batch(test_labels)
            # compute metric for current iteration
            dice_metric(y_pred=test_outputs, y=test_labels)
            for test_output in test_outputs:
                output_saver(np.multiply(test_output, 255))
            for test_label in test_labels:
                label_saver(np.multiply(test_label, 255))
        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        print("Test evaluation metric:", metric)
        # reset the status
        dice_metric.reset()
        # 🐝 log validation dice score for each validation round
        wandb.log({"test/dice_metric": metric})


def main():
    """
    Runs the training and validation loop for Synthetic data segmentation.
    """
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Setup data directory
    root_dir = "/home/akamath/Documents/toskipornot"

    data_variations = [
        "alphablend_0p10_normal",
        "alphablend_0p20_normal",
        "alphablend_0p30_normal",
        "alphablend_0p40_normal",
        "alphablend_0p50_normal",
        "alphablend_0p60_normal",
        "alphablend_0p70_normal",
        "alphablend_0p80_normal",
        "alphablend_0p82_normal",
        "alphablend_0p85_normal",
        "alphablend_0p88_normal",
        "alphablend_0p90_normal",
        "alphablend_0p92_normal",
        "alphablend_0p95_normal",
        "alphablend_0p98_normal",
    ]

    for seed_num in range(1, 6):
        for variation in data_variations:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cpu")

            # Define Configuration
            config = {
                # data
                "cache_rate": 1.0,
                "num_workers": 2,
                "seed": seed_num,
                # train settings
                "train_batch_size": 8,
                "val_batch_size": 1,
                "learning_rate": 1e-3,
                "max_epochs": 100,
                "val_interval": 2,  # check validation score after n epochs
                # UNet Model
                "model_type": "NoSkipVNet",  # just to keep track
                "model_params": dict(
                    spatial_dims=2,
                    in_channels=1,
                    out_channels=1,
                    act="ReLU",
                    dropout_prob=0.0,
                    # bias=False,
                ),
            }

            # 1. Start a W&B Run
            # now = datetime.now()  # current date and time
            # date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
            project_name = variation + "_seed_" + str(seed_num)
            run = wandb.init(
                project="synthetic-texture-NoSkipVNet-dice-v6",
                name=project_name,
                config=config,
            )

            # Set deterministic training for reproducibility
            set_determinism(seed=config["seed"])

            # create a temporary directory and 40 random image, mask pairs
            # data_dir = "/home/akamath/Documents/toskipornot/data/BUSI-processed/"
            data_dir = os.path.join(root_dir, "data", "foreground-processed", variation)

            images = sorted(glob(os.path.join(data_dir, "train", "*")))
            masks = sorted(glob(os.path.join(data_dir, "mask", "*")))

            train_files = [
                {"img": img, "seg": mask} for img, mask in zip(images[:70], masks[:70])
            ]
            val_files = [
                {"img": img, "seg": mask}
                for img, mask in zip(images[70:80], masks[70:80])
            ]
            test_files = [
                {"img": img, "seg": mask}
                for img, mask in zip(images[80:100], masks[80:100])
            ]

            # define transforms for image and segmentation
            train_transforms = Compose(
                [
                    LoadImaged(keys=["img", "seg"]),
                    EnsureChannelFirstd(keys=["img", "seg"]),
                    ScaleIntensityd(keys=["img", "seg"]),
                    RandRotate90d(keys=["img", "seg"], prob=0.5),
                ]
            )

            val_transforms = Compose(
                [
                    LoadImaged(keys=["img", "seg"]),
                    EnsureChannelFirstd(keys=["img", "seg"]),
                    ScaleIntensityd(keys=["img", "seg"]),
                ]
            )

            check_dataset(train_files, train_transforms)

            # create UNet, DiceLoss and Adam optimizer
            model = NoSkipVNet(**config["model_params"]).to(device)
            wandb.watch(model, log_freq=100)
            loss_function = monai.losses.DiceLoss(sigmoid=True)
            # loss_function = monai.losses.DiceCELoss(sigmoid=True)
            # loss_function = torch.nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(model.parameters(), config["learning_rate"])
            scheduler = CosineAnnealingLR(
                optimizer, T_max=config["max_epochs"], eta_min=1e-9
            )

            model = train_and_validate(
                model,
                device,
                optimizer,
                scheduler,
                loss_function,
                config,
                train_files,
                val_files,
                train_transforms,
                val_transforms,
                variation,
                seed_num,
            )

            test_model(
                model, device, config, test_files, val_transforms, variation, seed_num
            )
            # 🐝 Close your wandb run
            wandb.finish()


if __name__ == "__main__":
    main()

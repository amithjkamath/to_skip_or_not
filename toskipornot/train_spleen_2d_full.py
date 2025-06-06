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
import pandas as pd
import sys
import torch

import monai
from monai.data import (
    list_data_collate,
    DataLoader,
)
from monai.utils import set_determinism
from monai.networks.nets import AttentionUnet, UNet, VNet, BasicUNetPlusPlus
from monai.networks.layers import Norm

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
)
import wandb
from toskipornot.models.NoSkipUnet import NoSkipUNet
from toskipornot.models.NoSkipVnet import NoSkipVNet
from toskipornot.models.utils import *


def check_dataset(train_files, data_transforms):
    # define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=data_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(
        check_ds, batch_size=2, num_workers=4, collate_fn=list_data_collate
    )
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["seg"].shape)


def main():
    """
    Runs the training and validation loop for 2D MSD spleen data.
    """
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Setup data directory
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Configuration
    config = {
        # data
        "cache_rate": 1.0,
        "num_workers": 2,
        # train settings
        "train_batch_size": 8,
        "val_batch_size": 1,
        "max_epochs": 100,
        "val_interval": 2,  # check validation score after n epochs
        # UNet Model
        "att_model_params": dict(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
        ),
        "vnet_model_params": dict(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            act="ReLU",
            dropout_prob=0.0,
        ),
        "unet_model_params": dict(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=0,
            norm=Norm.BATCH,
            act="ReLU",
            bias=False,
        ),
        "unetplusplus_model_params": dict(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            features=(16, 32, 64, 128, 256, 512),
            norm=Norm.BATCH,
            act="ReLU",
            bias=False,
        ),
        "noskipvnet_model_params": dict(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            act="ReLU",
            dropout_prob=0.0,
        ),
        "noskipunet_model_params": dict(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=0,
            norm=Norm.BATCH,
            act="ReLU",
            bias=False,
        ),
    }

    # create a temporary directory and 40 random image, mask pairs
    data_dir = os.path.join(root_dir, "data/Task09_Spleen-processed")
    net_name = ["NoSkipUNet", "NoSkipVNet", "UNet", "VNet", "AttentionUNet", "UNet++"]
    seed_list = [1, 2, 3]
    swin_list = [256]
    learn_rate = 0.01

    for seed_num in seed_list:
        for swin_size in swin_list:
            for net in net_name:
                # 1. Start a W&B Run
                config["seed"] = seed_num
                config["learning_rate"] = learn_rate
                config["net_name"] = net
                config["swin_size"] = swin_size

                # Set deterministic training for reproducibility
                set_determinism(seed=config["seed"])

                train_df = pd.read_csv(
                    os.path.join(data_dir, "spleen_train_all_texture.csv")
                )

                images = sorted(train_df.iloc[:, 1].values.tolist())
                masks = [im.replace("image", "label") for im in images]

                train_files = [
                    {"img": img, "seg": mask} for img, mask in zip(images, masks)
                ]

                valid_df = pd.read_csv(
                    os.path.join(data_dir, "spleen_valid_all_texture.csv")
                )

                images = sorted(valid_df.iloc[:, 1].values.tolist())
                masks = [im.replace("image", "label") for im in images]

                val_files = [
                    {"img": img, "seg": mask} for img, mask in zip(images, masks)
                ]

                test_df = pd.read_csv(
                    os.path.join(data_dir, "spleen_test_all_texture.csv")
                )

                images = sorted(test_df.iloc[:, 1].values.tolist())
                masks = [im.replace("image", "label") for im in images]

                test_files = [
                    {"img": img, "seg": mask} for img, mask in zip(images, masks)
                ]

                # define transforms for image and segmentation
                if swin_size == 256:
                    train_transforms = Compose(
                        [
                            LoadImaged(keys=["img", "seg"]),
                            EnsureChannelFirstd(keys=["img", "seg"]),
                            ScaleIntensityd(keys=["img", "seg"]),
                            RandRotate90d(keys=["img", "seg"], prob=0.5),
                        ]
                    )
                else:
                    train_transforms = Compose(
                        [
                            LoadImaged(keys=["img", "seg"]),
                            EnsureChannelFirstd(keys=["img", "seg"]),
                            ScaleIntensityd(keys=["img", "seg"]),
                            RandCropByPosNegLabeld(
                                keys=["img", "seg"],
                                label_key="seg",
                                spatial_size=[swin_size, swin_size],
                                pos=1,
                                neg=1,
                                num_samples=config["train_batch_size"],
                            ),
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
                if net == "AttentionUNet":
                    model = AttentionUnet(**config["att_model_params"]).to(device)
                elif net == "NoSkipUNet":
                    model = NoSkipUNet(**config["noskipunet_model_params"]).to(device)
                elif net == "UNet":
                    model = UNet(**config["unet_model_params"]).to(device)
                elif net == "VNet":
                    model = VNet(**config["vnet_model_params"]).to(device)
                elif net == "NoSkipVNet":
                    model = NoSkipVNet(**config["noskipvnet_model_params"]).to(device)
                elif net == "UNet++":
                    model = BasicUNetPlusPlus(**config["unetplusplus_model_params"]).to(
                        device
                    )

                project = "spleen-" + net + "-full-v3"
                project_name = net + "_" + str(swin_size) + "_" + str(seed_num)
                output_folder = os.path.join(
                    root_dir, "reports", "spleen-v3", project_name
                )
                os.makedirs(output_folder, exist_ok=True)
                wandb.init(
                    project=project,
                    name=project_name,
                    config=config,
                    save_code=True,
                )

                wandb.watch(model, log_freq=100)
                loss_function = monai.losses.DiceLoss(sigmoid=True)
                # loss_function = monai.losses.DiceCELoss(sigmoid=True)
                # loss_function = torch.nn.BCEWithLogitsLoss()

                optimizer = torch.optim.Adam(
                    model.parameters(), config["learning_rate"]
                )

                model = train_and_validate(
                    model,
                    device,
                    optimizer,
                    loss_function,
                    config,
                    train_files,
                    val_files,
                    train_transforms,
                    val_transforms,
                    output_folder,
                    swin_size=swin_size,
                )

                test_model(
                    model,
                    device,
                    config,
                    test_files,
                    val_transforms,
                    output_folder,
                    swin_size=swin_size,
                )
                # 🐝 Close your wandb run
                wandb.finish()


if __name__ == "__main__":
    main()

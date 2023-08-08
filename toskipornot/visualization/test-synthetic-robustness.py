# -*- coding: utf-8 -*-
"""
Synthetic data experiments for resilience of UNets with texture.
"""

import os
from glob import glob
import numpy as np
import pandas as pd
import torch

from monai.transforms import (
    Activations,
    AsDiscrete,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Compose,
    LoadImaged,
    SaveImage,
)

from monai.networks.nets import AttentionUnet, UNet, BasicUNetPlusPlus, VNet
from monai.networks.layers import Norm
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)
from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate
from monai.config import print_config
from toskipornot.models.NoSkipUnet import NoSkipUNet
from toskipornot.models.NoSkipVnet import NoSkipVNet


def test_robustness(root_dir, net, model_path, output_folder):
    """
    Runs the test loop for Synthetic data segmentation.
    """

    data_variations = [
        "alphablend_0p10_normal",
        "alphablend_0p20_normal",
        "alphablend_0p30_normal",
        "alphablend_0p40_normal",
        "alphablend_0p50_normal",
        "alphablend_0p60_normal",
        "alphablend_0p70_normal",
        "alphablend_0p80_normal",
        "alphablend_0p90_normal",
    ]

    dice_results = {}
    hd_results = {}
    sdsc_results = {}
    surfdist_results = {}

    for variation in data_variations:
        print("For data in range: ", variation)
        train_data_dir = os.path.join(
            root_dir, "data", "background-processed", variation
        )
        images = sorted(glob(os.path.join(train_data_dir, "train", "*")))
        masks = sorted(glob(os.path.join(train_data_dir, "mask", "*")))

        test_files = [
            {"image": img, "label": mask} for img, mask in zip(images[80:], masks[80:])
        ]

        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys=["image", "label"]),
            ]
        )

        # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define Configuration
        config = {
            # data
            "cache_rate": 1.0,
            "num_workers": 1,
            # train settings
            "val_batch_size": 1,
            # UNet Model
            "att_model_params": dict(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
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
            "unet_model_params": dict(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=0,
                norm=Norm.BATCH,
                act="ReLU",
                # bias=False,
            ),
            "vnet_model_params": dict(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                act="ReLU",
                dropout_prob=0.0,
                # bias=False,
            ),
            "noskipunet_model_params": dict(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=0,
                norm=Norm.BATCH,
                act="ReLU",
                # bias=False,
            ),
            "noskipvnet_model_params": dict(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                act="ReLU",
                dropout_prob=0.0,
                # bias=False,
            ),
        }

        if net == "AttentionUNet":
            model = AttentionUnet(**config["att_model_params"])
        elif net == "NoSkipUNet":
            model = NoSkipUNet(**config["noskipunet_model_params"])
        elif net == "NoSkipVNet":
            model = NoSkipVNet(**config["noskipvnet_model_params"])
        elif net == "unet":
            model = UNet(**config["unet_model_params"])
        elif net == "UNet++":
            model = BasicUNetPlusPlus(**config["unetplusplus_model_params"])
        elif net == "vnet":
            model = VNet(**config["vnet_model_params"])
        model.load_state_dict(
            torch.load(
                os.path.join(model_path, "best_metric_model_segmentation2d_dict.pth")
            )
        )
        model.eval()

        dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )
        dice_metric.reset()

        hausdorff_metric = HausdorffDistanceMetric(
            include_background=True, reduction="mean"
        )
        hausdorff_metric.reset()

        surface_dsc_metric = SurfaceDiceMetric(
            [5], include_background=True, reduction="mean"
        )
        surface_dsc_metric.reset()

        surface_distance_metric = SurfaceDistanceMetric(
            symmetric=True, include_background=True, reduction="mean"
        )
        surface_distance_metric.reset()

        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(
            test_ds,
            batch_size=config["val_batch_size"],
            num_workers=config["num_workers"],
            collate_fn=list_data_collate,
        )

        this_output_folder = os.path.join(output_folder, variation)
        os.makedirs(this_output_folder, exist_ok=True)
        output_saver = SaveImage(
            output_dir=this_output_folder, output_ext=".png", output_postfix="out"
        )
        label_saver = SaveImage(
            output_dir=this_output_folder, output_ext=".png", output_postfix="seg"
        )

        test_dice_mean = []
        test_hausdorff_mean = []
        test_surfacedice_mean = []
        test_surfacedistance_mean = []

        with torch.no_grad():
            for test_data in test_loader:
                test_images, test_labels = test_data["image"].to(device), test_data[
                    "label"
                ].to(device)
                test_outputs = model(test_images)
                if type(test_outputs) is list:
                    test_outputs = test_outputs[0]
                test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
                test_labels = decollate_batch(test_labels)
                # compute metric for current iteration
                dice_metric(y_pred=test_outputs, y=test_labels)
                dsc_metric = dice_metric.aggregate().item()
                dice_metric.reset()

                hausdorff_metric(y_pred=test_outputs, y=test_labels)
                hd_metric = hausdorff_metric.aggregate().item()
                hausdorff_metric.reset()

                surface_dsc_metric(y_pred=test_outputs, y=test_labels)
                surface_dsc_value = surface_dsc_metric.aggregate().item()
                surface_dsc_metric.reset()

                surface_distance_metric(y_pred=test_outputs, y=test_labels)
                surface_distance_value = surface_distance_metric.aggregate().item()
                surface_distance_metric.reset()

                test_dice_mean.append(dsc_metric)
                test_hausdorff_mean.append(hd_metric)
                test_surfacedice_mean.append(surface_dsc_value)
                test_surfacedistance_mean.append(surface_distance_value)

                for test_output in test_outputs:
                    output_saver(np.multiply(test_output, 255))
                for test_label in test_labels:
                    label_saver(np.multiply(test_label, 255))

        dice_results[variation] = np.mean(test_dice_mean)
        hd_results[variation] = np.mean(test_hausdorff_mean)
        sdsc_results[variation] = np.mean(test_surfacedice_mean)
        surfdist_results[variation] = np.mean(test_surfacedistance_mean)

    return dice_results, hd_results, sdsc_results, surfdist_results


if __name__ == "__main__":
    print_config()
    root_dir = "/Users/amithkamath/repo/to_skip_or_not/"
    model_path = os.path.join(root_dir, "models/background-experiments-oneseed/")

    for net in [
        "UNet++",
        "vnet",
        "NoSkipVNet",
        # "unet",
        # "NoSkipUNet",
        # "AttentionUNet",
    ]:
        dice_for_model = {}
        hd_for_model = {}
        sdsc_for_model = {}
        surfdist_for_model = {}

        for idx in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            train_model_name = "alphablend_0p" + str(idx).zfill(2) + "_normal_seed_1"
            output_path = os.path.join(
                root_dir,
                "reports",
                "background_robustness",
                net,
                train_model_name,
            )
            os.makedirs(output_path, exist_ok=True)
            dice, hd, sdsc, surfdist = test_robustness(
                root_dir,
                net,
                os.path.join(model_path, net + "_dice_" + train_model_name),
                output_path,
            )
            dice_for_model[train_model_name] = dice
            hd_for_model[train_model_name] = hd
            sdsc_for_model[train_model_name] = sdsc
            surfdist_for_model[train_model_name] = surfdist

        dice_df = pd.DataFrame.from_dict(dice_for_model)
        hd_df = pd.DataFrame.from_dict(hd_for_model)
        sdsc_df = pd.DataFrame.from_dict(sdsc_for_model)
        surfdist_df = pd.DataFrame.from_dict(surfdist_for_model)

        dice_df.to_csv("dice_for_" + net + ".csv")
        hd_df.to_csv("hd_for_" + net + ".csv")
        sdsc_df.to_csv("sdsc_for_" + net + ".csv")
        surfdist_df.to_csv("surfdist_for_" + net + ".csv")

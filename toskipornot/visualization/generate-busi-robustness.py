import os
from glob import glob
import json
import pandas as pd

import torch

from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)
from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate

from toskipornot.features.analyze_features import *


def run_segmentation(model, config, images, labels, output_path, variant, seed_num):
    test_files = [{"image": img, "label": mask} for img, mask in zip(images, labels)]
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image", "label"]),
        ]
    )

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=config["val_batch_size"]["value"],
        num_workers=1,
        collate_fn=list_data_collate,
    )

    model.eval()
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    dice_metric = DiceMetric(include_background=True, reduction="mean")
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

    output_saver = SaveImage(
        output_dir=os.path.join(
            output_path,
            "BUSI_stats_" + model_name + "_" + variant + "_256_" + str(seed_num),
        ),
        output_ext=".png",
        output_postfix="out",
    )

    test_dice = []
    test_hausdorff = []
    test_surfaceDistance = []
    test_surfaceDSC = []

    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data["image"], test_data["label"]
            test_outputs = model(test_images)
            if isinstance(test_outputs, list):
                test_outputs = test_outputs[0]
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            test_labels = [post_label(i) for i in decollate_batch(test_labels)]

            dice_metric(y_pred=test_outputs, y=test_labels)
            dsc_value = dice_metric.aggregate().item()
            dice_metric.reset()

            hausdorff_metric(y_pred=test_outputs, y=test_labels)
            hd_value = hausdorff_metric.aggregate().item()
            hausdorff_metric.reset()

            surface_dsc_metric(y_pred=test_outputs, y=test_labels)
            surface_dsc_value = surface_dsc_metric.aggregate().item()
            surface_dsc_metric.reset()

            surface_distance_metric(y_pred=test_outputs, y=test_labels)
            surface_distance_value = surface_distance_metric.aggregate().item()
            surface_distance_metric.reset()

            test_dice.append(dsc_value)
            test_hausdorff.append(hd_value)
            test_surfaceDSC.append(surface_dsc_value)
            test_surfaceDistance.append(surface_distance_value)

            for test_output in test_outputs:
                output_saver(np.multiply(test_output, 255))

    return test_dice, test_hausdorff, test_surfaceDSC, test_surfaceDistance


def generate_plots(data_path, model_path, output_path, model_name, variant, seed_num):
    data_images = os.path.join(data_path, "image")
    images = sorted(glob(os.path.join(data_images, "*")))
    data_labels = os.path.join(data_path, "label")
    labels = sorted(glob(os.path.join(data_labels, "*")))

    config_path = os.path.join(
        model_path, model_name + "_256_" + str(seed_num), "config.json"
    )
    with open(config_path) as config_file:
        config = json.load(config_file)

    if model_name == "AttentionUNet":
        model = AttentionUnet(**config["att_model_params"]["value"])
    elif model_name == "NoSkipUNet":
        model = NoSkipUNet(**config["noskipunet_model_params"]["value"])
    elif model_name == "NoSkipVNet":
        model = NoSkipVNet(**config["noskipvnet_model_params"]["value"])
    elif model_name == "UNet":
        model = UNet(**config["unet_model_params"]["value"])
    elif model_name == "UNet++":
        model = BasicUNetPlusPlus(**config["unetplusplus_model_params"]["value"])
    elif model_name == "VNet":
        model = VNet(**config["vnet_model_params"]["value"])
    model.load_state_dict(
        torch.load(
            os.path.join(
                model_path,
                model_name + "_256_" + str(seed_num),
                "best_metric_model_segmentation2d_dict.pth",
            )
        )
    )
    model.eval()

    fold_state = find_split(config, len(images))
    (
        dice_scores,
        hd_scores,
        surface_dsc_scores,
        surface_distance_scores,
    ) = run_segmentation(model, config, images, labels, output_path, variant, seed_num)

    df = pd.DataFrame(
        list(
            zip(
                images,
                fold_state,
                dice_scores,
                hd_scores,
                surface_dsc_scores,
                surface_distance_scores,
            )
        ),
        columns=[
            "Filename",
            "Set",
            "Dice",
            "HD",
            "SurfaceDSC",
            "SurfaceDistance",
        ],
    )

    os.makedirs(os.path.join(model_path, "results"), exist_ok=True)
    df.to_csv(
        os.path.join(
            output_path,
            "BUSI_stats_"
            + model_name
            + "_"
            + variant
            + "_256_"
            + str(seed_num)
            + ".csv",
        )
    )


if __name__ == "__main__":
    root_dir = "/home/akamath/Documents/toskipornot/"
    model_dir = os.path.join(root_dir, "reports", "busi-v4")
    output_path = os.path.join(root_dir, "reports", "BUSI-results")
    for variant in ["lower", "low", "in-domain", "high", "higher"]:
        for seed_num in [1, 2, 3]:
            data_path = os.path.join(
                root_dir, "data", "BUSI-experiment", variant, "test"
            )

            for model_name in [
                "AttentionUNet",
                "UNet",
                "NoSkipUNet",
                "UNet++",
                "VNet",
                "NoSkipVNet",
            ]:
                generate_plots(
                    data_path, model_dir, output_path, model_name, variant, seed_num
                )

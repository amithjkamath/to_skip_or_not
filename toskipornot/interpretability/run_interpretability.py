# %%
import os
import torch
from glob import glob

import numpy as np

from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate

from monai.transforms import (
    Activations,
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    SaveImage,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm

from toskipornot.interpretability.attributionalgorithms import IG_simple
from toskipornot.interpretability.baseline_generator import GlobalMinimumBaseline
from toskipornot.interpretability.image_interpolation import LinearImageInterpolator

# %%

if __name__ == "__main__":
    root_dir = "/Users/amithkamath/repo/toskipornot/"
    model_path = "/Users/amithkamath/repo/toskipornot/reports/busi-v4/UNet_256_1/best_metric_model_segmentation2d_dict.pth"
    output_path = os.path.join(root_dir, "reports", "BUSI-interpretability")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
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
    }

    model = UNet(**config["unet_model_params"]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # %%
    baseline = GlobalMinimumBaseline()
    interpolation = LinearImageInterpolator(5)
    IG = IG_simple(model, baseline, interpolation)

    # %%
    for variant in ["lower", "low", "in-domain", "high", "higher"]:
        data_path = os.path.join(root_dir, "data", "BUSI-experiment", variant, "test")

        data_images = os.path.join(data_path, "image")
        images = sorted(glob(os.path.join(data_images, "*")))
        data_labels = os.path.join(data_path, "label")
        labels = sorted(glob(os.path.join(data_labels, "*")))

        test_files = [
            {"image": img, "label": mask} for img, mask in zip(images, labels)
        ]
        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys=["image", "label"]),
            ]
        )

        output_saver = SaveImage(
            output_dir=os.path.join(
                output_path,
                "BUSI_stats_UNet" + "_" + variant + "_256_",
            ),
            output_ext=".png",
            output_postfix="out",
        )

        test_ds = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            num_workers=1,
            collate_fn=list_data_collate,
        )

        with torch.no_grad():
            for test_data in test_loader:
                test_images, test_labels = test_data["image"], test_data["label"]
                integrated_gradients = IG.execute(test_images, 0)
                output_saver(np.multiply(integrated_gradients, 255))
                print("saved IG!")

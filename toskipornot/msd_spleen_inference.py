
import glob
import os
import pandas as pd

import pytorch_lightning
from monai.data import CacheDataset, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import AttentionUnet, UNet, VNet, BasicUNetPlusPlus
from monai.networks.layers import Norm
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRicianNoised,
    RandCoarseDropoutd,
)
from monai.utils import set_determinism

from toskipornot.models.NoSkipUnet import NoSkipUNet
from toskipornot.models.NoSkipVnet import NoSkipVNet

PATCH_SIZE = 128
DEVICE = "cpu" #"cuda"


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
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
        self._model = NoSkipVNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            act="ReLU",
            dropout_prob=0.0,
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
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType("tensor", device=DEVICE), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device=DEVICE), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.hausdorff_metric_95 = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean", get_not_nans=False)
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
        self.dice_metric.reset()

        hd = self.hausdorff_metric(y_pred=outputs, y=labels)
        self.hausdorff_metric.reset()

        hd_95 = self.hausdorff_metric_95(y_pred=outputs, y=labels)
        self.hausdorff_metric_95.reset()

        d = {"test_dice": dice, "test_hd100": hd, "test_hd95": hd_95, "test_number": len(outputs)}
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

    # Run these transforms in the image input to simulate noise.
    # https://docs.monai.io/en/stable/transforms.html#gibbsnoised
    # https://docs.monai.io/en/stable/transforms.html#randcoarsedropoutd
    # https://docs.monai.io/en/stable/transforms.html#randriciannoised
    # https://docs.monai.io/en/stable/transforms.html#randgaussiannoised
    # https://docs.monai.io/en/stable/transforms.html#randgaussiansmoothd
    # https://docs.monai.io/en/stable/transforms.html#kspacespikenoised
    # https://docs.monai.io/en/stable/transforms.html#randkspacespikenoised

    # write code to filter image using opencv and then apply the transforms.


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
            #RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=0.5, allow_missing_keys=False, sample_std=True),
            #RandGaussianSmoothd(keys=["image"], prob=1.0, sigma_x=(0.1, 0.1), sigma_y=(0.1, 0.1), sigma_z=(0.1, 0.1), allow_missing_keys=False),
            #RandRicianNoised(keys=["image"], prob=1.0, mean=0.0, std=0.9),
            #RandCoarseDropoutd(keys=["image"], prob=1, holes=256, spatial_size=3, fill_value=0),
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
    results_dict = {}
    for idx in range(len(test_files)):
        results_dict[idx] = {'image': test_files[idx]['image'],
                             'dice': model.test_step_outputs[idx]['test_dice'].cpu().detach().numpy()[0][0],
                             'hd100': model.test_step_outputs[idx]['test_hd100'].cpu().detach().numpy()[0][0],
                             'hd95': model.test_step_outputs[idx]['test_hd95'].cpu().detach().numpy()[0][0],
                             }
    results_df = pd.DataFrame.from_dict(results_dict)

    save_file_name = os.path.join(os.path.split(saved_path)[:-1][0], "test_randriciannoise_0p9.csv")
    results_df.transpose().to_csv(save_file_name)
    print(results_df.transpose())


if __name__ == "__main__":
    data_root = "/Users/amithkamath/data/MSD/Task09_Spleen"
    saved_path = "/Users/amithkamath/repo/to_skip_or_not/reports/3d-results/logs-202406-1814-5400-unet-spleen/model-epoch=299-val_loss=0.0323-val_dice=0.9377.ckpt"

    # LBP for 2.5 - run it per slice and average.
    # Compare to 2D results - we expect 3D to be better.
    inference(saved_path, data_root)
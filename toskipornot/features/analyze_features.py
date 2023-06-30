import numpy as np
import torch
from tqdm import tqdm

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import local_binary_pattern, canny
from skimage.measure import perimeter
from skimage.morphology import disk, binary_dilation

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference

from monai.transforms import (
    Activations,
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
)


def run_segmentation(model, config, images, labels):

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
    hausdorff_metric = HausdorffDistanceMetric(
        include_background=True, reduction="mean"
    )
    dice_metric.reset()
    test_dice = []
    test_hausdorff = []
    test_oversegmentation = []

    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data["image"], test_data["label"]
            if config["swin_size"] == 256:
                test_outputs = model(test_images)
            else:
                roi_size = (config["swin_size"], config["swin_size"])
                sw_batch_size = 4
                test_outputs = sliding_window_inference(
                    test_images, roi_size, sw_batch_size, model
                )
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            test_labels = [post_label(i) for i in decollate_batch(test_labels)]

            oversegment_ratio = (
                np.sum(test_outputs[0]) - np.sum(test_labels[0])
            ) / np.prod(test_outputs[0].shape)
            test_oversegmentation.append(oversegment_ratio)

            dice_metric(y_pred=test_outputs, y=test_labels)
            dsc_metric = dice_metric.aggregate().item()
            dice_metric.reset()

            hausdorff_metric(y_pred=test_outputs, y=test_labels)
            hd_metric = hausdorff_metric.aggregate().item()
            hausdorff_metric.reset()

            test_dice.append(dsc_metric)
            test_hausdorff.append(hd_metric)

    return test_oversegmentation, test_dice, test_hausdorff


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def run_lbp_analysis(images, labels):

    distance_scores = []
    n_images = len(images)

    for image_index in tqdm(range(n_images)):
        image = imread(images[image_index])
        label = imread(labels[image_index])

        image = resize(image, (256, 256), anti_aliasing=True)
        label = resize(label, (256, 256), anti_aliasing=False)

        radius = 3
        n_points = 8 * radius
        METHOD = "uniform"

        lbp = local_binary_pattern(image, n_points, radius, METHOD)

        lbp_fg = lbp.copy()
        img_fg = image.copy()
        lbp_fg[label == 0] = np.nan
        img_fg[label == 0] = 0

        lbp_bg = lbp.copy()
        img_bg = image.copy()
        lbp_bg[label > 0] = np.nan
        img_bg[label > 0] = 0

        n_bins = int(lbp.max() + 1)
        hist_fg, _ = np.histogram(lbp_fg, density=True, bins=n_bins, range=(0, n_bins))
        hist_bg, _ = np.histogram(lbp_bg, density=True, bins=n_bins, range=(0, n_bins))
        score = kullback_leibler_divergence(hist_fg, hist_bg)
        distance_scores.append(score)
    return distance_scores


def run_texture_flatness_analysis(images, labels):

    flatness_scores = []
    n_images = len(images)

    for image_index in tqdm(range(n_images)):
        image = imread(images[image_index])
        label = imread(labels[image_index])

        image = resize(image, (256, 256), anti_aliasing=True)
        label = resize(label, (256, 256), anti_aliasing=False)

        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, "uniform")
        # greater 5 and less than 19 are 'non-flat' regions, which we want to quantify.
        # Note: 5 and 19 are 1/4 and 3/4 of the boundaries of the values LBP can get.
        # This depends on the choice of radius and n_points. We leave it as in doc for now.
        score = 1 - np.sum(
            np.logical_and(np.greater(lbp, 5), np.less(lbp, 19))
        ) / np.prod(lbp.shape)
        flatness_scores.append(score)
    return flatness_scores


def run_shape_analysis(labels):
    perimeter_scores = []
    n_images = len(labels)

    for image_index in tqdm(range(n_images)):
        label = imread(labels[image_index])
        label = resize(label, (256, 256), anti_aliasing=False)
        measure = perimeter(label, neighbourhood=4)
        perimeter_scores.append(measure)
    return perimeter_scores


def run_edge_analysis(images, labels):

    fore_edge_density = []
    back_edge_density = []
    boundary_edge_density = []

    n_images = len(images)

    for image_index in tqdm(range(n_images)):
        image = imread(images[image_index])
        label = imread(labels[image_index])

        image = resize(image, (256, 256), anti_aliasing=True)
        label = resize(label, (256, 256), anti_aliasing=False)

        edges = canny(image)

        edge_boundary = edges.copy().astype("float64")
        boundary_mask = label.copy()
        boundary_contour = canny(boundary_mask)
        footprint = disk(3)
        dilated_boundary = binary_dilation(boundary_contour, footprint)
        edge_boundary[dilated_boundary == 0] = np.nan
        boundary_edge_density.append(np.nanmean(edge_boundary))

        edge_fg = edges.copy().astype("float64")
        edge_fg[label == 0] = np.nan
        fore_edge_density.append(np.nanmean(edge_fg))

        edge_bg = edges.copy().astype("float64")
        edge_bg[label > 0] = np.nan
        back_edge_density.append(np.nanmean(edge_bg))

    return boundary_edge_density, fore_edge_density, back_edge_density

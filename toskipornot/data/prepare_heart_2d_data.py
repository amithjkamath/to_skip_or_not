import os
from glob import glob

import numpy as np
import SimpleITK as sitk
from skimage.io import imsave
from skimage.transform import resize
from skimage.exposure import rescale_intensity


def prepare_data(source_dir, dest_dir):
    """
    This is only for the Task09_Spleen data set
    """

    for fold in ["train", "valid", "test"]:
        os.makedirs(os.path.join(dest_dir, fold, "images"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, fold, "labels"), exist_ok=True)

    all_imgs = sorted(glob(os.path.join(source_dir, "imagesTr", "*.nii.gz")))
    all_labels = sorted(glob(os.path.join(source_dir, "labelsTr", "*.nii.gz")))

    train_imgs = all_imgs[:13]
    train_labels = all_labels[:13]

    train_idx = 0
    for idx in range(len(train_imgs)):
        train_img = train_imgs[idx]
        print("Looking at image: ", train_img)
        img = sitk.GetArrayFromImage(sitk.ReadImage(train_img))
        train_label = train_labels[idx]
        label = sitk.GetArrayFromImage(sitk.ReadImage(train_label))
        for slice_idx in range(label.shape[0]):
            img_slice = np.squeeze(img[slice_idx, :, :])
            label_slice = np.squeeze(label[slice_idx, :, :])
            if np.sum(label_slice) / np.prod(label_slice.shape) > 0.005:
                image_data = resize(img_slice, (256, 256), anti_aliasing=True)
                image_data = rescale_intensity(
                    image_data, in_range="image", out_range="uint8"
                )
                image_data = np.fliplr(image_data)
                image_data = np.flipud(image_data)
                label_data = resize(label_slice, (256, 256), anti_aliasing=False)
                label_data = np.fliplr(label_data)
                label_data = np.flipud(label_data)
                imsave(
                    os.path.join(
                        dest_dir,
                        "train",
                        "images",
                        "image_" + str(train_idx).zfill(4) + ".png",
                    ),
                    image_data,
                )
                imsave(
                    os.path.join(
                        dest_dir,
                        "train",
                        "labels",
                        "label_" + str(train_idx).zfill(4) + ".png",
                    ),
                    np.multiply(label_data, 255),
                )
                train_idx += 1

    valid_imgs = all_imgs[13:17]
    valid_labels = all_labels[13:17]

    valid_idx = 0
    for idx in range(len(valid_imgs)):
        valid_img = valid_imgs[idx]
        print("Looking at image: ", valid_img)
        img = sitk.GetArrayFromImage(sitk.ReadImage(valid_img))
        valid_label = valid_labels[idx]
        label = sitk.GetArrayFromImage(sitk.ReadImage(valid_label))
        for slice_idx in range(label.shape[0]):
            img_slice = np.squeeze(img[slice_idx, :, :])
            label_slice = np.squeeze(label[slice_idx, :, :])
            if np.sum(label_slice) / np.prod(label_slice.shape) > 0.005:
                image_data = resize(img_slice, (256, 256), anti_aliasing=True)
                image_data = rescale_intensity(
                    image_data, in_range="image", out_range="uint8"
                )
                image_data = np.fliplr(image_data)
                image_data = np.flipud(image_data)
                label_data = resize(label_slice, (256, 256), anti_aliasing=False)
                label_data = np.fliplr(label_data)
                label_data = np.flipud(label_data)
                imsave(
                    os.path.join(
                        dest_dir,
                        "valid",
                        "images",
                        "image_" + str(valid_idx).zfill(4) + ".png",
                    ),
                    image_data,
                )
                imsave(
                    os.path.join(
                        dest_dir,
                        "valid",
                        "labels",
                        "label_" + str(valid_idx).zfill(4) + ".png",
                    ),
                    np.multiply(label_data, 255),
                )
                valid_idx += 1

    test_imgs = all_imgs[17:]
    test_labels = all_labels[17:]

    test_idx = 0
    for idx in range(len(test_imgs)):
        test_img = test_imgs[idx]
        print("Looking at image: ", test_img)
        img = sitk.GetArrayFromImage(sitk.ReadImage(test_img))
        test_label = test_labels[idx]
        label = sitk.GetArrayFromImage(sitk.ReadImage(test_label))
        for slice_idx in range(label.shape[0]):
            img_slice = np.squeeze(img[slice_idx, :, :])
            label_slice = np.squeeze(label[slice_idx, :, :])
            if np.sum(label_slice) / np.prod(label_slice.shape) > 0.005:
                image_data = resize(img_slice, (256, 256), anti_aliasing=True)
                image_data = rescale_intensity(
                    image_data, in_range="image", out_range="uint8"
                )
                image_data = np.fliplr(image_data)
                image_data = np.flipud(image_data)
                label_data = resize(label_slice, (256, 256), anti_aliasing=False)
                label_data = np.fliplr(label_data)
                label_data = np.flipud(label_data)
                imsave(
                    os.path.join(
                        dest_dir,
                        "test",
                        "images",
                        "image_" + str(test_idx).zfill(4) + ".png",
                    ),
                    image_data,
                )
                imsave(
                    os.path.join(
                        dest_dir,
                        "test",
                        "labels",
                        "label_" + str(test_idx).zfill(4) + ".png",
                    ),
                    np.multiply(label_data, 255),
                )
                test_idx += 1


if __name__ == "__main__":
    root_dir = "/home/akamath/Documents/toskipornot/data"
    source_dir = os.path.join(root_dir, "Task02_Heart")
    destination_dir = os.path.join(root_dir, "Task02_Heart-processed")
    prepare_data(source_dir, destination_dir)

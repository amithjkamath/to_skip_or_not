import os
from glob import glob

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2gray


def convert_images(folder_name, source_dir, dest_dir, n_images, out_idx):
    for idx in range(1, n_images):
        image_name = (
            source_dir + "/" + folder_name + "/" + folder_name + " (" + str(idx) + ")"
        )
        image_data = imread(image_name + ".png")
        image = resize(image_data, (256, 256), anti_aliasing=True)
        image = rgb2gray(image)
        imsave(dest_dir + "/image/" + "image_" + str(out_idx).zfill(4) + ".png", image)

        label = np.zeros((256, 256), dtype=bool)
        label_masks = sorted(glob(image_name + "_mask*"))
        for label_mask in label_masks:
            this_label_data = imread(label_mask)
            this_label_data = np.array(this_label_data, dtype=bool)

            # Handle situations where label image has more than one channel
            if len(this_label_data.shape) > 2:
                this_label_data = this_label_data[:, :, 0]
            this_label_data = resize(this_label_data, (256, 256), anti_aliasing=False)

            label = np.logical_or(label, this_label_data)
        imsave(dest_dir + "/label/" + "label_" + str(out_idx).zfill(4) + ".png", label)
        out_idx += 1


def prepare_data(source_dir, dest_dir):
    """
    This is only for the BUSI data set - with subfolders for
    benign, malignant, and normal
    """
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "label"), exist_ok=True)

    # benign
    convert_images("benign", source_dir, dest_dir, 438, 1)
    # malignant
    convert_images("malignant", source_dir, dest_dir, 211, 438)
    # normal
    # convert_images("normal", source_dir, dest_dir, 134, 438 + 211)


if __name__ == "__main__":
    root_dir = "/home/akamath/Documents/toskipornot/data/"
    source_dir = os.path.join(root_dir, "BUSI")
    destination_dir = os.path.join(root_dir, "BUSI-processed-test")
    prepare_data(source_dir, destination_dir)

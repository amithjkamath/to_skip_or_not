import os
from glob import glob

import random
from PIL import Image

import numpy as np
from skimage.io import imread, imsave


def convert_images(image_set, source_dir, dest_dir, dest_name, n_images, out_idx=1):
    dest_dir = os.path.join(dest_dir, dest_name)
    os.makedirs(os.path.join(dest_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "label"), exist_ok=True)

    for idx in range(1, n_images + 1):
        image_path = os.path.join(source_dir, image_set + "_" + str(idx) + ".bmp")
        label_path = os.path.join(source_dir, image_set + "_" + str(idx) + "_anno.bmp")
        img = Image.open(image_path).convert("L")  # also converts to grayscale.
        img_data = np.array(img)

        label_data = imread(label_path)
        label_data = np.array(label_data, dtype=bool)
        w, h = img.size
        th, tw = 256, 256

        for _ in range(4):  # generate 4 images from one slide.
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            img_crop = img_data[y1 : y1 + th, x1 : x1 + tw]
            label_crop = label_data[y1 : y1 + th, x1 : x1 + tw]

            Image.fromarray(img_crop).save(
                dest_dir + "/image/" + "image_" + str(out_idx).zfill(4) + ".png"
            )

            Image.fromarray(label_crop).save(
                dest_dir + "/label/" + "label_" + str(out_idx).zfill(4) + ".png"
            )
            out_idx += 1
    return out_idx


def prepare_data(source_dir, dest_dir):
    """
    This is only for the BUSI data set - with subfolders for
    benign, malignant, and normal
    """
    os.makedirs(dest_dir, exist_ok=True)
    out_idx = convert_images("train", source_dir, dest_dir, "train", 85)
    convert_images("testA", source_dir, dest_dir, "valid", 60)
    convert_images("testB", source_dir, dest_dir, "test", 20)


if __name__ == "__main__":
    root_dir = "/home/akamath/Documents/toskipornot/data/"
    source_dir = os.path.join(root_dir, "GLaS-raw")
    destination_dir = os.path.join(root_dir, "GLaS-processed")
    prepare_data(source_dir, destination_dir)

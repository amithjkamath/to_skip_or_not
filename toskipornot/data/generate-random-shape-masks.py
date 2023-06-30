"""
generate-random_shape-masks creates random shape mask images.
"""

import os

# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from toskipornot.features.build_features import get_random_points, get_bezier_curve


def generate_random_mask(im_size, rad, edgy, n_pts, scale):
    """
    GENERATE_RANDOM_MASK generates a 2D image img of size im_size,
    with a random mask defined by rad and edgy.
    """
    img = np.zeros(shape=im_size, dtype=np.uint8)
    a = get_random_points(n=n_pts, scale=scale, mindst=0.001) + [0.25, 0.25]
    x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
    x *= im_size[0]
    y *= im_size[1]
    for x_id, y_id in zip(np.uint32(x), np.uint32(y)):
        x_id = np.clip(x_id, 0, im_size[0] - 1)
        y_id = np.clip(y_id, 0, im_size[1] - 1)
        img[x_id, y_id] = np.uint8(1)

    kernel = np.ones((9, 9), np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    contours = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(
        dilated_img, contours[0], -1, color=(255, 255, 255), thickness=cv2.FILLED
    )
    return dilated_img


if __name__ == "__main__":
    root_path = "/home/akamath/Documents/toskipornot"
    data_path = os.path.join(root_path, "data")
    im_size = [256, 256]
    n_images = 100
    out_folder_name = "raw"

    # Hardest

    output_path = os.path.join(data_path, out_folder_name, "masks_hardest")
    os.makedirs(output_path, exist_ok=True)
    rad = 1.25
    edgy = 0.5
    n_pts = 20
    scale = 0.5

    for i in range(n_images):
        mask = generate_random_mask(im_size, rad, edgy, n_pts, scale)
        im = Image.fromarray(mask)
        file_path = os.path.join(
            output_path,
            "mask_" + str(i).zfill(3) + ".png",
        )
        im.save(file_path)

    # Harder

    output_path = os.path.join(data_path, out_folder_name, "masks_harder")
    os.makedirs(output_path, exist_ok=True)
    rad = 1.25
    edgy = 0.75
    n_pts = 7
    scale = 0.5

    for i in range(n_images):
        mask = generate_random_mask(im_size, rad, edgy, n_pts, scale)
        im = Image.fromarray(mask)
        file_path = os.path.join(
            output_path,
            "mask_" + str(i).zfill(3) + ".png",
        )
        im.save(file_path)

    # Normal

    output_path = os.path.join(data_path, out_folder_name, "masks_normal")
    os.makedirs(output_path, exist_ok=True)
    rad = 1.0
    edgy = 0.05
    n_pts = 4
    scale = 0.5

    for i in range(n_images):
        mask = generate_random_mask(im_size, rad, edgy, n_pts, scale)
        im = Image.fromarray(mask)
        file_path = os.path.join(
            output_path,
            "mask_" + str(i).zfill(3) + ".png",
        )
        im.save(file_path)

    # Easier

    output_path = os.path.join(data_path, out_folder_name, "masks_easier")
    os.makedirs(output_path, exist_ok=True)
    rad = 0.75
    edgy = 0.0
    n_pts = 3
    scale = 0.5

    for i in range(n_images):
        mask = generate_random_mask(im_size, rad, edgy, n_pts, scale)
        im = Image.fromarray(mask)
        file_path = os.path.join(
            output_path,
            "mask_" + str(i).zfill(3) + ".png",
        )
        im.save(file_path)

    # Easier

    output_path = os.path.join(data_path, out_folder_name, "masks_easiest")
    os.makedirs(output_path, exist_ok=True)
    rad = 0.75
    edgy = 0.0
    n_pts = 4
    scale = 0.5

    for i in range(n_images):
        mask = generate_random_mask(im_size, rad, edgy, n_pts, scale)
        im = Image.fromarray(mask)
        file_path = os.path.join(
            output_path,
            "mask_" + str(i).zfill(3) + ".png",
        )
        im.save(file_path)

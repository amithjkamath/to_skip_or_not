import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from skimage.filters import gaussian
from skimage.util import random_noise

from toskipornot.features.analyze_features import kullback_leibler_divergence


def compute_and_plot_histogram(image_data, label_data):
    n_images = len(image_data)

    texture_similarity = []
    for image_index in tqdm(range(n_images)):
        image = imread(image_data[image_index])
        label = imread(label_data[image_index])

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
        texture_similarity.append(score)

    fig, ax = plt.subplots()

    plot = sns.kdeplot(data=texture_similarity)
    plt.xlabel("Texture similarity (larger = more fg/bg difference)")
    plt.ylabel("Count")
    plt.xscale("log")
    plt.grid()
    plt.show()
    return texture_similarity


def make_blurry_background_image(image_path, label_path, sigma):
    image = imread(image_path)
    label = imread(label_path)

    mod_image = image.copy()

    # apply Gaussian blur, creating a new image
    blurred_image = gaussian(image, sigma=(sigma, sigma), channel_axis=2)
    scaled_blurred_image = np.multiply(blurred_image, 255).astype("uint8")

    invert_idx = label == 0
    mod_image[invert_idx] = scaled_blurred_image[invert_idx]
    return mod_image, label


def make_speckle_image(image_path, label_path, speckle_var):
    image = imread(image_path)
    label = imread(label_path)

    mod_image = image.copy()

    noisy_image = random_noise(image, mode="speckle", var=speckle_var)
    scaled_noisy_image = np.multiply(noisy_image, 255).astype("uint8")
    mod_image = scaled_noisy_image
    return mod_image, label

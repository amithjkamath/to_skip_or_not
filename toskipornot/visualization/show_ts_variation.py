import os
from glob import glob
import numpy as np
import torchio.transforms as tiotfm
import monai.transforms as tfm
import skimage.io as skio
import matplotlib.pyplot as plt
import seaborn as sns


from toskipornot.features.analyze_features import *


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_dir, "..", "..", "data", "test-spleen")

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    images_path = os.path.join(data_path, "images")
    images = sorted(glob(os.path.join(images_path, "*.png")))

    labels_path = os.path.join(data_path, "labels")
    labels = sorted(glob(os.path.join(labels_path, "*.png")))

    distance_scores = run_lbp_analysis(images, labels)

    hist, bins = np.histogram(distance_scores, bins=10)
    plot = sns.kdeplot(data=distance_scores, linestyle="--", color="black")

    edit_generators = [
                       #tiotfm.RandomBiasField(coefficients=0.25, order=3),
                       tiotfm.RandomNoise(mean=0.0, std=(0.0, 0.25)),
                       tiotfm.RandomGhosting(num_ghosts=(4, 10), axes=(0, 1), intensity=(0.0, 1.0)),
                       #tfm.GibbsNoise(alpha=0.7),
                       tfm.RandGaussianNoise(prob=1.0, mean=0.0, std=0.25),
                       tfm.AdjustContrast(gamma=2),
                       #tfm.GaussianSharpen(),
                       ]
    variants = []
    normalize_for_proc = tiotfm.RescaleIntensity(out_min_max=(0, 1))
    normalize_to_disp = tiotfm.RescaleIntensity(out_min_max=(0, 255))
    for edit_generator in edit_generators:
        os.makedirs(os.path.join(data_path, edit_generator.__class__.__name__), exist_ok=True)
        edited_distance_scores = []
        n_images = len(images)
        print(f"Running {edit_generator.__class__.__name__} ... ")
        for image_index in tqdm(range(n_images)):
            image = skio.imread(images[image_index])
            label = skio.imread(labels[image_index])
            image = image[np.newaxis, :, :, np.newaxis].astype(np.float64)
            image = normalize_for_proc(image).astype(np.float32)
            edited_image = edit_generator(image)
            edited_image = normalize_to_disp(edited_image)
            edited_image = np.squeeze(np.array(edited_image)).astype(np.uint8)
            score = compute_texture_similarity(edited_image, label)
            skio.imsave(os.path.join(data_path, edit_generator.__class__.__name__, f"edited_{image_index}.png"), edited_image.astype(np.uint8))
            edited_distance_scores.append(score)

        hist, bins = np.histogram(edited_distance_scores, bins=50)
        plot = sns.kdeplot(data=edited_distance_scores, ax=ax)
        variants.append(edit_generator.__class__.__name__)

    plt.xlabel("Texture similarity (smaller = more similar fg/bg)")
    plt.ylabel("Density")
    plt.xscale("log")
    plt.grid()
    ax.set_xlim(1e-3, 1e1)
    plt.legend(["Original"] + variants)
    plt.savefig(os.path.join(data_path, "ts_variation.png"))
    plt.close()

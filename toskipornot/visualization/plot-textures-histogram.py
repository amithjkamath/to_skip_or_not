import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

from toskipornot.features.analyze_features import *


if __name__ == "__main__":
    root_dir = "/Users/amithkamath/repo/toskipornot"
    data_path = os.path.join(root_dir, "data", "foreground-processed")

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    for blend_ratio in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        blend_path = os.path.join(
            data_path, "alphablend_0p" + str(blend_ratio) + "_normal"
        )
        images_path = os.path.join(blend_path, "train")
        images = sorted(glob(os.path.join(images_path, "*.png")))

        labels_path = os.path.join(blend_path, "mask")
        labels = sorted(glob(os.path.join(labels_path, "*.png")))

        distance_scores = run_lbp_analysis(images, labels)

        hist, bins = np.histogram(distance_scores, bins=10)
        # histogram on log scale.
        # Use non-equal bin sizes, such that they look equal on log scale.
        # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        # plot = sns.histplot(
        #    distance_scores,
        #    kde=True,
        #    bins=logbins,
        #    ax=ax,
        #    alpha=0.0,
        # )
        plot = sns.kdeplot(data=distance_scores)
    cm = sns.color_palette("RdYlGn", 10)
    c_idx = 9
    for line in ax.lines:
        line.set_color(cm[c_idx])
        c_idx -= 1
    plt.xlabel("Texture similarity (smaller = more similar fg/bg)")
    plt.ylabel("Density")
    plt.xscale("log")
    plt.grid()
    ax.set_xlim(5e-3, 1e0)
    ax.legend(
        ("0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"),
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
    )
    plt.savefig(os.path.join(data_path, "texture_hist.png"))
    plt.close()

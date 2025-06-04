import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def analyze_synthetic_performance():
    current_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(current_path, "..", "results")

    report_path = os.path.join(results_path, "synthetic-results")
    for_10_only = True

    if for_10_only:
        agunet_noskip_data = pd.read_csv(
            os.path.join(report_path, "agunet-noskipunet-ratios-10-only.csv"),
            header=None,
        )
        unet_noskip_data = pd.read_csv(
            os.path.join(report_path, "unet-noskipunet-ratios-10-only.csv"), header=None
        )
    else:
        agunet_noskip_data = pd.read_csv(
            os.path.join(report_path, "agunet-noskipunet-ratios.csv"), header=None
        )
        unet_noskip_data = pd.read_csv(
            os.path.join(report_path, "unet-noskipunet-ratios.csv"), header=None
        )

    agunet_noskip = pd.Series(
        np.diag(agunet_noskip_data),
        index=[agunet_noskip_data.index, agunet_noskip_data.columns],
    )
    agunet_noskip = agunet_noskip.iloc[::-1]

    unet_noskip = pd.Series(
        np.diag(unet_noskip_data),
        index=[unet_noskip_data.index, unet_noskip_data.columns],
    )
    unet_noskip = unet_noskip.iloc[::-1]

    if for_10_only:
        blend_ratios = [90, 80, 70, 60, 50, 40, 30, 20, 10]
    else:
        blend_ratios = [98, 95, 92, 90, 88, 85, 82, 80, 70, 60, 50, 40, 30, 20, 10]

    # blend_ratios = blend_ratios[::-1]
    blend_ratio_rev = [(100 - i) / 100 for i in blend_ratios]
    plt.figure()
    plt.plot(blend_ratio_rev, agunet_noskip)
    plt.plot(blend_ratio_rev, unet_noskip)

    if for_10_only:
        plt.plot(blend_ratio_rev, [0] * 9)  # for NoSkipUNet
    else:
        plt.plot(blend_ratio_rev, [0] * 15)  # for NoSkipUNet
    # plt.xscale("log")
    plt.grid()
    plt.legend(("AG-UNet", "UNet", "NoSkip-UNet"))
    plt.xlabel("Proportion of foreground blended")
    plt.ylabel("Relative improvement in Dice versus NoSkip-UNet")
    plt.xlim(1e-2, 1)
    plt.show()


if __name__ == "__main__":
    analyze_synthetic_performance()

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

font = {"family": "normal", "weight": "normal", "size": 20}

matplotlib.rc("font", **font)


def plot_robustness(df, cmap="RdYlGn", max_val=1.0, min_val=0.0):
    # plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    fig, ax = plt.subplots()
    fig.set_figwidth(30)
    fig.set_figheight(30)
    sns.heatmap(
        df, annot=True, cmap=cmap, vmax=max_val, vmin=min_val, cbar=False, fmt=".2f"
    )
    ax.set_xticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.xlabel("α tested on")
    plt.ylabel("α trained on")


def plot_relative_robustness(df, cmap="bwr", max_val=1.0, min_val=0.0):
    # plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    fig, ax = plt.subplots()
    fig.set_figwidth(30)
    fig.set_figheight(30)
    sns.heatmap(
        df, annot=True, cmap=cmap, vmax=max_val, vmin=min_val, cbar=False, fmt=".2f"
    )
    ax.set_xticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.xlabel("α tested on")
    plt.ylabel("α trained on")


def analyze_synthetic_robustness(metric="dice"):
    current_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(current_path, "..", "..", "results")

    direction = "back"  # "back" or "fore"
    report_path = os.path.join(results_path, direction + "ground-results")

    agunet_data = pd.read_csv(
        os.path.join(report_path, metric + "_for_AttentionUNet.csv"), index_col=0
    )
    unetplusplus_data = pd.read_csv(
        os.path.join(report_path, metric + "_for_UNet++.csv"), index_col=0
    )
    vnet_data = pd.read_csv(
        os.path.join(report_path, metric + "_for_vnet.csv"), index_col=0
    )
    unet_data = pd.read_csv(
        os.path.join(report_path, metric + "_for_unet.csv"), index_col=0
    )
    noskipunet_data = pd.read_csv(
        os.path.join(report_path, metric + "_for_NoSkipUNet.csv"), index_col=0
    )
    noskipvnet_data = pd.read_csv(
        os.path.join(report_path, metric + "_for_NoSkipVNet.csv"), index_col=0
    )

    df_list = {
        "UNet++": unetplusplus_data,
        "AG-UNet": agunet_data,
        "VNet": vnet_data,
        "UNet": unet_data,
        "NoSkipUNet": noskipunet_data,
        "NoSkipVNet": noskipvnet_data,
    }

    model_alias = {
        "UNet++": "UNet++",
        "AG-UNet": "AGU-Net",
        "UNet": "U-Net",
        "VNet": "V-Net",
        "NoSkipUNet": "NoSkipU-Net",
        "NoSkipVNet": "NoSkipV-Net",
    }

    for net_name in df_list.keys():
        if metric in ["hd", "surfdist"]:
            max_val = 150.0
            min_val = 0.0
            in_cmap = "RdYlGn_r"
        else:
            max_val = 1.0
            min_val = 0.0
            in_cmap = "RdYlGn"
        plot_robustness(
            df_list[net_name], cmap=in_cmap, max_val=max_val, min_val=min_val
        )
        plt.title(model_alias[net_name])
        plt.savefig(
            os.path.join(
                results_path,
                "synthetic-robustness-" + net_name + "-" + metric + ".png",
            ),
            bbox_inches="tight",
        )
        plt.close()

    baseline_data = noskipunet_data

    for net_name in df_list.keys():
        relative_df = df_list[net_name] - baseline_data
        relative_df.divide(baseline_data, fill_value=0.0)
        if metric in ["hd", "surfdist"]:
            max_val = 100.0
            min_val = -100.0
            in_cmap = "bwr_r"
        else:
            max_val = 1.0
            min_val = -1.0
            in_cmap = "bwr"
        plot_relative_robustness(
            relative_df, cmap=in_cmap, max_val=max_val, min_val=min_val
        )
        plt.title(model_alias[net_name] + " - relative to NoSkipU-Net")
        plt.savefig(
            os.path.join(
                results_path,
                "synthetic-robustness-relative-to-NoSkipU-Net-"
                + net_name
                + "-"
                + metric
                + ".png",
            ),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    for metric in ["dice", "hd", "sdsc", "surfdist"]:
        analyze_synthetic_robustness(metric)

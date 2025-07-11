import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_clinical_robustness(metric_name="Dice"):
    current_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(current_path, "..", "..", "results")

    model_list = ["UNet++", "AttentionUNet", "UNet", "VNet", "NoSkipUNet", "NoSkipVNet"]
    model_alias = {
        "UNet++": "UNet++",
        "AttentionUNet": "AGU-Net",
        "UNet": "U-Net",
        "VNet": "V-Net",
        "NoSkipUNet": "NoSkipU-Net",
        "NoSkipVNet": "NoSkipV-Net",
    }

    anatomy_list = ["BUSI", "GLaS", "Heart", "Spleen"]
    anatomy_alias = {
        "BUSI": "Breast \n (Ultrasound)",
        "GLaS": "Colon \n (Histology)",
        "Heart": "Heart \n (MRI)",
        "Spleen": "Spleen \n (CT)",
    }

    variant_list = ["lower", "low", "in-domain", "high", "higher"]
    variant_alias = {
        "lower": "Hardest",
        "low": "Harder",
        "in-domain": "Unperturbed",
        "high": "Easier",
        "higher": "Easiest",
    }

    mean_data = {}
    std_data = {}

    for texture_variant in variant_list:
        metric_mean = pd.DataFrame(columns=model_list, index=anatomy_list)

        metric_sd = pd.DataFrame(columns=model_list, index=anatomy_list)

        for anatomy in anatomy_list:
            for model_name in model_list:
                result_seed_1 = os.path.join(
                    results_path,
                    anatomy + "-results",
                    anatomy
                    + "_stats_"
                    + model_name
                    + "_"
                    + texture_variant
                    + "_256_1.csv",
                )
                result_seed_2 = os.path.join(
                    results_path,
                    anatomy + "-results",
                    anatomy
                    + "_stats_"
                    + model_name
                    + "_"
                    + texture_variant
                    + "_256_2.csv",
                )
                result_seed_3 = os.path.join(
                    results_path,
                    anatomy + "-results",
                    anatomy
                    + "_stats_"
                    + model_name
                    + "_"
                    + texture_variant
                    + "_256_3.csv",
                )

                seed_1_data = pd.read_csv(result_seed_1)
                seed_1_data = seed_1_data.loc[seed_1_data["Set"] == "test"]

                seed_2_data = pd.read_csv(result_seed_2)
                seed_2_data = seed_2_data.loc[seed_2_data["Set"] == "test"]

                seed_3_data = pd.read_csv(result_seed_3)
                seed_3_data = seed_3_data.loc[seed_3_data["Set"] == "test"]

                scores = np.hstack(
                    (
                        seed_1_data[metric_name].to_numpy(),
                        seed_2_data[metric_name].to_numpy(),
                        seed_3_data[metric_name].to_numpy(),
                    )
                )

                scores[~np.isfinite(scores)] = np.sqrt(2) * 256

                metric_mean[model_name][anatomy] = np.mean(scores)
                metric_sd[model_name][anatomy] = np.std(scores)

        mean_data[texture_variant] = metric_mean
        std_data[texture_variant] = metric_sd

        print("For texture variant: " + str(texture_variant) + " mean scores;")
        print(metric_mean)
        print("For texture variant: " + str(texture_variant) + " std scores;")
        print(metric_sd)

    # Plotting the results
    for anatomy in anatomy_list:
        mean_data_dict = {
            'x': [], # mean delta dice, for each of the four levels of perturbation
            'y': [], # dice for the in-domain images.
            'category': [] # model name.
        }
        for model_name in model_list:
            # First, aggregate for perturbation variants
            mean_list = []
            std_list = []
            for texture_variant in ["lower", "low", "high", "higher"]:
                mean_list.append(
                    mean_data[texture_variant][model_name][anatomy]
                )
                std_list.append(std_data[texture_variant][model_name][anatomy])
            mu = np.mean(mean_list)
            sigma = np.std(mean_list)

            perf_mean = mean_data["in-domain"][model_name][anatomy]
            perf_std = std_data["in-domain"][model_name][anatomy]

            mean_data_dict["x"].append(mu - perf_mean)
            mean_data_dict["y"].append(perf_mean)
            mean_data_dict["category"].append(model_alias[model_name])

        sns.scatterplot(data=mean_data_dict, x='x', y='y', hue='category', style='category')
        plt.title(
            "Clinical Robustness: "
            + metric_name
            + " for "
            + anatomy_alias[anatomy]
            )
        plt.xlabel(
            "Mean Delta "
            + metric_name
            + " (perturbed - in-domain) for "
            + anatomy_alias[anatomy]
        )
        plt.ylabel(
            "Mean "
            + metric_name
            + " for "
            + anatomy_alias[anatomy]
        )
        plt.grid(True)
        if metric_name == "Dice":
            plt.ylim([0.0, 1.0])
            plt.xlim([-1.0, 0.0])

        plt.savefig(
            os.path.join(
                results_path,
                "clinical-robustness-" + metric_name + "_" + anatomy_alias[anatomy] + ".png",
            ),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    for metric_name in [
        "Dice",
        "SurfaceDSC",
        "HD",
        "SurfaceDistance",
    ]:
        analyze_clinical_robustness(metric_name)

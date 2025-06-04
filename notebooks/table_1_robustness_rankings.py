import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def results_by_metric():
    current_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(current_path, "..", "reports")

    metric_list = ["Dice", "HD", "SurfaceDSC", "SurfaceDistance"]

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

    # %%
    for metric in metric_list:
        for anatomy in anatomy_list:
            for texture_variant in variant_list:
                metric_mean = pd.DataFrame(columns=model_list)
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

                    per_model_mean_scores = np.mean(
                        np.vstack(
                            (
                                seed_1_data[metric].to_numpy(),
                                seed_2_data[metric].to_numpy(),
                                seed_3_data[metric].to_numpy(),
                            )
                        ).T,
                        axis=1,
                    )
                    per_model_mean_scores[~np.isfinite(per_model_mean_scores)] = (
                        np.sqrt(2) * 256
                    )

                    metric_mean[model_name] = per_model_mean_scores

                if metric in ["Dice", "SurfaceDSC"]:
                    best_model = metric_mean.idxmax(axis=1, skipna=True)
                else:
                    best_model = metric_mean.idxmin(axis=1, skipna=True)
                # plt.figure()
                # sns.histplot(best_model)
                # plt.title("For " + metric + " in data: " + anatomy_alias[anatomy] + " and texture level: " + variant_alias[texture_variant])
                print(
                    "For "
                    + metric
                    + " in data: "
                    + anatomy_alias[anatomy]
                    + " and texture level: "
                    + variant_alias[texture_variant]
                )
                print(best_model.value_counts().to_frame() / len(metric_mean))


if __name__ == "__main__":
    results_by_metric()

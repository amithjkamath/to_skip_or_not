import os
from datetime import timedelta
import numpy as np
import pandas as pd

import warnings
# Make the repository importable when this file is run as a plain script.
import os as _os, sys as _sys
_sys.path.insert(
    0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
)
from toskipornot.config import RESULTS_DIR  # noqa: E402

warnings.filterwarnings("ignore")


def train_time_metrics():
    current_path = os.path.dirname(os.path.abspath(__file__))
    results_path = str(RESULTS_DIR)

    report_path = os.path.join(results_path, "train-time-results")

    model_names = [
        "UNet++",
        "AttentionUNet",
        "UNet",
        "VNet",
        "NoSkipUNet",
        "NoSkipVNet",
    ]
    dataset_names = ["busi", "glas", "heart", "spleen"]

    # %%
    num_epochs = 100

    df = pd.DataFrame(columns=model_names, index=dataset_names)
    for ds_idx in range(len(dataset_names)):
        for model_idx in range(len(model_names)):
            wandb_file = os.path.join(
                report_path,
                dataset_names[ds_idx] + "-" + model_names[model_idx] + "-full-v3.csv",
            )
            run_data = pd.read_csv(wandb_file)
            run_data = run_data[run_data["name"].str.contains("256")]

            mean_time_per_epoch = np.mean(
                [x / num_epochs for x in run_data["_runtime"].tolist()]
            )
            std_dev_per_epoch = np.std(
                [x / num_epochs for x in run_data["_runtime"].tolist()]
            )
            df.iloc[ds_idx, model_idx] = (
                str(timedelta(seconds=mean_time_per_epoch))[:-4]
                + "("
                + str(timedelta(seconds=std_dev_per_epoch))[:-4]
                + ")"
            )
            df.iloc[ds_idx, model_idx] = (
                str(mean_time_per_epoch) + "(" + str(std_dev_per_epoch) + ")"
            )

    print(df)


if __name__ == "__main__":
    train_time_metrics()

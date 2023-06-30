import pandas as pd
import wandb


def generate_csv_results(user_name, project_name):
    """
    generate_csv_results: uses the wandb API to save
    results from experiments.
    """
    # Project is specified by <entity/project-name>
    api = wandb.Api()
    runs = api.runs(user_name + "/" + project_name)
    summary_list = []
    config_list = []
    name_list = []
    for run in runs:
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        config_list.append(config)

        # run.name is the name of the run.
        name_list.append(run.name)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({"name": name_list})
    all_df = pd.concat([name_df, config_df, summary_df], axis=1)

    all_df.to_csv(project_name + ".csv")


if __name__ == "__main__":
    user_name = "amithjkamath"
    project_name = "spleen-AttentionUNet-full-v3"
    generate_csv_results(user_name, project_name)

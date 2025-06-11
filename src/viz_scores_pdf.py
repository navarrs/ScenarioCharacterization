import os
import pickle  # nosec B403

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from torch.utils.data import Dataset

from scorer import SUPPORTED_SCORERS
from utils.common import get_logger
from utils.viz.visualizer import BaseVisualizer

logger = get_logger(__name__)


def get_sample_to_plot(
    df: pd.DataFrame,
    key: str,
    min_value: float,
    max_value: float,
    seed: int,
    sample_size: int,
) -> pd.DataFrame:
    df_subset = df[(df[key] >= min_value) & (df[key] < max_value)]
    subset_size = len(df_subset)
    logger.info(f"Found {subset_size} rows between [{round(min_value, 2)} to {round(max_value, 2)}] for {key}")
    sample_size = min(sample_size, subset_size)
    return df_subset.sample(n=sample_size, random_state=seed)


def plot_histograms_from_dataframe(df, output_filepath: str = "temp.png", dpi: int = 30, alpha=0.5):
    """
    Plots overlapping histograms for each numeric column in a DataFrame,
    excluding a specified column.

    Parameters:
    - df: pandas DataFrame containing numeric data.
    - exclude_column: column name to exclude from plotting.
    - bins: number of histogram bins.
    - alpha: transparency level for the histograms (0 = transparent, 1 = solid).
    """
    # Select numeric columns, excluding the specified one
    columns_to_plot = df.select_dtypes(include="number").columns
    N = len(columns_to_plot)

    if N == 0:
        logger.error("No numeric columns to plot after excluding the specified column.")
        return

    palette = sns.color_palette("husl", N)

    plt.figure(figsize=(10, 6))

    for i, col in enumerate(columns_to_plot):
        sns.histplot(
            df[col],
            color=palette[i],
            label=col,
            kde=True,
            stat="density",
            alpha=alpha,
            edgecolor="white",
        )

    sns.despine(top=True, right=True)

    plt.legend()
    plt.xlabel("Scores")
    plt.ylabel("Density")
    plt.title("Score Density Function over Scenarios")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=dpi)
    plt.close()


@hydra.main(config_path="config", config_name="viz_scores_pdf", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """
    Run the processor with the given configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.
    """
    # TODO: Adapt to different scoring criteria types (e.g, GT, Critical-vs-Average, etc.)
    seed = cfg.get("seed", 42)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Verify scorer type is supported
    unsupported_scorers = [scorer for scorer in cfg.scorers if scorer not in SUPPORTED_SCORERS]
    if unsupported_scorers:
        logger.error(f"Scorers {unsupported_scorers} not in supported list {SUPPORTED_SCORERS}")
        raise ValueError
    else:
        scores: dict = {}  # Initialize with an empty list for scenarios
        for scorer in cfg.scorers:
            scores[scorer] = []

    # Instantiate dataset and visualizer
    cfg.dataset.config.load = False
    logger.info("Instatiating dataset: %s", cfg.dataset._target_)
    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)

    logger.info("Instatiating visualizer: %s", cfg.viz._target_)
    visualizer: BaseVisualizer = hydra.utils.instantiate(cfg.viz, dataset=dataset)

    # Load scores from score path
    scenario_scores = [os.path.join(cfg.scores_path, f) for f in os.listdir(cfg.scores_path)]
    scores["scenario_ids"] = [f for f in os.listdir(cfg.scores_path) if f.endswith(".pkl")]

    # Generate score histogram and density plot
    logger.info(f"Visualizing density function for scorers: {cfg.scorers}")
    for scenario in scenario_scores:
        with open(scenario, "rb") as f:
            scenario_scores = pickle.load(f)  # nosec B301

        for scorer in cfg.scorers:
            scores[scorer].append(scenario_scores[scorer]["scene_score"])

    scores_df = pd.DataFrame(scores)
    output_filepath = os.path.join(cfg.output_dir, "score_density_plot.png")
    plot_histograms_from_dataframe(scores_df, output_filepath, cfg.dpi)

    # Generate scenario visualizations
    for key in scores_df.keys():
        if "scenario" in key:
            continue
        key_path = os.path.join(cfg.output_dir, key)
        os.makedirs(key_path, exist_ok=True)

        # Visualize a few scenarios across various percentiles
        # Get score percentiles
        percentiles = np.percentile(scores_df[key], cfg.percentiles)
        logger.info(f"Percentiles for {key}: {percentiles}")
        percentiles_low = np.append(scores_df[key].min(), percentiles)
        percentiles_high = np.append(percentiles, scores_df[key].max())
        percentile_ranges = zip(percentiles_low, percentiles_high)

        scenarios_path = os.path.join(key_path, "scenarios")
        os.makedirs(scenarios_path, exist_ok=True)
        for min_value, max_value in percentile_ranges:
            rows = get_sample_to_plot(scores_df, key, min_value, max_value, seed, cfg.min_scenarios_to_plot)
            if rows.empty:
                logger.warning(f"No rows found for {key} in range [{min_value}, {max_value}]")
                continue

            for index, row in rows.iterrows():
                score = row[key]
                scenario_id = row["scenario_ids"].split(".")[0]
                logger.info(f"Processing {scenario_id} for scorer {key}")
                scenario_input_ilepath = os.path.join(cfg.paths.scenario_base_path, f"sample_{scenario_id}.pkl")
                with open(scenario_input_ilepath, "rb") as f:
                    scenario_data = pickle.load(f)  # nosec B301

                scenario_title = f"Scenario Score: {score:.2f}"
                scenario_output_filepath = os.path.join(
                    scenarios_path, f"scenario-id-{scenario_id}_score-{score:.2f}.png"
                )
                visualizer.visualize_scenario(scenario_data, scenario_title, scenario_output_filepath)


if __name__ == "__main__":
    run()

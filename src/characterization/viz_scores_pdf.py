import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from torch.utils.data import Dataset

from characterization.scorer import SUPPORTED_SCORERS
from characterization.utils.common import from_pickle, get_logger
from characterization.utils.schemas import ScenarioScores
from characterization.utils.viz.visualizer import BaseVisualizer

logger = get_logger(__name__)


def get_sample_to_plot(
    df: pd.DataFrame,
    key: str,
    min_value: float,
    max_value: float,
    seed: int,
    sample_size: int,
) -> pd.DataFrame:
    """
    Selects a random sample of rows from a DataFrame within a specified value range for a given column.

    Args:
        df (pd.DataFrame): The DataFrame to sample from.
        key (str): The column name to filter by value range.
        min_value (float): The minimum value (inclusive) for filtering.
        max_value (float): The maximum value (exclusive) for filtering.
        seed (int): Random seed for reproducibility.
        sample_size (int): Number of samples to return.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled rows within the specified range.
    """
    df_subset = df[(df[key] >= min_value) & (df[key] < max_value)]
    subset_size = len(df_subset)
    logger.info(f"Found {subset_size} rows between [{round(min_value, 2)} to {round(max_value, 2)}] for {key}")
    sample_size = min(sample_size, subset_size)
    return df_subset.sample(n=sample_size, random_state=seed)


def plot_histograms_from_dataframe(df, output_filepath: str = "temp.png", dpi: int = 30, alpha=0.5):
    """
    Plots overlapping histograms and density curves for each numeric column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing numeric data to plot.
        output_filepath (str): Path to save the output plot image.
        dpi (int): Dots per inch for the saved figure.
        alpha (float): Transparency level for the histograms (0 = transparent, 1 = solid).

    Raises:
        ValueError: If no numeric columns are found in the DataFrame.
    """
    # Select numeric columns, excluding the specified one
    columns_to_plot = df.select_dtypes(include="number").columns
    N = len(columns_to_plot)

    if N == 0:
        raise ValueError("No numeric columns to plot.")

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
    Runs the scenario score visualization pipeline using the provided configuration.

    This function loads scenario scores, generates density plots for each scoring method, and visualizes example
    scenarios across score percentiles. It supports multiple scoring criteria and flexible dataset/visualizer
    instantiation via Hydra.

    Args:
        cfg (DictConfig): Configuration dictionary specifying dataset, visualizer, scoring methods, paths, and output
            options.

    Raises:
        ValueError: If unsupported scorers are specified in the configuration.
    """
    # TODO: Adapt to different scoring criteria types (e.g, GT, Critical-vs-Average, etc.)
    seed = cfg.get("seed", 42)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Verify scorer type is supported
    unsupported_scores = [scorer for scorer in cfg.scores if scorer not in SUPPORTED_SCORERS]
    if unsupported_scores:
        raise ValueError(f"Scorers {unsupported_scores} not in supported list {SUPPORTED_SCORERS}")
    else:
        scores: dict = {}  # Initialize with an empty list for scenarios
        agent_scores: dict = {}  # Initialize with an empty list for agents
        for scorer in cfg.scores:
            scores[scorer] = []
            agent_scores[scorer] = []

    # Instantiate dataset and visualizer
    cfg.dataset.config.load = False
    logger.info("Instatiating dataset: %s", cfg.dataset._target_)
    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)

    logger.info("Instatiating visualizer: %s", cfg.viz._target_)
    visualizer: BaseVisualizer = hydra.utils.instantiate(cfg.viz)

    # Load scores from score path
    scenario_scores_filepaths = [os.path.join(cfg.scores_path, f) for f in os.listdir(cfg.scores_path)]
    scores["scenario_ids"] = [f for f in os.listdir(cfg.scores_path) if f.endswith(".pkl")]
    agent_scores["scenario_ids"] = scores["scenario_ids"]

    # Generate score histogram and density plot
    logger.info(f"Visualizing density function for scores: {cfg.scores}")
    for scenario_filepath in scenario_scores_filepaths:
        scenario_scores = from_pickle(scenario_filepath)  # nosec B301
        scenario_scores = ScenarioScores.model_validate(scenario_scores)

        for scorer in cfg.scores:
            key = f"{scorer}_scene_score"
            scores[scorer].append(scenario_scores[key])
            key = f"{scorer}_agent_scores"
            agent_scores[scorer].append(scenario_scores[key])

    scores_df = pd.DataFrame(scores)
    output_filepath = os.path.join(cfg.output_dir, "score_density_plot.png")
    plot_histograms_from_dataframe(scores_df, output_filepath, cfg.dpi)

    agent_scores_df = pd.DataFrame(agent_scores)
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
                scenario_id = row["scenario_ids"]
                agent_scores = agent_scores_df[agent_scores_df["scenario_ids"] == scenario_id][key].values[0]
                scenario_id = row["scenario_ids"].split(".")[0]

                logger.info(f"Processing {scenario_id} for scorer {key}")
                scenario_input_filepath = os.path.join(cfg.paths.scenario_base_path, f"sample_{scenario_id}.pkl")

                scenario_data = from_pickle(scenario_input_filepath)  # nosec B301
                scenario = dataset.transform_scenario_data(scenario_data)

                scenario_title = f"Scenario Score: {score:.2f}"
                scenario_output_filepath = os.path.join(
                    scenarios_path, f"scenario-id-{scenario_id}_score-{score:.2f}.png"
                )
                visualizer.visualize_scenario(
                    scenario, scores=agent_scores, title=scenario_title, output_filepath=scenario_output_filepath
                )


if __name__ == "__main__":
    run()

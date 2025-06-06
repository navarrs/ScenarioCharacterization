import os
import pickle

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from scorer import SUPPORTED_SCORERS
from utils.common import get_logger

logger = get_logger(__name__)


def get_sample_to_plot(
    df: pd.DataFrame, key: str, min_value: float, max_value: float, seed: int, sample_size: int
) -> pd.DataFrame:
    df_subset = df[(df[key] >= min_value) & (df[key] < max_value)]
    subset_size = len(df_subset)    
    logger.info(
        f"Found {subset_size} rows between [{round(min_value, 2)} to {round(max_value, 2)}] for {key}")
    sample_size = min(sample_size, subset_size)
    return df_subset.sample(n=sample_size, random_state=seed)

@hydra.main(config_path="config", config_name="viz_scores_pdf", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """
    Run the processor with the given configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.
    """
    # TODO: adapt to multiple score types
    seed = cfg.get("seed", 42)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "scenarios"), exist_ok=True)

    unsupported_scorers = [
        scorer for scorer in cfg.scorers if scorer not in SUPPORTED_SCORERS
    ]
    if unsupported_scorers:
        logger.error(
            f"Scorers {unsupported_scorers} not in supported list {SUPPORTED_SCORERS}"
        )
        raise ValueError
    else:
        scores = {}  # Initialize with an empty list for scenarios
        for scorer in cfg.scorers:
            scores[scorer] = []

    logger.info(f"Visualizing density function for scorers: {cfg.scorers}")

    # Load scores from score path
    scenario_scores = [
        os.path.join(cfg.scores_path, f) for f in os.listdir(cfg.scores_path)
    ]
    scores['scenario_ids'] = [f for f in os.listdir(cfg.scores_path) if f.endswith('.pkl')]

    for scenario in scenario_scores:
        with open(scenario, "rb") as f:
            scenario_scores = pickle.load(f)

        for scorer in cfg.scorers:
            scores[scorer].append(scenario_scores[scorer]["scene_score"])

    # Plot the density functions for each scorer
    scores_df = pd.DataFrame(scores)
    for key in scores_df.keys():
        if "scenario" in key:
            continue
        # Visualize the score densities
        kde_output_path = os.path.join(cfg.output_dir, f"{key}_kde.png")
        sns.histplot(data=scores_df, x=key, kde=True)
        plt.tight_layout()
        plt.savefig(kde_output_path, dpi=cfg.dpi)
        plt.close()
        logger.info(f"Saved KDE plot for {key} to {kde_output_path}")

        # Visualize a few scenarios across various percentiles
        # Get score percentiles
        percentiles = np.percentile(scores_df[key], cfg.percentiles)
        logger.info(f"Percentiles for {key}: {percentiles}")
        percentiles_low = np.append(scores_df[key].min(), percentiles)
        percentiles_high = np.append(percentiles, scores_df[key].max())
        percentile_ranges = zip(percentiles_low, percentiles_high)

        for min_value, max_value in percentile_ranges:
            rows = get_sample_to_plot(
                scores_df, key, min_value, max_value, seed, cfg.min_scenarios_to_plot)
            # TODO: add visualization

if __name__ == "__main__":
    run()

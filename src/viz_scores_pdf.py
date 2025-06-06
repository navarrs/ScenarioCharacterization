import os
import pickle

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from scorer import SUPPORTED_SCORERS
from utils.common import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="config", config_name="viz_scores_pdf", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """
    Run the processor with the given configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.
    """
    # TODO: adapt to multiple score types
    os.makedirs(cfg.output_dir, exist_ok=True)

    unsupported_scorers = [
        scorer for scorer in cfg.scorers if scorer not in SUPPORTED_SCORERS
    ]
    if unsupported_scorers:
        logger.error(
            f"Scorers {unsupported_scorers} not in supported list {SUPPORTED_SCORERS}"
        )
        raise ValueError
    else:
        scores = {}
        for scorer in cfg.scorers:
            scores[scorer] = []

    logger.info(f"Visualizing density function for scorers: {cfg.scorers}")

    # Load scores from score path
    scenario_scores = [
        os.path.join(cfg.scores_path, f) for f in os.listdir(cfg.scores_path)
    ]
    for scenario in scenario_scores:
        with open(scenario, "rb") as f:
            scenario_scores = pickle.load(f)

        for scorer in cfg.scorers:
            scores[scorer].append(scenario_scores[scorer]["scene_score"])

    scores_df = pd.DataFrame(scores)
    for key in scores_df.keys():
        if key == "scenario":
            continue

        # Visualize the score densities
        sns.histplot(data=scores_df, x=key, kde=True)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, "scores.png"), dpi=cfg.dpi)
        plt.close()

        # Plot the KDE
        sns.kdeplot(
            data=scores_df[key],
            label=key,
            linewidth=2,
            bw_adjust=5,
            common_norm=False,
            cut=0,
            fill=True,
            alpha=0.15,
        )
        plt.xlabel("Scores")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            os.path.join(cfg.output_dir, f"{key}_kde.png"), bbox_inches="tight", dpi=300
        )
        plt.close()


if __name__ == "__main__":
    run()

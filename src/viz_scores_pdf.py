import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from utils.logger import get_logger

logger = get_logger(__name__)

@hydra.main(config_path="config", config_name="viz_scores_pdf", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """
    Run the processor with the given configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    scores = {}
    for score_type, score_type_path in cfg.score_types.items():
        # TODO: this is pretty hacky, need to be simplified
        scores_cache = np.load(score_type_path, allow_pickle=True)["scores"]
        if scores.get("scenario") is None:
            scores["scenario"] = [list(s.keys())[0] for s in scores_cache]

        scores[score_type] = [
            float(s[list(s.keys())[0]]["scene_score"]) for s in scores_cache
        ]

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

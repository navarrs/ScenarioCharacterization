import math
import os
import pickle  # nosec B403
from typing import Any

import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from natsort import natsorted
from omegaconf import DictConfig

from characterization.utils.io_utils import get_logger, make_output_paths

logger = get_logger(__name__)


def reset_infos(cfg: DictConfig) -> dict[str, list[Any]]:
    """Reset the information dictionary for a new shard.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        dict: A dictionary with empty lists for each feature and score.
    """
    infos = {"scenario_id": []}
    for feature in cfg.features:
        infos[feature] = []

    for score in cfg.scores:
        agent_scores = f"{score}_agents"
        infos[agent_scores] = []

        scene_scores = f"{score}_scenes"
        infos[scene_scores] = []

    return infos


@hydra.main(config_path="config", config_name="run_converter", version_base="1.3")
def run(cfg: DictConfig) -> None:
    """Run the processor with the given configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.
    """
    make_output_paths(cfg.copy())

    logger.info("Starting the converter...")

    features_path = cfg.input_features_path
    scores_path = cfg.input_scores_path
    features_files = natsorted(os.listdir(features_path))
    scores_files = natsorted(os.listdir(scores_path))

    if set(features_files) != set(scores_files):
        error_message = (
            "Feature files and score files do not match. "
            f"Found {len(features_files)} feature files and {len(scores_files)} score files. "
            "Please ensure that each feature file has a corresponding score file with the same name."
        )
        raise AssertionError(error_message)

    logger.info("Loading input features from: %s", features_path)
    features_files = [os.path.join(features_path, f) for f in features_files]

    logger.info("Loading input scores from: %s", scores_path)
    scores_files = [os.path.join(scores_path, f) for f in scores_files]

    # TODO: Shard scenarios into parquet files

    # Information to parquet
    infos = reset_infos(cfg)
    n_per_shard = math.ceil(len(features_files) / cfg.num_shards)

    logger.info("Creating %d shards with %d scenarios each.", cfg.num_shards, n_per_shard)
    for n, (feature_file, score_file) in enumerate(zip(features_files, scores_files, strict=False)):
        # logger.info(f"Processing {feature_file} and {score_file}")

        infos["scenario_id"].append(feature_file.split("/")[-1].split(".")[0])
        with open(feature_file, "rb") as f:
            features = pickle.load(f)  # nosec B301
        for feature in cfg.features:
            if feature not in features:
                logger.warning("Feature %s not found in dictionary.", feature)
                continue
            infos[feature].append(features[feature])

        with open(score_file, "rb") as f:
            scores = pickle.load(f)  # nosec B301
        for score in cfg.scores:
            if score not in scores:
                logger.warning("Score %s not found in dictionary.", score)
                continue
            agent_scores = f"{score}_agents"
            scene_scores = f"{score}_scenes"
            infos[agent_scores].append(scores[score]["agent_scores"])
            infos[scene_scores].append(scores[score]["scene_score"])

        if ((n + 1) % n_per_shard == 0 and n > 0) or n == len(features_files) - 1:
            # Convert to DataFrame
            df = pd.DataFrame(infos)

            # Convert to Arrow Table
            table = pa.Table.from_pandas(df)

            # Write to Parquet file
            output_path = os.path.join(
                cfg.paths.output_paths.shards_cache_path,
                f"shard_{n // n_per_shard}.parquet",
            )
            pq.write_table(table, output_path)

            # Reset infos for next shard
            num_scenarios = len(infos["scenario_id"])
            infos = reset_infos(cfg)

            logger.info("Wrote shard %d with %d scenarios to: %s", n // n_per_shard, num_scenarios, output_path)


if __name__ == "__main__":
    run()

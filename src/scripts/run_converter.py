import math
import os
import pickle
from typing import AnyStr, Dict

import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from natsort import natsorted
from omegaconf import DictConfig

from src.utils.common import get_logger, make_output_paths

logger = get_logger(__name__)


def reset_infos(cfg: DictConfig) -> dict:
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
    """
    Run the processor with the given configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.
    """
    make_output_paths(cfg.copy())

    logger.info("Starting the converter...")

    features_path = cfg.input_features_path
    scores_path = cfg.input_scores_path
    features_files = natsorted(os.listdir(features_path))
    scores_files = natsorted(os.listdir(scores_path))
    try:
        assert set(features_files) == set(
            scores_files
        ), "Feature files and score files must match."
    except AssertionError as e:
        logger.error(f"Error: {e}")
        return

    logger.info(f"Loading input features from: {features_path}")
    features_files = [os.path.join(features_path, f) for f in features_files]

    logger.info(f"Loading input scores from: {scores_path}")
    scores_files = [os.path.join(scores_path, f) for f in scores_files]

    # TODO: Shard scenarios into parquet files

    # Information to parquet
    infos = reset_infos(cfg)
    n_per_shard = math.ceil(len(features_files) / cfg.num_shards)

    logger.info(f"Creating {cfg.num_shards} shards with {n_per_shard} scenarios each.")
    for n, (feature_file, score_file) in enumerate(zip(features_files, scores_files)):
        # logger.info(f"Processing {feature_file} and {score_file}")

        infos["scenario_id"].append(feature_file.split("/")[-1].split(".")[0])
        with open(feature_file, "rb") as f:
            features = pickle.load(f)
        for feature in cfg.features:
            if feature not in features:
                logger.warning(f"Feature {feature} not found in dictionary.")
                continue
            infos[feature].append(features[feature])

        with open(score_file, "rb") as f:
            scores = pickle.load(f)
        for score in cfg.scores:
            if score not in scores:
                logger.warning(f"Score {score} not found in dictionary.")
                continue
            agent_scores = f"{score}_agents"
            scene_scores = f"{score}_scenes"
            infos[agent_scores].append(scores[score]["agent_scores"])
            infos[scene_scores].append(scores[score]["scene_score"])

        if (n + 1) % n_per_shard == 0 and n > 0 or n == len(features_files) - 1:
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
            N = len(infos["scenario_id"])
            infos = reset_infos(cfg)

            logger.info(
                f"Wrote shard {n // n_per_shard} with {N} scenarios to: {output_path}"
            )


if __name__ == "__main__":
    run()

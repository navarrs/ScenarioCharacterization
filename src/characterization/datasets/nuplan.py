from characterization.datasets.nuscenes import NuScenesData


class NuPlanData(NuScenesData):
    """Dataset adapter for the nuPlan dataset.

    nuPlan scenarios are preprocessed into the same Waymo-format pickles as nuScenes
    (see nuplan_preprocess.py), so loading and transformation are identical to
    NuScenesData; only the dataset tag differs.
    """

    DATASET_NAME = "nuplan"

"""Trajectory prediction scenario visualizer."""

import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from numpy.typing import NDArray
from omegaconf import DictConfig

from characterization.domains.aviation.scenario_types import AgentTrajectory
from characterization.utils.geometric_utils import XY_DIMENSIONS
from characterization.utils.logging_utils import get_pylogger
from safeair.scenario_visualizer.base_local_visualizer import BaseLocalVisualizer
from safeair.schemas import ModelOutput, Scenario

_LOGGER = get_pylogger(__name__)


class ScenarioTrajpredVisualizer(BaseLocalVisualizer):
    """Visualizer for trajectory prediction scenarios.

    Displays a single ego-centric pane with map features, agent histories colored by type, ground truth future
    trajectories as dashed lines, and predicted future trajectory modes colored by probability.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes the ScenarioTrajpredVisualizer."""
        super().__init__(config)

        self.pred_colormap = cm.get_cmap(config.get("pred_colormap", "summer"))

    def plot_predicted_modes(self, ax: Axes, model_output: ModelOutput) -> None:
        """Plots predicted trajectory modes as lines colored by probability.

        Modes are drawn lowest-probability first so the best mode renders on top.

        Args:
            ax: Axes to plot on.
            model_output: Model output containing trajectory prediction results.
        """
        traj_pred_output = model_output.trajectory_prediction_output
        assert traj_pred_output is not None, "Trajectory prediction output must be present."

        pred_trajs = traj_pred_output.decoded_trajectories.value.numpy()  # (M, K, F, D)
        pred_probs = traj_pred_output.mode_probabilities.value.numpy()  # (M,)
        best_mode_idx = int(pred_probs.argmax())
        num_agents = pred_trajs.shape[1]

        for idx in np.argsort(pred_probs):
            prob = float(pred_probs[idx])
            color = self.pred_colormap(prob)
            lw = 2.0 if idx == best_mode_idx else 1.0
            zorder = int(10 + prob * 10)
            alpha = min(1.0, prob + 0.2)

            for k in range(num_agents):
                traj = pred_trajs[idx, k]  # (F, D)
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=lw, alpha=alpha, zorder=zorder)
                ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=5, zorder=zorder + 1, alpha=alpha)

    def visualize_scenario_local(self, ax: Axes, scenario: Scenario, model_output: ModelOutput) -> None:
        """Plots trajectory prediction in ego-centric coordinate space.

        Draws map features, agent histories (colored by type with icons), ground truth futures (dashed, same colors),
        and all predicted future trajectory modes colored by probability.

        Args:
            ax: Axes to plot on.
            scenario: Scenario used to derive the map transform and agent type lookups.
            model_output: Model output containing trajectory prediction results.
        """
        assert model_output.trajectory_prediction_output is not None, "Trajectory prediction output must be present."

        history_gt = model_output.history_ground_truth.value.numpy()  # (N, H, S+1)
        future_gt = model_output.future_ground_truth.value.numpy()  # (N, F, S+1)
        ref_idx = history_gt.shape[1] - 1

        # Ego global position and heading are needed only for the map transform
        ego_idx = scenario.metadata.ego_agent_index
        agent_traj = AgentTrajectory(scenario.agent_data.agent_trajectories)
        ego_xy = agent_traj.xy_position[ego_idx, ref_idx]  # (2,)
        if not np.all(np.isfinite(ego_xy)):
            return
        ego_heading = agent_traj.heading[ego_idx, ref_idx]  # (1,)

        self.plot_map_data_in_local_frame(ax, ego_xy, ego_heading)
        self.plot_predicted_modes(ax, model_output)
        self.plot_sequences_local(ax, history_gt, future_gt)

        full_traj = self._build_full_trajectory(history_gt, future_gt)
        self.configure_axes_local(ax, full_traj)

    @staticmethod
    def _build_full_trajectory(history_gt: NDArray[np.float32], future_gt: NDArray[np.float32]) -> NDArray[np.float32]:
        """Concatenates history and future ground truth trajectories into a single array.

        Args:
            history_gt: Ground truth history trajectories, shape (N, H, S+1) with last feature as validity mask.
            future_gt: Ground truth future trajectories, shape (N, F, S+1) with last feature as validity mask.

        Returns:
            Full trajectories array of shape (N, H+F, 2+1) with
        """
        valid_hist = history_gt[:, :, -1].astype(bool)  # (N, H)
        xy_hist = history_gt[:, :, :XY_DIMENSIONS]  # (N, H, 2)
        hist = np.concatenate([xy_hist, valid_hist[..., None]], axis=-1)  # (N, H, 2+1)

        fut_valid = future_gt[:, :, -1].astype(bool)
        xy_fut = future_gt[:, :, :XY_DIMENSIONS]
        fut = np.concatenate([xy_fut, fut_valid[..., None]], axis=-1)  # (N, F, 2+1)

        return np.concatenate([hist, fut], axis=1)  # (N, H+F, 2+1)

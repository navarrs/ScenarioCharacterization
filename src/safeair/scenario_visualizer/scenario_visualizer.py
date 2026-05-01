from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig

from characterization.utils.logging_utils import get_pylogger
from safeair.scenario_visualizer.base_visualizer import BaseVisualizer, SupportedPanes
from safeair.schemas import ModelOutput, Scenario
from safeair.schemas.scenario_scores import ScenarioScores

_LOGGER = get_pylogger(__name__)


class ScenarioVisualizer(BaseVisualizer):
    """Visualizer for scenarios."""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the ScenarioVisualizer with the given configuration."""
        super().__init__(config)

        # Set up panes to plot. It will fail if an invalid pane is provided.
        panes = config.get("panes_to_plot", ["ALL_AGENTS"])
        self.panes_to_plot = [SupportedPanes[pane] for pane in panes]
        self.num_panes_to_plot = len(self.panes_to_plot)

    def visualize_scenario(  # noqa: PLR0912
        self,
        scenario: Scenario,
        model_output: ModelOutput | None,
        scores: ScenarioScores | None = None,
        output_dir: Path = Path("./temp"),
    ) -> Path:
        """Visualizes a single scenario and saves the output to a file.

        Renders one pane per entry in ``panes_to_plot``. Panes that cannot be drawn (e.g. missing critical metadata)
        are skipped and the remaining panes are still rendered.

        Args:
            scenario: Scenario to visualize.
            model_output: Model output for the scenario.
            scores: Optional scenario scores for score-based panes (SCORED_INDIVIDUAL, SCORED_INTERACTION).
            output_dir: Directory where the visualization will be saved.

        Returns:
            Path to the saved visualization file.
        """
        scenario_id = scenario.metadata.scenario_id
        output_filepath = output_dir / f"{scenario_id}.png"

        # Plot static and dynamic map information in the scenario
        axs = plt.subplots(1, self.num_panes_to_plot, figsize=(5 * self.num_panes_to_plot, 5 * 1))[1]
        self.plot_map_data(axs, self.num_panes_to_plot)

        for i, pane in enumerate(self.panes_to_plot):
            ax = axs[i] if self.num_panes_to_plot > 1 else axs
            match pane:
                case SupportedPanes.ALL_AGENTS:
                    self.plot_sequences(ax, scenario, title="All Agents")
                case SupportedPanes.HIGHLIGHT_RELEVANT_AGENTS:
                    # Plot trajectory data with relevant agents in a different color
                    self.plot_sequences(ax, scenario, show_relevant=True, title="Highlighted Relevant Trajectories")
                case SupportedPanes.CRITICAL_SCENARIO:
                    if scenario.critical_metadata is None:
                        _LOGGER.warning(
                            "CRITICAL_SCENARIO pane requested but no critical metadata available; skipping."
                        )
                        continue
                    self.plot_sequences_with_critical_metadata(ax, scenario)
                case SupportedPanes.CRITICAL_SCENARIO_IDENTIFICATION_PREDICTION:
                    if scenario.critical_metadata is None or model_output is None:
                        _LOGGER.warning(
                            "CRITICAL_SCENARIO_IDENTIFICATION_PREDICTION pane requested but critical metadata or model "
                            "output is missing; skipping."
                        )
                        continue
                    self.plot_sequences_with_critical_scenario_prediction(ax, scenario, model_output=model_output)
                case SupportedPanes.SCORED_INDIVIDUAL:
                    if scores is None or not scores.individual_scores:
                        _LOGGER.warning(
                            "SCORED_INDIVIDUAL pane requested but no individual scores available; skipping."
                        )
                        continue
                    self.plot_sequences_with_scores(ax, scenario, scores.individual_scores, title="Individual Scores")
                case SupportedPanes.SCORED_INTERACTION:
                    if scores is None or not scores.interaction_scores:
                        _LOGGER.warning(
                            "SCORED_INTERACTION pane requested but no interaction scores available; skipping."
                        )
                        continue
                    self.plot_sequences_with_scores(ax, scenario, scores.interaction_scores, title="Interaction Scores")
                case SupportedPanes.COUNTERFACTUAL_PROBE:
                    if scenario.critical_probe is None:
                        _LOGGER.warning(
                            "COUNTERFACTUAL_PROBE pane requested but scenario.critical_probe is None; skipping."
                        )
                        continue
                    self.plot_sequences_with_probe(ax, scenario)
                case _:
                    error_message = f"Unsupported pane type: {pane}"
                    raise ValueError(error_message)

        # Prepare and save plot
        self.configure_axes(axs, scenario, self.num_panes_to_plot)
        if self.add_title:
            plt.suptitle(f"Scenario: {scenario_id}")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        plt.close()

        _LOGGER.info("Visualized scenario to %s", output_filepath)
        return output_filepath

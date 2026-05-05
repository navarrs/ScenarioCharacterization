"""Aviation model output schemas for saving and loading ML model predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TensorField:
    """Wraps a ``torch.Tensor`` to expose it through a ``.value`` attribute."""

    value: torch.Tensor


def _wrap(tensor: torch.Tensor) -> TensorField:
    return TensorField(tensor)


def _maybe_wrap(tensor: torch.Tensor | None) -> TensorField | None:
    return None if tensor is None else TensorField(tensor)


@dataclass
class ScenarioEmbedding:
    """Scenario-level embedding produced by the model.

    Attributes:
        scenario_enc: Optional encoder embedding. Shape ``(D_enc,)``.
        scenario_dec: Decoder embedding. Shape ``(D_dec,)``.
    """

    scenario_enc: TensorField | None
    scenario_dec: TensorField

    def __init__(self, scenario_enc: torch.Tensor | None, scenario_dec: torch.Tensor) -> None:
        """Wrap raw tensors into TensorField accessors."""
        self.scenario_enc = _maybe_wrap(scenario_enc)
        self.scenario_dec = _wrap(scenario_dec)


@dataclass
class TrajectoryPredictionOutput:
    """Output of the trajectory prediction head.

    Attributes:
        decoded_trajectories: Predicted future trajectories. Shape ``(M, T, D)``.
        mode_probabilities: Per-mode probability scores. Shape ``(M,)``.
        mode_logits: Raw pre-softmax logits. Shape ``(M,)``.
    """

    decoded_trajectories: TensorField
    mode_probabilities: TensorField
    mode_logits: TensorField

    def __init__(
        self,
        decoded_trajectories: torch.Tensor,
        mode_probabilities: torch.Tensor,
        mode_logits: torch.Tensor,
    ) -> None:
        """Wrap raw tensors into TensorField accessors."""
        self.decoded_trajectories = _wrap(decoded_trajectories)
        self.mode_probabilities = _wrap(mode_probabilities)
        self.mode_logits = _wrap(mode_logits)


@dataclass
class TokenizationOutput:
    """Output of the tokenization head (scene or per-agent).

    Attributes:
        token_probabilities: Per-token probability distribution. Optional.
        token_indices: Discrete token index assignments. Shape ``(K,)``.
        input_embedding: Input embedding before quantization. Shape ``(D,)``.
        reconstructed_embedding: Reconstructed embedding after quantization. Optional.
        quantized_embedding: Quantized (codebook) embedding. Optional.
    """

    token_probabilities: TensorField | None
    token_indices: TensorField
    input_embedding: TensorField
    reconstructed_embedding: TensorField | None
    quantized_embedding: TensorField | None

    def __init__(
        self,
        token_indices: torch.Tensor,
        input_embedding: torch.Tensor,
        token_probabilities: torch.Tensor | None = None,
        reconstructed_embedding: torch.Tensor | None = None,
        quantized_embedding: torch.Tensor | None = None,
    ) -> None:
        """Wrap raw tensors into TensorField accessors."""
        self.token_probabilities = _maybe_wrap(token_probabilities)
        self.token_indices = _wrap(token_indices)
        self.input_embedding = _wrap(input_embedding)
        self.reconstructed_embedding = _maybe_wrap(reconstructed_embedding)
        self.quantized_embedding = _maybe_wrap(quantized_embedding)


@dataclass
class CriticalScenarioIdentificationOutput:
    """Output of the critical-scenario identification head.

    Attributes:
        critical_agents_gt_mask: Ground-truth binary mask for critical agents. Shape ``(N,)``.
        critical_agents_pred_probabilities: Predicted criticality probabilities. Shape ``(N,)``.
        los_timestep_index: Predicted loss-of-separation timestep index. Shape ``()``.
        los_timestep_probabilities: Per-timestep LoS probability distribution. Shape ``(T,)``.
    """

    critical_agents_gt_mask: TensorField
    critical_agents_pred_probabilities: TensorField
    los_timestep_index: TensorField
    los_timestep_probabilities: TensorField

    def __init__(
        self,
        critical_agents_gt_mask: torch.Tensor,
        critical_agents_pred_probabilities: torch.Tensor,
        los_timestep_index: torch.Tensor,
        los_timestep_probabilities: torch.Tensor,
    ) -> None:
        """Wrap raw tensors into TensorField accessors."""
        self.critical_agents_gt_mask = _wrap(critical_agents_gt_mask)
        self.critical_agents_pred_probabilities = _wrap(critical_agents_pred_probabilities)
        self.los_timestep_index = _wrap(los_timestep_index)
        self.los_timestep_probabilities = _wrap(los_timestep_probabilities)


@dataclass
class ModelOutput:
    """Full model output for a single scenario or a batch of scenarios.

    Tensor fields use :class:`TensorField` wrappers so consumers always access data via ``.value``,
    regardless of whether the object holds a batch or a single-scenario slice.

    Attributes:
        scenario_id: Scenario identifier(s).
        dataset_name: Dataset name(s) for each scenario.
        history_ground_truth: Historical trajectory ground truth. Shape ``(T_hist, N, D)`` or ``(N, D)``.
        future_ground_truth: Future trajectory ground truth. Shape ``(T_fut, N, D)`` or ``(N, D)``.
        ego_ground_truth: Ego-agent trajectory ground truth.
        agent_ids: Agent identifier tensor. Shape ``(N,)``.
        scenario_embedding: Optional scenario-level embedding.
        trajectory_prediction_output: Optional trajectory prediction output.
        tokenization_output: Optional scene-level tokenization output.
        agent_tokenization_output: Optional per-agent tokenization output.
        critical_scenario_identification_output: Optional criticality identification output.
    """

    scenario_id: list[str]
    dataset_name: list[str]
    history_ground_truth: TensorField
    future_ground_truth: TensorField
    ego_ground_truth: TensorField
    agent_ids: TensorField
    scenario_embedding: ScenarioEmbedding | None
    trajectory_prediction_output: TrajectoryPredictionOutput | None
    tokenization_output: TokenizationOutput | None
    agent_tokenization_output: TokenizationOutput | None
    critical_scenario_identification_output: CriticalScenarioIdentificationOutput | None

    def __init__(
        self,
        scenario_id: list[str],
        dataset_name: list[str],
        history_ground_truth: torch.Tensor,
        future_ground_truth: torch.Tensor,
        ego_ground_truth: torch.Tensor,
        agent_ids: torch.Tensor,
        scenario_embedding: ScenarioEmbedding | None = None,
        trajectory_prediction_output: TrajectoryPredictionOutput | None = None,
        tokenization_output: TokenizationOutput | None = None,
        agent_tokenization_output: TokenizationOutput | None = None,
        critical_scenario_identification_output: CriticalScenarioIdentificationOutput | None = None,
    ) -> None:
        """Wrap raw tensors into TensorField accessors."""
        self.scenario_id = scenario_id
        self.dataset_name = dataset_name
        self.history_ground_truth = _wrap(history_ground_truth)
        self.future_ground_truth = _wrap(future_ground_truth)
        self.ego_ground_truth = _wrap(ego_ground_truth)
        self.agent_ids = _wrap(agent_ids)
        self.scenario_embedding = scenario_embedding
        self.trajectory_prediction_output = trajectory_prediction_output
        self.tokenization_output = tokenization_output
        self.agent_tokenization_output = agent_tokenization_output
        self.critical_scenario_identification_output = critical_scenario_identification_output

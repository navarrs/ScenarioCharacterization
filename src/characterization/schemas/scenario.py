import pickle
from pathlib import Path
from typing import Self, cast

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from characterization.schemas.types import Float32NDArray1D, Float32NDArray3D


class BaseAgentData(BaseModel):
    """Common agent data fields shared across domains.

    Subclasses define domain-specific ``agent_ids`` and ``agent_types`` container types and add the ``num_agents``
    computed field.

    Attributes:
        agent_trajectories: Shape (N, T, 10) trajectory array. Layout is domain-specific.
        agent_relevance: Shape (N,) relevance scores. Higher values mean more relevant; NaN or negative means
            irrelevant. If None, all agents are equally relevant.
    """

    agent_trajectories: Float32NDArray3D
    agent_relevance: Float32NDArray1D | None = None

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}


class BaseScenarioMetadata(BaseModel):
    """Common scenario metadata fields shared across domains.

    Subclasses define ``timestamps_seconds`` (type differs by domain) and domain-specific ego/agent fields. The
    ``duration_s`` computed field is also domain-defined since it depends on ``timestamps_seconds``.

    Attributes:
        scenario_id: Unique scenario identifier.
        frequency_hz: Data frequency in Hz.
        current_time_index: Index of the current timestep. Optional at the base level; some domains require it.
        track_length: Number of timesteps in the track.
        dataset: Name of the source dataset.
    """

    scenario_id: str
    frequency_hz: float
    current_time_index: int | None = None
    track_length: int
    dataset: str

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore from pickle, backfilling defaults for fields added after the pickle was created.

        This maintains backward compatibility with pickled objects created before new fields were added.
        """
        field_dict = cast("dict[str, object]", state.get("__dict__", state))
        for field_name, field_info in self.__class__.model_fields.items():
            if field_name not in field_dict and field_info.default is not PydanticUndefined:
                field_dict[field_name] = field_info.default
        super().__setstate__(state)


class BaseScenario(BaseModel):
    """Base scenario class providing pickle serialization and backward-compatible unpickling.

    Domain subclasses inherit ``to_pickle``, ``from_pickle``, and ``__setstate__`` and only need to declare their
    domain-specific fields.
    """

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore from pickle, backfilling defaults for fields added after the pickle was created."""
        field_dict = cast("dict[str, object]", state.get("__dict__", state))
        for field_name, field_info in self.__class__.model_fields.items():
            if field_name not in field_dict and field_info.default is not PydanticUndefined:
                field_dict[field_name] = field_info.default
        super().__setstate__(state)

    def to_pickle(self, save_dir: Path) -> None:
        """Serialize to a pickle file named ``{scenario_id}.pkl`` inside ``save_dir``.

        Args:
            save_dir: Directory in which to write the pickle file.
        """
        scene_filepath = save_dir / f"{self.metadata.scenario_id}.pkl"  # pyright: ignore[reportAttributeAccessIssue]
        with scene_filepath.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, filepath: Path) -> "Self":
        """Deserialize from a pickle file.

        Args:
            filepath: Path to the pickle file.

        Returns:
            The deserialized scenario instance.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)  # pyright: ignore[reportReturnType]

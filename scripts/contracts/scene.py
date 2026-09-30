from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .state import StateFrame


@runtime_checkable
class Scene(Protocol):
    def metadata(self) -> dict[str, Any]: ...
    def initial_state(self) -> StateFrame: ...
    def sample_times(self) -> list[float]: ...

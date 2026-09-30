from __future__ import annotations

from typing import Protocol, runtime_checkable

from .scene import Scene
from .state import StateFrame


@runtime_checkable
class Solver(Protocol):
    def initialize(self, scene: Scene, seed: int) -> None: ...
    def advance_to(self, time: float) -> StateFrame: ...
    def diagnostics(self) -> dict[str, object]: ...

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import imageio.v2 as imageio
import numpy as np


def write_video(path: Path, frames: Iterable[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)  # type: ignore[attr-defined]

"""Select a source video's physical-time window without changing the trajectory."""
from __future__ import annotations

import math


def video_window(config: dict) -> tuple[float, float] | None:
    window = config.get("extension", {}).get("video_window")
    if window is None:
        return None
    if not isinstance(window, dict):
        raise ValueError("extension.video_window must contain start and end physical times")
    values = (window.get("start"), window.get("end"))
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("Video window times must be numeric")
    start, end = map(float, values)
    if not all(map(math.isfinite, (start, end))) or start < 0 or end < start:
        raise ValueError("Invalid video physical-time window")
    return start, end


def selected_video_frames(frames, config: dict):
    window = video_window(config)
    selected = 0
    for frame in frames:
        if window is not None and not window[0] - 1e-9 <= float(frame.time) <= window[1] + 1e-9:
            continue
        selected += 1
        yield frame
    if not selected:
        raise ValueError("No trajectory frames lie in the configured video window")

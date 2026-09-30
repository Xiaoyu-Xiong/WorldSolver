from __future__ import annotations

from typing import Any


def aggregate_evaluation(
    execution: dict[str, Any],
    physics: list[dict[str, Any]],
    visual: list[dict[str, Any]],
) -> dict[str, Any]:
    physics_complete = bool(physics) and all(
        item.get("normalized_error") is not None for item in physics
    )
    visual_complete = bool(visual) and all(
        item.get("score") is not None
        and not item.get("status", "").startswith("failed")
        and not item.get("details", {}).get("error")
        for item in visual
    )
    if not physics_complete:
        status = "physics_measurement_error"
    elif not visual_complete:
        status = "partial_visual_pending"
    else:
        status = "complete"
    return {
        "status": status,
        "execution_pass": bool(execution["passed"]),
        "visual_score": (
            sum(float(item["score"]) for item in visual) / len(visual)
            if visual_complete
            else None
        ),
        "criteria": {
            "execution": [execution],
            "physics": physics,
            "visual": visual,
        },
    }

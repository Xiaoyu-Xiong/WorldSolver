from __future__ import annotations

import math
from statistics import fmean
from typing import Any


# Frozen error-to-score mappings used by the published WorldSolver evaluation.
PHYSICS_MAPPINGS = {
    "balance.extensive_quantity.v1": ("reciprocal", 0.01),
    "balance.source_aware.v1": ("reciprocal", 0.01),
    "boundary.velocity_residual.v1": ("reciprocal", 0.1),
    "constraint.target.v1": ("reciprocal", 0.1),
    "contact.coulomb_law.v1": ("reciprocal", 0.1),
    "contact.penetration.v1": ("reciprocal", 0.05),
    "coupling.force_balance.v1": ("linear_bound", 1.0),
    "deformation.jacobian.v1": ("linear_bound", 1.0),
    "deformation.strain_limit.v1": ("reciprocal", 0.1),
    "dynamics.angular_momentum_budget.v1": ("reciprocal", 0.1),
    "dynamics.linear_momentum_balance.v1": ("linear_bound", 1.0),
    "dynamics.momentum_budget.v1": ("reciprocal", 0.1),
    "field.continuity.v1": ("linear_bound", 1.0),
    "field.divergence.v1": ("reciprocal", 1.0),
    "field.scalar_transport.v1": ("linear_bound", 1.0),
    "interface.level_set_kinematic.v1": ("linear_bound", 1.0),
    "porous.darcy_balance.v1": ("reciprocal", 0.1),
    "rod.attachment_inextensibility.v1": ("reciprocal", 0.05),
    "rod.cantilever_force_balance.v1": ("reciprocal", 0.1),
    "rod.cantilever_inextensibility.v1": ("reciprocal", 0.05),
    "rod.cantilever_moment_balance.v1": ("reciprocal", 0.1),
    "trajectory.center_of_mass_consistency.v1": ("linear_bound", 2.0),
    "trajectory.kinematic_continuity.v1": ("linear_bound", 2.0),
}


def _physics_score(error: Any, calculator: str) -> float:
    if (
        not isinstance(error, (int, float))
        or isinstance(error, bool)
        or not math.isfinite(error)
        or error < 0
    ):
        return 0.0
    kind, scale = PHYSICS_MAPPINGS[calculator]
    value = 1.0 - error / scale if kind == "linear_bound" else scale / (scale + error)
    return 100.0 * max(0.0, min(1.0, value))


def attach_scores(report: dict[str, Any]) -> dict[str, Any]:
    physics_scores = []
    for criterion in report["criteria"]["physics"]:
        calculator = criterion.get("calculator")
        score = (
            _physics_score(criterion.get("normalized_error"), calculator)
            if calculator in PHYSICS_MAPPINGS
            else 0.0
        )
        criterion["score"] = score
        physics_scores.append(score)

    visual = report.get("visual_score")
    visual_fidelity = (
        100.0 * float(visual)
        if isinstance(visual, (int, float)) and not isinstance(visual, bool)
        else None
    )
    physical_plausibility = fmean(physics_scores) if physics_scores else 0.0
    total = (
        fmean((visual_fidelity, physical_plausibility))
        if visual_fidelity is not None
        else None
    )
    report.update(
        {
            "score_scale": "0-100",
            "visual_fidelity_score": visual_fidelity,
            "physical_plausibility_score": physical_plausibility,
            "total_score": total,
        }
    )
    return report

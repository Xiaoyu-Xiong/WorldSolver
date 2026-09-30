from .balance import ExtensiveQuantityCalculator
from .base import Calculator, CalculatorResult
from .boundary import BoundaryVelocityCalculator
from .case_specific import (
    BeltDriveReversalCalculator,
    BeltGeometricContactCalculator,
    BeltSurfaceVelocityTransferCalculator,
    BeltTensionedTriangleCalculator,
    ContactCoverageCalculator,
    FeatureContactCoverageCalculator,
    InclinedPlaneReferenceCalculator,
    KinematicContinuityCalculator,
    LocalDeformationSignatureCalculator,
    LoopWrapCalculator,
    ObjectRetentionCalculator,
    OscillationFrequencyPhaseCalculator,
    RigidPoseTrajectoryCalculator,
    RodAttachmentInextensibilityCalculator,
    RodCantileverInextensibilityCalculator,
    RodCantileverStaticForceBalanceCalculator,
    RodCantileverStaticMomentBalanceCalculator,
)
from .constitutive_transport import (
    CoulombLawCalculator,
    DarcyBalanceCalculator,
    NewtonianStressCalculator,
)
from .deformable_cases import (
    BarContactActivationCalculator,
    DroopProgressionCalculator,
    LocalizedJiggleCalculator,
    MergeBridgeContinuityCalculator,
    PeakResidualRecoveryCalculator,
    PersistentMergeProgressCalculator,
    ResidualDeformationGrowthCalculator,
    StairContactEventCountCalculator,
    TopologyTemporalCoherenceCalculator,
)
from .elastic_law import ElasticLawCalculator
from .field import DivergenceCalculator
from .generic import (
    ContactPenetrationCalculator,
    ContinuityCalculator,
    DeformationJacobianCalculator,
    InteractionForceBalanceCalculator,
    LinearMomentumBalanceCalculator,
    RigidCentroidDriftCalculator,
    RigidImpulseTransferCalculator,
    RigidKineticEventSettlingCalculator,
    RigidLoopContactProgressCalculator,
    RigidResidualKineticEnergyCalculator,
    RigidShapeCalculator,
    ScalarTransportCalculator,
    SourceAwareBalanceCalculator,
    StrainLimitCalculator,
    TargetConstraintCalculator,
    WilberforceModeEnergyExchangeCalculator,
)
from .interface import LevelSetKinematicCalculator
from .kinematics import CenterOfMassConsistencyCalculator
from .mechanical_budgets import (
    AngularMomentumBudgetCalculator,
    MomentumBudgetCalculator,
    WorkBalanceCalculator,
)

CALCULATORS: dict[str, Calculator] = {
    calculator.id: calculator
    for calculator in (
        ExtensiveQuantityCalculator(),
        DivergenceCalculator(),
        BoundaryVelocityCalculator(),
        LevelSetKinematicCalculator(),
        SourceAwareBalanceCalculator(),
        LinearMomentumBalanceCalculator(),
        ContactPenetrationCalculator(),
        TargetConstraintCalculator(),
        RigidShapeCalculator(),
        CenterOfMassConsistencyCalculator(),
        RigidImpulseTransferCalculator(),
        RigidResidualKineticEnergyCalculator(),
        RigidKineticEventSettlingCalculator(),
        RigidLoopContactProgressCalculator(),
        RigidCentroidDriftCalculator(),
        DeformationJacobianCalculator(),
        StrainLimitCalculator(),
        ContinuityCalculator(),
        ScalarTransportCalculator(),
        InteractionForceBalanceCalculator(),
        WilberforceModeEnergyExchangeCalculator(),
        InclinedPlaneReferenceCalculator(),
        OscillationFrequencyPhaseCalculator(),
        RodAttachmentInextensibilityCalculator(),
        RodCantileverInextensibilityCalculator(),
        RodCantileverStaticForceBalanceCalculator(),
        RodCantileverStaticMomentBalanceCalculator(),
        KinematicContinuityCalculator(),
        ContactCoverageCalculator(),
        LoopWrapCalculator(),
        FeatureContactCoverageCalculator(),
        LocalDeformationSignatureCalculator(),
        RigidPoseTrajectoryCalculator(),
        ObjectRetentionCalculator(),
        BeltGeometricContactCalculator(),
        BeltTensionedTriangleCalculator(),
        BeltSurfaceVelocityTransferCalculator(),
        BeltDriveReversalCalculator(),
        StairContactEventCountCalculator(),
        ResidualDeformationGrowthCalculator(),
        DroopProgressionCalculator(),
        BarContactActivationCalculator(),
        PeakResidualRecoveryCalculator(),
        LocalizedJiggleCalculator(),
        PersistentMergeProgressCalculator(),
        MergeBridgeContinuityCalculator(),
        TopologyTemporalCoherenceCalculator(),
        MomentumBudgetCalculator(),
        AngularMomentumBudgetCalculator(),
        WorkBalanceCalculator(),
        CoulombLawCalculator(),
        NewtonianStressCalculator(),
        DarcyBalanceCalculator(),
        ElasticLawCalculator(),
    )
}

__all__ = ["CALCULATORS", "Calculator", "CalculatorResult"]

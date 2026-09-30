"""WorldSolver runtime, rendering, and evaluation scripts."""

import sys


# Frozen solver submissions import ``worldsolver.contracts``. Keep that public
# contract working without retaining the original platform/src/worldsolver tree.
sys.modules.setdefault("worldsolver", sys.modules[__name__])

__version__ = "0.1.0"

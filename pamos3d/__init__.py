"""
pamos3d - decision engine for the PAMOS-3D streaming framework.

Loads a Perceptual Representation Contract Manifest (PRCM), runs D2AN to
predict perceptual quality, and allocates one representation per object
under a bandwidth budget.

Author: Simone Porcu <simone.porcu@unica.it>

Example:
    from pamos3d import Contract, D2AN, PAMOS3DEngine

    contract = Contract.load("scene.prcm.json")
    model = D2AN.from_pretrained("pointcloud-v1")
    engine = PAMOS3DEngine(model)
    decision = engine.decide(contract)
"""

from .contract import (
    Contract,
    ContractError,
    Representation,
    ResourceProfile,
    RuntimeContext,
    SceneObject,
    VisibilityPolicy,
)
from .model import D2AN, ModelError
from .allocator import SceneDecision, AllocationError, allocate, STRATEGIES
from .engine import PAMOS3DEngine
from .delivery import DeliveryReport, DeliveredObject, DeliveryError, deliver_scene, deliver_object

__version__ = "0.1.0"
__author__ = "Simone Porcu"

__all__ = [
    "Contract",
    "ContractError",
    "Representation",
    "ResourceProfile",
    "RuntimeContext",
    "SceneObject",
    "VisibilityPolicy",
    "D2AN",
    "ModelError",
    "SceneDecision",
    "AllocationError",
    "allocate",
    "STRATEGIES",
    "PAMOS3DEngine",
    "DeliveryReport",
    "DeliveredObject",
    "DeliveryError",
    "deliver_scene",
    "deliver_object",
]

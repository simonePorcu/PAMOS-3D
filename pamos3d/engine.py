"""
pamos3d.engine

High-level facade tying contract, model, and allocator together.

PAMOS3DEngine.decide() is meant to be called once per control interval
with the latest Contract snapshot. It only re-invokes D2AN when
something relevant to the decision (bandwidth, or a visible object's
distance) has actually changed since the last call.

Author: Simone Porcu
"""

from __future__ import annotations

from typing import Dict, Optional

from .allocator import SceneDecision, Strategy, allocate
from .contract import Contract
from .model import D2AN


class PAMOS3DEngine:
    """
    Usage:

        model = D2AN.from_pretrained("pointcloud-v1")
        engine = PAMOS3DEngine(model)

        decision = engine.decide(contract)
        decision2 = engine.decide(contract)  # unchanged -> cached
        decision3 = engine.decide(contract.with_updated_context(bandwidth_mbps=90))
    """

    def __init__(self, model: D2AN, visible_slots: int = 2):
        self._model = model
        self._visible_slots = visible_slots
        self._last_signature: Optional[str] = None
        self._last_decisions: Dict[Strategy, SceneDecision] = {}

    def reset(self) -> None:
        """Forget the cached decision, forcing the next call to recompute."""
        self._last_signature = None
        self._last_decisions = {}

    def decide(self, contract: Contract, strategy: Strategy = "d2an") -> SceneDecision:
        """
        Return the scene decision for `strategy`, reusing the previous
        result if the contract's decision-relevant fields have not
        changed since the last call.
        """
        signature = contract.signature()
        if signature != self._last_signature:
            self._last_signature = signature
            self._last_decisions = {}
        if strategy not in self._last_decisions:
            self._last_decisions[strategy] = allocate(
                contract, self._model, strategy=strategy, visible_slots=self._visible_slots
            )
        return self._last_decisions[strategy]

    def was_recomputed(self, contract: Contract) -> bool:
        """Whether the next decide() call for this contract would trigger D2AN."""
        return contract.signature() != self._last_signature

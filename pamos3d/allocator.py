"""
pamos3d.allocator

Joint scene allocator: given a Contract and a trained D2AN instance,
choose one representation per visible object to maximize
importance-weighted predicted quality under the bandwidth budget.

Solved by exact enumeration over complete scene configurations. This is
simple to audit and adequate for a handful of objects/candidates; a
scalable solver (pruning, a knapsack solver, ...) can implement the same
interface used here.

Author: Simone Porcu
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional

from .contract import Contract, SceneObject, Representation
from .model import D2AN

Strategy = str  # "d2an" | "max-bitrate" | "greedy" | "uniform" | "hybrid"

STRATEGIES = ("d2an", "max-bitrate", "greedy", "uniform", "hybrid")


@dataclass
class SceneDecision:
    """The result of one allocation."""

    document_version: int
    strategy: Strategy
    feasible: bool
    selection: Dict[str, str] = field(default_factory=dict)      # object_id -> representation_id
    per_object_mos: Dict[str, float] = field(default_factory=dict)
    per_object_bitrate_bps: Dict[str, int] = field(default_factory=dict)
    total_bitrate_bps: int = 0
    scene_mos: float = 0.0          # mean predicted MOS across selected objects
    scene_utility: float = 0.0      # importance-weighted utility sum (the objective in Eq. "allocation")

    def to_dict(self) -> dict:
        return {
            "documentVersion": self.document_version,
            "strategy": self.strategy,
            "feasible": self.feasible,
            "selection": dict(self.selection),
            "perObjectMOS": dict(self.per_object_mos),
            "perObjectBitrateBps": dict(self.per_object_bitrate_bps),
            "totalBitrateBps": self.total_bitrate_bps,
            "sceneMOS": self.scene_mos,
            "sceneUtility": self.scene_utility,
        }


class AllocationError(RuntimeError):
    """Raised when no feasible configuration exists under the given budget."""


@dataclass
class _Candidate:
    object_id: str
    representation: Representation
    mos: float
    utility: float
    weighted_utility: float
    bitrate_bps: int


def _build_candidates(
    contract: Contract, model: D2AN, objects: List[SceneObject]
) -> Dict[str, List[_Candidate]]:
    viewer = contract.runtime_context.viewer_position
    per_object: Dict[str, List[_Candidate]] = {}
    for obj in objects:
        distance = obj.distance_to(viewer)
        weight = model.distance_weight(distance)
        cands = []
        for rep in obj.representations:
            if not model.supports(rep.encoding_method):
                # Candidates outside this model's calibration domain
                # (e.g. a Gaussian Splat format) are skipped.
                continue
            mos = model.predict_mos(rep, distance)
            utility = (mos - 1.0) / 4.0
            cands.append(
                _Candidate(
                    object_id=obj.object_id,
                    representation=rep,
                    mos=mos,
                    utility=utility,
                    weighted_utility=utility * weight,
                    bitrate_bps=rep.resource_profile.bitrate_bps,
                )
            )
        if cands:
            per_object[obj.object_id] = cands
    return per_object


def _decision_from_chosen(
    document_version: int, strategy: Strategy, chosen: Dict[str, _Candidate]
) -> SceneDecision:
    if not chosen:
        return SceneDecision(document_version=document_version, strategy=strategy, feasible=False)
    total_bitrate = sum(c.bitrate_bps for c in chosen.values())
    scene_mos = sum(c.mos for c in chosen.values()) / len(chosen)
    scene_utility = sum(c.weighted_utility for c in chosen.values())
    return SceneDecision(
        document_version=document_version,
        strategy=strategy,
        feasible=True,
        selection={oid: c.representation.representation_id for oid, c in chosen.items()},
        per_object_mos={oid: c.mos for oid, c in chosen.items()},
        per_object_bitrate_bps={oid: c.bitrate_bps for oid, c in chosen.items()},
        total_bitrate_bps=total_bitrate,
        scene_mos=scene_mos,
        scene_utility=scene_utility,
    )


def _allocate_exact(
    per_object: Dict[str, List[_Candidate]], budget_bps: float, maximize: str
) -> Optional[Dict[str, _Candidate]]:
    """
    Exact enumeration over all complete scene configurations. maximize is
    "utility" (D2AN's own objective) or "bitrate" (Max-bitrate baseline).
    """
    object_ids = list(per_object.keys())
    best_chosen: Optional[Dict[str, _Candidate]] = None
    best_score = float("-inf")
    for combo in product(*(per_object[oid] for oid in object_ids)):
        total_bitrate = sum(c.bitrate_bps for c in combo)
        if total_bitrate > budget_bps + 1e-6:
            continue
        score = total_bitrate if maximize == "bitrate" else sum(c.weighted_utility for c in combo)
        if score > best_score:
            best_score = score
            best_chosen = dict(zip(object_ids, combo))
    return best_chosen


def _allocate_baseline(
    per_object: Dict[str, List[_Candidate]],
    objects_by_id: Dict[str, SceneObject],
    viewer_position,
    budget_bps: float,
    scheme: str,
    visible_slots: int = 2,
) -> Dict[str, _Candidate]:
    """Greedy / Uniform / Hybrid baselines from PCC-DASH, adapted to PRCM candidates."""
    ordered_ids = sorted(
        per_object.keys(), key=lambda oid: objects_by_id[oid].distance_to(viewer_position)
    )
    ladders = {
        oid: sorted(per_object[oid], key=lambda c: (c.bitrate_bps, c.representation.encoding_method, c.representation.quality_setting))
        for oid in ordered_ids
    }
    idx = {oid: 0 for oid in ordered_ids}

    def total_cost() -> float:
        return sum(ladders[oid][idx[oid]].bitrate_bps for oid in ordered_ids)

    def try_upgrade(oid: str) -> bool:
        i = idx[oid]
        if i + 1 >= len(ladders[oid]):
            return False
        inc = ladders[oid][i + 1].bitrate_bps - ladders[oid][i].bitrate_bps
        if total_cost() + inc > budget_bps + 1e-6:
            return False
        idx[oid] += 1
        return True

    if total_cost() <= budget_bps + 1e-6:
        if scheme == "greedy":
            for oid in ordered_ids:
                while try_upgrade(oid):
                    pass
        elif scheme == "uniform":
            changed = True
            while changed:
                changed = False
                for oid in ordered_ids:
                    changed = try_upgrade(oid) or changed
        elif scheme == "hybrid":
            visible = ordered_ids[:visible_slots]
            rest = ordered_ids[visible_slots:]
            changed = True
            while changed:
                changed = False
                for oid in visible:
                    changed = try_upgrade(oid) or changed
            for oid in rest:
                while try_upgrade(oid):
                    pass
        else:
            raise ValueError(f"unknown baseline scheme: {scheme!r}")

    return {oid: ladders[oid][idx[oid]] for oid in ordered_ids}


def allocate(contract: Contract, model: D2AN, strategy: Strategy = "d2an", visible_slots: int = 2) -> SceneDecision:
    """
    Select one representation per visible object under the contract's
    current bandwidth budget.

    Parameters
    ----------
    contract:
        The current PRCM snapshot. Only objects whose visibility policy
        allows it are considered.
    model:
        The trained D2AN instance used for quality prediction.
    strategy:
        "d2an" (perception-aware joint allocation), or one of
        "max-bitrate", "greedy", "uniform", "hybrid" for comparison.
    visible_slots:
        Number of nearest objects treated as visible by the Hybrid
        baseline (ignored by the other strategies).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    objects = contract.visible_objects()
    per_object = _build_candidates(contract, model, objects)
    if not per_object:
        return SceneDecision(document_version=contract.document_version, strategy=strategy, feasible=False)

    budget_bps = contract.runtime_context.bandwidth_mbps * 1e6
    objects_by_id = {o.object_id: o for o in objects}

    if strategy == "d2an":
        chosen = _allocate_exact(per_object, budget_bps, maximize="utility")
    elif strategy == "max-bitrate":
        chosen = _allocate_exact(per_object, budget_bps, maximize="bitrate")
    else:
        chosen = _allocate_baseline(
            per_object, objects_by_id, contract.runtime_context.viewer_position,
            budget_bps, strategy, visible_slots,
        )

    if not chosen:
        return SceneDecision(document_version=contract.document_version, strategy=strategy, feasible=False)
    return _decision_from_chosen(contract.document_version, strategy, chosen)

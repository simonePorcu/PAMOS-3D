"""
Checks that pamos3d's Python allocator reproduces the same scene
decisions as the JS implementation, given the same example contract.

Expected values were computed with the JS implementation on
pamos3d_reference_contract.json at its 120 Mbps budget, for all five
strategies.

Author: Simone Porcu
"""

from pathlib import Path

import pytest

from pamos3d import Contract, D2AN, PAMOS3DEngine

HERE = Path(__file__).parent
CONTRACT_PATH = HERE.parent / "examples" / "pamos3d_reference_contract.json"

EXPECTED_SELECTION = {
    "d2an": {"PC1": "pc1-vpcc_q3", "PC2": "pc2-vpcc_q3", "PC3": "pc3-vpcc_q2"},
    "max-bitrate": {"PC1": "pc1-vpcc_q2", "PC2": "pc2-gpcc_trisoup_q3", "PC3": "pc3-vpcc_q3"},
    "greedy": {"PC1": "pc1-gpcc_trisoup_q3", "PC2": "pc2-gpcc_trisoup_q1", "PC3": "pc3-gpcc_octree_q2"},
    "uniform": {"PC1": "pc1-gpcc_trisoup_q2", "PC2": "pc2-vpcc_q3", "PC3": "pc3-vpcc_q1"},
    "hybrid": {"PC1": "pc1-gpcc_trisoup_q3", "PC2": "pc2-gpcc_trisoup_q1", "PC3": "pc3-gpcc_octree_q2"},
}
EXPECTED_TOTAL_MBPS = {
    "d2an": 119.81,
    "max-bitrate": 119.96,
    "greedy": 117.33,
    "uniform": 119.76,
    "hybrid": 117.33,
}


@pytest.fixture(scope="module")
def engine():
    model = D2AN.from_pretrained("pointcloud-v1")
    return PAMOS3DEngine(model)


@pytest.fixture(scope="module")
def contract():
    return Contract.load(CONTRACT_PATH)


def test_contract_loads_and_validates(contract):
    assert contract.contract == "PRCM"
    assert len(contract.objects) == 3
    assert {o.object_id for o in contract.objects} == {"PC1", "PC2", "PC3"}
    for obj in contract.objects:
        assert len(obj.representations) == 9


def test_all_strategies_match_js_reference(engine, contract):
    for strategy, expected_sel in EXPECTED_SELECTION.items():
        decision = engine.decide(contract, strategy=strategy)
        assert decision.feasible, strategy
        assert decision.selection == expected_sel, strategy
        assert decision.total_bitrate_bps / 1e6 == pytest.approx(EXPECTED_TOTAL_MBPS[strategy], abs=0.05), strategy


def test_all_strategies_feasible_and_within_budget(engine, contract):
    budget_bps = contract.runtime_context.bandwidth_mbps * 1e6
    for strategy in ("d2an", "max-bitrate", "greedy", "uniform", "hybrid"):
        decision = engine.decide(contract, strategy=strategy)
        assert decision.feasible, strategy
        assert decision.total_bitrate_bps <= budget_bps + 1, strategy


def test_engine_caches_unchanged_contract(engine, contract):
    engine.reset()
    assert engine.was_recomputed(contract) is True
    _ = engine.decide(contract)
    assert engine.was_recomputed(contract) is False  # same signature -> would reuse

    moved = contract.with_updated_context(bandwidth_mbps=contract.runtime_context.bandwidth_mbps - 1)
    assert engine.was_recomputed(moved) is True  # bandwidth changed -> would recompute


def test_visibility_policy_excludes_object(engine, contract):
    hidden_ids = {o.object_id for o in contract.objects if o.object_id != "PC2"}
    reduced = contract.with_updated_context(visible_object_ids=hidden_ids)
    decision = engine.decide(reduced, strategy="d2an")
    assert set(decision.selection.keys()) == {"PC1", "PC3"}


def test_unsupported_encoding_method_is_skipped(engine, contract):
    # Synthetically add a Gaussian-Splat-like candidate the point-cloud
    # D2AN instance does not cover; it must be ignored, not crash.
    import copy
    from pamos3d.contract import Representation, ResourceProfile

    c2 = copy.deepcopy(contract)
    c2.objects[0].representations.append(
        Representation(
            representation_id="pc1-gsplat-fake",
            format="gaussian-splat",
            encoding_method="GSPLAT",
            quality_setting=1,
            content_uri="",
            resource_profile=ResourceProfile(bitrate_bps=1_000_000),
        )
    )
    decision = engine.decide(c2, strategy="d2an")
    assert decision.feasible
    assert decision.selection["PC1"] != "pc1-gsplat-fake"

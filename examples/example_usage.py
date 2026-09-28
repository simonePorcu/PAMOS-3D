"""
Minimal usage example for the pamos3d decision-engine library.

Run with:  python examples/example_usage.py

Author: Simone Porcu
"""

from pathlib import Path

from pamos3d import Contract, D2AN, PAMOS3DEngine, STRATEGIES

HERE = Path(__file__).parent


def main():
    contract = Contract.load(HERE / "pamos3d_reference_contract.json")
    model = D2AN.from_pretrained("pointcloud-v1")
    engine = PAMOS3DEngine(model)

    print(f"Loaded contract v{contract.document_version} with "
          f"{len(contract.objects)} object(s), "
          f"budget = {contract.runtime_context.bandwidth_mbps} Mbps\n")

    print(f"{'strategy':<12} {'scene MOS':>10} {'total Mbps':>12}  selection")
    for strategy in STRATEGIES:
        decision = engine.decide(contract, strategy=strategy)
        if not decision.feasible:
            print(f"{strategy:<12} {'--':>10} {'--':>12}  (infeasible)")
            continue
        mbps = decision.total_bitrate_bps / 1e6
        print(f"{strategy:<12} {decision.scene_mos:>10.3f} {mbps:>12.2f}  {decision.selection}")

    # --- demonstrate the version-aware control loop ---
    print("\n--- control loop demo ---")
    d1 = engine.decide(contract)
    print("call 1 (initial):        recomputed =", engine.was_recomputed(contract) is False)  # already cached now

    same = engine.decide(contract)
    print("call 2 (same contract):  cached decision reused, same selection =",
          same.selection == d1.selection)

    moved = contract.with_updated_context(bandwidth_mbps=60.0)
    d2 = engine.decide(moved)
    print("call 3 (bandwidth cut to 60 Mbps): new document_version =", moved.document_version,
          " scene_mos =", round(d2.scene_mos, 3))


if __name__ == "__main__":
    main()

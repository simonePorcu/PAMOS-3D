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

    # --- delivery demo: build a small contract from real files, then
    #     retrieve the bytes D2AN selected. The reference contract above
    #     uses placeholder content_uri values, so this part uses its own
    #     tiny, self-contained scene instead. ---
    print("\n--- delivery demo ---")
    import tempfile
    from pamos3d.adapters.pointcloud import scan_object_directory, build_contract
    from pamos3d.delivery import deliver_scene

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        obj_dir = tmp / "PC1"
        obj_dir.mkdir()
        (obj_dir / "vpcc_q1.bin").write_bytes(b"\x00" * 500_000)
        (obj_dir / "vpcc_q2.bin").write_bytes(b"\x00" * 1_500_000)
        (obj_dir / "vpcc_q3.bin").write_bytes(b"\x00" * 3_000_000)

        pc1 = scan_object_directory(obj_dir, object_id="PC1", position=(0, 0, 4.0), fps=1.0)
        small_contract = build_contract([pc1], bandwidth_mbps=30.0, scene_id="delivery-demo")

        small_decision = engine.decide(small_contract)
        report = deliver_scene(small_decision, small_contract, destination_dir=tmp / "received")

        print(f"delivered {len(report.objects)} object(s), {report.total_bytes} bytes, "
              f"{report.total_time_s:.4f}s")
        for object_id, delivered in report.objects.items():
            print(f"  {object_id}: {delivered.destination_path.name} "
                  f"({delivered.bytes_transferred} bytes, size_matches_expected="
                  f"{delivered.size_matches_expected})")


if __name__ == "__main__":
    main()

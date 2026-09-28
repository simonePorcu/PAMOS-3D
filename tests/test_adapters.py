"""
Checks the point-cloud adapter's full pipeline: files on disk ->
Representation -> SceneObject -> Contract -> D2AN decision, against the
already-verified reference contract.

Author: Simone Porcu
"""

import json
from pathlib import Path

import pytest

from pamos3d import D2AN, PAMOS3DEngine
from pamos3d.adapters.pointcloud import (
    AdapterError,
    build_contract,
    parse_encoding_method,
    parse_quality_level,
    representation_from_file,
    scan_object_directory,
)

HERE = Path(__file__).parent
REFERENCE_CONTRACT_PATH = HERE.parent / "examples" / "pamos3d_reference_contract.json"


def test_parse_encoding_method_and_level():
    assert parse_encoding_method("vpcc_q2.bin") == "V-PCC"
    assert parse_encoding_method("PC1-V-PCC-Q3.bin") == "V-PCC"
    assert parse_encoding_method("gpcc_octree_q1.ply") == "G-PCC Octree"
    assert parse_encoding_method("G-PCC-Trisoup-level2.bin") == "G-PCC Trisoup"
    assert parse_encoding_method("mesh_lod2.glb") is None

    assert parse_quality_level("vpcc_q2.bin") == 2
    assert parse_quality_level("gpcc_trisoup_quality-3.bin") == 3
    assert parse_quality_level("no_level_here.bin") is None


@pytest.fixture(scope="module")
def reference_bitrates_bps():
    """The exact bitrates (in bps) baked into the already-verified reference contract."""
    data = json.loads(REFERENCE_CONTRACT_PATH.read_text())
    out = {}
    for obj in data["objects"]:
        for rep in obj["representations"]:
            out[(obj["objectId"], rep["representationId"])] = rep["resourceProfile"]["bitrateBps"]
    return out


def test_representation_from_file_recovers_exact_bitrate(tmp_path, reference_bitrates_bps):
    # Build a dummy compressed bitstream whose size in bytes corresponds
    # exactly (at fps=1.0) to one of the reference contract's real bitrates.
    target_bps = reference_bitrates_bps[("PC1", "pc1-vpcc_q3")]
    size_bytes = target_bps // 8

    fake_file = tmp_path / "vpcc_q3.bin"
    fake_file.write_bytes(b"\x00" * size_bytes)

    rep = representation_from_file(fake_file, fps=1.0)
    assert rep.encoding_method == "V-PCC"
    assert rep.quality_setting == 3
    assert rep.resource_profile.bitrate_bps == size_bytes * 8


def test_scan_object_directory_builds_full_object(tmp_path, reference_bitrates_bps):
    obj_dir = tmp_path / "PC1"
    obj_dir.mkdir()

    names_and_bps = {
        "vpcc_q1.bin": reference_bitrates_bps[("PC1", "pc1-vpcc_q1")],
        "vpcc_q2.bin": reference_bitrates_bps[("PC1", "pc1-vpcc_q2")],
        "vpcc_q3.bin": reference_bitrates_bps[("PC1", "pc1-vpcc_q3")],
        "gpcc_octree_q1.bin": reference_bitrates_bps[("PC1", "pc1-gpcc_octree_q1")],
        "gpcc_octree_q2.bin": reference_bitrates_bps[("PC1", "pc1-gpcc_octree_q2")],
        "gpcc_octree_q3.bin": reference_bitrates_bps[("PC1", "pc1-gpcc_octree_q3")],
        "gpcc_trisoup_q1.bin": reference_bitrates_bps[("PC1", "pc1-gpcc_trisoup_q1")],
        "gpcc_trisoup_q2.bin": reference_bitrates_bps[("PC1", "pc1-gpcc_trisoup_q2")],
        "gpcc_trisoup_q3.bin": reference_bitrates_bps[("PC1", "pc1-gpcc_trisoup_q3")],
    }
    for name, bps in names_and_bps.items():
        (obj_dir / name).write_bytes(b"\x00" * (bps // 8))

    # an unrelated, unrecognizable file should be skipped, not crash the scan
    (obj_dir / "readme.txt").write_text("not a point cloud")

    obj = scan_object_directory(obj_dir, object_id="PC1", position=(0, 0, 4.0), fps=1.0)
    assert obj.object_id == "PC1"
    assert len(obj.representations) == 9  # readme.txt correctly excluded

    bitrates = {r.representation_id.replace("v-pcc", "vpcc").replace("g-pcc-octree", "gpcc_octree").replace("g-pcc-trisoup", "gpcc_trisoup"):
                r.resource_profile.bitrate_bps for r in obj.representations}
    # spot-check one value round-trips exactly
    q3 = [r for r in obj.representations if r.encoding_method == "V-PCC" and r.quality_setting == 3][0]
    assert q3.resource_profile.bitrate_bps == (names_and_bps["vpcc_q3.bin"] // 8) * 8


def test_scan_object_directory_raises_on_unrecognized_when_asked(tmp_path):
    obj_dir = tmp_path / "PC1"
    obj_dir.mkdir()
    (obj_dir / "vpcc_q1.bin").write_bytes(b"\x00" * 1000)
    (obj_dir / "mystery_file.bin").write_bytes(b"\x00" * 1000)

    with pytest.raises(AdapterError):
        scan_object_directory(obj_dir, object_id="PC1", position=(0, 0, 4.0), on_unrecognized="raise")


def test_end_to_end_matches_reference_decision(tmp_path, reference_bitrates_bps):
    """
    Build all 3 objects from real files on disk, assemble a Contract with
    build_contract(), and check the D2AN decision matches the
    already-verified reference decision.
    """
    objects = []
    positions = {"PC1": (0, 0, 4.0), "PC2": (0, 0, 6.5), "PC3": (0, 0, 10.5)}
    for object_id, position in positions.items():
        obj_dir = tmp_path / object_id
        obj_dir.mkdir()
        for (oid, rep_id), bps in reference_bitrates_bps.items():
            if oid != object_id:
                continue
            # rep_id looks like "pc1-vpcc_q3" -> filename "vpcc_q3.bin"
            fname = rep_id.split("-", 1)[1] + ".bin"
            (obj_dir / fname).write_bytes(b"\x00" * (bps // 8))
        objects.append(scan_object_directory(obj_dir, object_id=object_id, position=position, fps=1.0))

    contract = build_contract(objects, bandwidth_mbps=120.0, scene_id="adapter-e2e-test")
    contract.validate()

    model = D2AN.from_pretrained("pointcloud-v1")
    engine = PAMOS3DEngine(model)
    decision = engine.decide(contract, strategy="d2an")

    assert decision.feasible
    assert decision.selection == {
        "PC1": "pc1-vpcc_q3", "PC2": "pc2-vpcc_q3", "PC3": "pc3-vpcc_q2",
    }
    assert decision.total_bitrate_bps / 1e6 == pytest.approx(119.81, abs=0.05)

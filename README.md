# pamos3d

Decision-engine library for the PAMOS-3D streaming framework: loads a
Perceptual Representation Contract Manifest (PRCM), runs D2AN to predict
perceptual quality (continuous MOS) for every candidate representation,
and performs the joint scene allocation under a bandwidth budget.

This package is a decision engine, not a media server: it tells you
which representation to send for each object, given a PRCM. Delivering
the actual point-cloud segments over the network is left to your own
transport/CDN layer.

**Author:** Simone Porcu ([simone.porcu@unica.it](mailto:simone.porcu@unica.it))

## Install

```bash
pip install -e .
```

## Quick start

```python
from pamos3d import Contract, D2AN, PAMOS3DEngine

contract = Contract.load("scene.prcm.json")
model = D2AN.from_pretrained("pointcloud-v1")
engine = PAMOS3DEngine(model)

decision = engine.decide(contract)
print(decision.selection)          # {'PC1': 'pc1-vpcc_q3', ...}
print(decision.scene_mos)          # 4.74
print(decision.total_bitrate_bps)  # 119810000
```

Comparison strategies are one keyword away:

```python
for strategy in ("d2an", "max-bitrate", "greedy", "uniform", "hybrid"):
    d = engine.decide(contract, strategy=strategy)
    print(strategy, d.scene_mos, d.total_bitrate_bps)
```

## Control loop

`PAMOS3DEngine.decide` is meant to be called once per control interval
with an updated contract snapshot. It only re-invokes D2AN when
something relevant to the decision (bandwidth, or a visible object's
distance) actually changed:

```python
updated = contract.with_updated_context(bandwidth_mbps=95.0)
decision = engine.decide(updated)   # recomputed: bandwidth changed
same_again = engine.decide(updated) # cached: nothing changed since last call
```

## Building a PRCM from real point-cloud files

`pamos3d.adapters.pointcloud` turns already-compressed bitstreams on
disk into a valid contract, so the JSON never has to be hand-written:

```python
from pamos3d.adapters.pointcloud import scan_object_directory, build_contract

# pointclouds/PC1/{vpcc_q1.bin, vpcc_q2.bin, ..., gpcc_trisoup_q3.bin}
pc1 = scan_object_directory("pointclouds/PC1", object_id="PC1", position=(0, 0, 4.0), fps=1.0)
pc2 = scan_object_directory("pointclouds/PC2", object_id="PC2", position=(0, 0, 6.5), fps=1.0)

contract = build_contract([pc1, pc2], bandwidth_mbps=120.0, scene_id="my-scene")
contract.save("scene.prcm.json")
```

Encoding method and quality level are parsed from each file name (any of
`vpcc`, `gpcc_octree`, `gpcc_trisoup` combined with `q<N>`); bitrate is
computed from the file's real byte size. Files that don't match a
recognized pattern are skipped by default, or raise with
`on_unrecognized="raise"`. For a single file instead of a whole
directory, use `representation_from_file(...)` directly.

Only the point-cloud adapter is implemented today; a Gaussian Splat or
mesh adapter would live at `pamos3d.adapters.<format>` with the same
kind of functions, feeding a D2AN instance retrained for that format.

## Package layout

| Module              | Responsibility |
|----------------------|----------------|
| `pamos3d.contract`   | PRCM dataclasses, `Contract.load/save/validate`, `with_updated_context` |
| `pamos3d.model`      | `D2AN`: loads a pretrained bundle, predicts MOS for a candidate + distance |
| `pamos3d.allocator`  | `allocate()`: D2AN / Max-bitrate / Greedy / Uniform / Hybrid strategies |
| `pamos3d.engine`     | `PAMOS3DEngine`: the version-aware control loop |
| `pamos3d.adapters.pointcloud` | Builds `Representation`/`SceneObject`/`Contract` from real compressed point-cloud files |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

11 tests cover contract validation, all five allocation strategies, the
version-aware caching logic, and the full file-to-decision adapter
pipeline. Only the point-cloud instance (V-PCC, G-PCC Octree, G-PCC
Trisoup) is currently trained and bundled; other formats need their own
adapter and a retrained D2AN bundle, with no change to `Contract`,
`allocator`, or `engine`.

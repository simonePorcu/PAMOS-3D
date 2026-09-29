"""
Checks pamos3d.delivery against both a local file path and a real HTTP
transfer (served by a local http.server thread, no external network
needed).

Author: Simone Porcu
"""

import http.server
import threading
from pathlib import Path

import pytest

from pamos3d import D2AN, PAMOS3DEngine
from pamos3d.contract import Contract
from pamos3d.delivery import DeliveryError, deliver_object, deliver_scene


@pytest.fixture()
def small_contract(tmp_path):
    """A 1-object, 1-representation contract pointing at a real local file."""
    content_file = tmp_path / "pc1_vpcc_q2.bin"
    payload = b"\x42" * 1000  # 1000 bytes -> 8000 bits -> bitrate_bps at fps=1.0
    content_file.write_bytes(payload)

    contract_dict = {
        "contract": "PRCM",
        "schemaVersion": "1.0.0",
        "documentVersion": 1,
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
        "controlIntervalMs": 1000,
        "scene": {"sceneId": "test-scene", "coordinateSystem": "right-handed-y-up", "baseUrls": []},
        "runtimeContext": {"bandwidthMbps": 50, "viewerPosition": [0, 0, 0]},
        "objects": [
            {
                "objectId": "PC1",
                "label": "Point Cloud 1",
                "position": [0, 0, 4.0],
                "importance": 1.0,
                "visibilityPolicy": {"visible": True, "deferAllowed": True, "deferUtility": 0.0},
                "representations": [
                    {
                        "representationId": "pc1-vpcc_q2",
                        "format": "point-cloud",
                        "encodingMethod": "V-PCC",
                        "qualitySetting": 2,
                        "contentUri": str(content_file),
                        "dependencies": [],
                        "resourceProfile": {
                            "bitrateBps": len(payload) * 8,
                            "decodingTimeMs": None,
                            "renderingCostMs": None,
                            "memoryMB": None,
                            "energyJ": None,
                            "switchingCostMs": 0,
                        },
                        "extensions": {},
                    }
                ],
                "extensions": {},
            }
        ],
        "extensions": {},
    }
    return Contract.from_dict(contract_dict), payload


def test_deliver_object_from_local_file(tmp_path, small_contract):
    contract, payload = small_contract
    dest = tmp_path / "out"
    delivered = deliver_object(contract, "PC1", "pc1-vpcc_q2", dest)

    assert delivered.bytes_transferred == len(payload)
    assert delivered.destination_path.read_bytes() == payload
    assert delivered.size_matches_expected is True


def test_deliver_object_missing_representation_raises(tmp_path, small_contract):
    contract, _ = small_contract
    with pytest.raises(DeliveryError):
        deliver_object(contract, "PC1", "does-not-exist", tmp_path / "out")


def test_deliver_scene_end_to_end(tmp_path, small_contract):
    contract, payload = small_contract
    model = D2AN.from_pretrained("pointcloud-v1")
    engine = PAMOS3DEngine(model)
    decision = engine.decide(contract, strategy="d2an")
    assert decision.feasible

    report = deliver_scene(decision, contract, tmp_path / "out")
    assert report.total_bytes == len(payload)
    assert "PC1" in report.objects
    assert report.objects["PC1"].destination_path.exists()


def test_deliver_scene_rejects_infeasible_decision(tmp_path, small_contract):
    contract, _ = small_contract
    from pamos3d.allocator import SceneDecision
    infeasible = SceneDecision(document_version=1, strategy="d2an", feasible=False)
    with pytest.raises(DeliveryError):
        deliver_scene(infeasible, contract, tmp_path / "out")


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep test output quiet


@pytest.fixture()
def http_server(tmp_path):
    """A local HTTP server serving tmp_path, for testing the http:// fetch path."""
    handler = lambda *args, **kwargs: _Handler(*args, directory=str(tmp_path), **kwargs)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join(timeout=2)


def test_deliver_object_over_http(tmp_path, http_server):
    payload = b"\x99" * 2000
    (tmp_path / "pc1_vpcc_q3.bin").write_bytes(payload)

    contract_dict = {
        "contract": "PRCM",
        "schemaVersion": "1.0.0",
        "documentVersion": 1,
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
        "controlIntervalMs": 1000,
        "scene": {"sceneId": "test-scene", "coordinateSystem": "right-handed-y-up", "baseUrls": []},
        "runtimeContext": {"bandwidthMbps": 50, "viewerPosition": [0, 0, 0]},
        "objects": [
            {
                "objectId": "PC1",
                "label": "Point Cloud 1",
                "position": [0, 0, 4.0],
                "representations": [
                    {
                        "representationId": "pc1-vpcc_q3",
                        "format": "point-cloud",
                        "encodingMethod": "V-PCC",
                        "qualitySetting": 3,
                        "contentUri": f"{http_server}/pc1_vpcc_q3.bin",
                        "resourceProfile": {"bitrateBps": len(payload) * 8, "switchingCostMs": 0},
                    }
                ],
            }
        ],
    }
    contract = Contract.from_dict(contract_dict)

    dest = tmp_path / "received"
    delivered = deliver_object(contract, "PC1", "pc1-vpcc_q3", dest)

    assert delivered.bytes_transferred == len(payload)
    assert delivered.destination_path.read_bytes() == payload
    assert delivered.source_uri.startswith("http://")

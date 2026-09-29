"""
pamos3d.delivery

Transfers the bytes of a scene's selected representations to a local
destination, given a SceneDecision and the Contract it was computed
from.

This resolves each selected representation's content_uri (a local file
path, a file:// URL, or an http(s):// URL) and retrieves its bytes. It
does not decode or render anything: the retrieved files are handed off
as-is, decoding and rendering are the responsibility of the consuming
application and its target device.

Author: Simone Porcu
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urlparse

from .allocator import SceneDecision
from .contract import Contract, Representation


class DeliveryError(RuntimeError):
    """Raised when a selected representation's content cannot be retrieved."""


@dataclass
class DeliveredObject:
    """The result of delivering one object's selected representation."""

    object_id: str
    representation_id: str
    source_uri: str
    destination_path: Path
    bytes_transferred: int
    expected_bitrate_bps: int
    transfer_time_s: float
    size_matches_expected: bool

    def to_dict(self) -> dict:
        return {
            "objectId": self.object_id,
            "representationId": self.representation_id,
            "sourceUri": self.source_uri,
            "destinationPath": str(self.destination_path),
            "bytesTransferred": self.bytes_transferred,
            "expectedBitrateBps": self.expected_bitrate_bps,
            "transferTimeS": self.transfer_time_s,
            "sizeMatchesExpected": self.size_matches_expected,
        }


@dataclass
class DeliveryReport:
    """The result of delivering a complete scene."""

    document_version: int
    objects: Dict[str, DeliveredObject] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(o.bytes_transferred for o in self.objects.values())

    @property
    def total_time_s(self) -> float:
        return sum(o.transfer_time_s for o in self.objects.values())

    def to_dict(self) -> dict:
        return {
            "documentVersion": self.document_version,
            "totalBytes": self.total_bytes,
            "totalTimeS": self.total_time_s,
            "objects": {oid: o.to_dict() for oid, o in self.objects.items()},
        }


def _find_representation(contract: Contract, object_id: str, representation_id: str) -> Representation:
    for obj in contract.objects:
        if obj.object_id != object_id:
            continue
        for rep in obj.representations:
            if rep.representation_id == representation_id:
                return rep
        raise DeliveryError(f"object {object_id!r} has no representation {representation_id!r}")
    raise DeliveryError(f"contract has no object {object_id!r}")


def _fetch(uri: str, destination: Path, timeout_s: float) -> int:
    """Retrieve uri's bytes into destination, return bytes written."""
    parsed = urlparse(uri)

    if parsed.scheme in ("http", "https"):
        with urllib.request.urlopen(uri, timeout=timeout_s) as response:
            data = response.read()
        destination.write_bytes(data)
        return len(data)

    if parsed.scheme == "file":
        source_path = Path(parsed.path)
    elif parsed.scheme == "":
        source_path = Path(uri)
    else:
        raise DeliveryError(f"unsupported URI scheme {parsed.scheme!r} in {uri!r}")

    if not source_path.exists():
        raise DeliveryError(f"source file not found: {source_path}")
    data = source_path.read_bytes()
    destination.write_bytes(data)
    return len(data)


def deliver_object(
    contract: Contract,
    object_id: str,
    representation_id: str,
    destination_dir: Union[str, Path],
    timeout_s: float = 10.0,
    fps: float = 1.0,
    size_tolerance: float = 0.05,
) -> DeliveredObject:
    """
    Retrieve one selected representation's bytes into destination_dir.

    fps and size_tolerance control the sanity check against the
    representation's declared bitrate_bps (computed elsewhere as
    file_size_bytes * 8 * fps): if the retrieved byte count implies a
    bitrate more than size_tolerance away from the declared one, the
    result is still returned but size_matches_expected is False.
    """
    rep = _find_representation(contract, object_id, representation_id)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(urlparse(rep.content_uri).path).suffix or ".bin"
    destination_path = destination_dir / f"{object_id}_{representation_id}{suffix}"

    start = time.monotonic()
    bytes_transferred = _fetch(rep.content_uri, destination_path, timeout_s)
    elapsed = time.monotonic() - start

    implied_bitrate_bps = bytes_transferred * 8 * fps
    expected = rep.resource_profile.bitrate_bps
    matches = expected == 0 or abs(implied_bitrate_bps - expected) <= size_tolerance * expected

    return DeliveredObject(
        object_id=object_id,
        representation_id=representation_id,
        source_uri=rep.content_uri,
        destination_path=destination_path,
        bytes_transferred=bytes_transferred,
        expected_bitrate_bps=expected,
        transfer_time_s=elapsed,
        size_matches_expected=matches,
    )


def deliver_scene(
    decision: SceneDecision,
    contract: Contract,
    destination_dir: Union[str, Path],
    timeout_s: float = 10.0,
    fps: float = 1.0,
    size_tolerance: float = 0.05,
) -> DeliveryReport:
    """
    Retrieve the bytes of every selected representation in decision into
    destination_dir. Raises DeliveryError on the first object whose
    content cannot be retrieved; objects already fetched keep their
    files on disk.
    """
    if not decision.feasible:
        raise DeliveryError("cannot deliver an infeasible SceneDecision")

    report = DeliveryReport(document_version=decision.document_version)
    for object_id, representation_id in decision.selection.items():
        report.objects[object_id] = deliver_object(
            contract, object_id, representation_id, destination_dir,
            timeout_s=timeout_s, fps=fps, size_tolerance=size_tolerance,
        )
    return report

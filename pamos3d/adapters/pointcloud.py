"""
pamos3d.adapters.pointcloud

Point-cloud format adapter: turns compressed point-cloud files on disk
into Representation / SceneObject / Contract objects, so a PRCM does not
have to be hand-written.

This does not encode or decode point clouds. It reads the byte size of
an already-compressed bitstream and turns that, plus the encoding
method and quality level, into a representation with the correct
bitrate. Producing the compressed files is the job of your V-PCC/G-PCC
encoder.

Example directory layout:

    pointclouds/
      PC1/
        vpcc_q1.bin
        vpcc_q2.bin
        vpcc_q3.bin
        gpcc_octree_q1.bin
        ...
      PC2/
        ...

>>> from pamos3d.adapters.pointcloud import scan_object_directory, build_contract
>>> pc1 = scan_object_directory("pointclouds/PC1", object_id="PC1", position=(0, 0, 4.0), fps=1.0)
>>> pc2 = scan_object_directory("pointclouds/PC2", object_id="PC2", position=(0, 0, 6.5), fps=1.0)
>>> contract = build_contract([pc1, pc2], bandwidth_mbps=120.0, scene_id="my-scene")
>>> contract.save("scene.prcm.json")

Author: Simone Porcu
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..contract import (
    Contract,
    Representation,
    ResourceProfile,
    RuntimeContext,
    SceneObject,
    VisibilityPolicy,
)


class AdapterError(ValueError):
    """Raised when a file or directory cannot be turned into a valid representation."""


# --------------------------------------------------------------------- #
# Encoding-method / quality-level name parsing
# --------------------------------------------------------------------- #

# Recognized spellings for each encoding method, all mapping to the exact
# string D2AN's calibration domain expects. Matching is done on a
# lowercased, punctuation-stripped version of the filename, so
# "V-PCC", "vpcc", "v_pcc" are all recognized as the same method.
_METHOD_ALIASES: Dict[str, str] = {
    "vpcc": "V-PCC",
    "gpccoctree": "G-PCC Octree",
    "gpcctrisoup": "G-PCC Trisoup",
}

_LEVEL_RE = re.compile(r"q(?:uality)?[_-]?(\d+)", re.IGNORECASE)


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def parse_encoding_method(text: str) -> Optional[str]:
    """
    Recover a canonical encoding-method name (``"V-PCC"``, ``"G-PCC Octree"``,
    ``"G-PCC Trisoup"``) from a free-form string (typically a filename).
    Returns ``None`` if no known method is recognized.
    """
    n = _normalize(text)
    for alias, canonical in _METHOD_ALIASES.items():
        if alias in n:
            return canonical
    return None


def parse_quality_level(text: str) -> Optional[int]:
    """Recover a quality level (an integer) from a free-form string, e.g. ``"...q2..."`` -> ``2``."""
    m = _LEVEL_RE.search(text)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------- #
# Building one representation
# --------------------------------------------------------------------- #

@dataclass
class PointCloudFile:
    """One compressed point-cloud bitstream candidate, as found on disk."""

    path: Path
    encoding_method: str
    quality_setting: int
    representation_id: Optional[str] = None
    content_uri: Optional[str] = None


def representation_from_file(
    file: Union[str, Path, PointCloudFile],
    encoding_method: Optional[str] = None,
    quality_setting: Optional[int] = None,
    fps: float = 1.0,
    representation_id: Optional[str] = None,
    content_uri: Optional[str] = None,
    format_name: str = "point-cloud",
) -> Representation:
    """
    Build one Representation from a compressed point-cloud file already
    on disk.

    Bitrate is computed from the file's byte size and fps (frames per
    second the stream is played back at): bitrate_bps = file_size_bytes
    * 8 * fps. Use fps=1.0 (the default) for a single static frame
    delivered once per control interval.

    encoding_method and quality_setting are taken from the arguments if
    given, otherwise parsed from the file name (e.g. "vpcc_q2.bin",
    "gpcc_trisoup_level-3.ply").
    """
    if isinstance(file, PointCloudFile):
        path = Path(file.path)
        encoding_method = encoding_method or file.encoding_method
        quality_setting = quality_setting if quality_setting is not None else file.quality_setting
        representation_id = representation_id or file.representation_id
        content_uri = content_uri or file.content_uri
    else:
        path = Path(file)

    if not path.exists():
        raise AdapterError(f"file not found: {path}")

    if encoding_method is None:
        encoding_method = parse_encoding_method(path.name)
        if encoding_method is None:
            raise AdapterError(
                f"could not infer an encoding method from {path.name!r}; "
                f"pass encoding_method explicitly (expected one of {sorted(set(_METHOD_ALIASES.values()))})"
            )
    if quality_setting is None:
        quality_setting = parse_quality_level(path.name)
        if quality_setting is None:
            raise AdapterError(
                f"could not infer a quality level from {path.name!r}; pass quality_setting explicitly"
            )

    size_bytes = path.stat().st_size
    bitrate_bps = int(round(size_bytes * 8 * fps))

    rep_id = representation_id or f"{_normalize(encoding_method)}_q{quality_setting}"
    uri = content_uri or str(path)

    return Representation(
        representation_id=rep_id,
        format=format_name,
        encoding_method=encoding_method,
        quality_setting=quality_setting,
        content_uri=uri,
        resource_profile=ResourceProfile(bitrate_bps=bitrate_bps),
    )


# --------------------------------------------------------------------- #
# Building a whole object from a directory of candidate files
# --------------------------------------------------------------------- #

def scan_object_directory(
    directory: Union[str, Path],
    object_id: str,
    position: Tuple[float, float, float],
    fps: float = 1.0,
    label: Optional[str] = None,
    importance: float = 1.0,
    glob: str = "*",
    on_unrecognized: str = "skip",
) -> SceneObject:
    """
    Scan a directory of compressed point-cloud files and build one
    SceneObject with one Representation per recognized file.

    Parameters
    ----------
    directory:
        Folder containing this object's candidate bitstreams, one file
        per (encoding method, quality level) combination.
    object_id, position, label, importance:
        Passed straight through to the resulting SceneObject.
    fps:
        See representation_from_file().
    glob:
        Filename pattern to scan (default: every file in the directory).
    on_unrecognized:
        "skip" (default) ignores files whose method/level cannot be
        parsed from their name; "raise" raises AdapterError instead.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise AdapterError(f"not a directory: {directory}")

    representations: List[Representation] = []
    skipped: List[str] = []
    for path in sorted(directory.glob(glob)):
        if not path.is_file():
            continue
        method = parse_encoding_method(path.name)
        level = parse_quality_level(path.name)
        if method is None or level is None:
            if on_unrecognized == "raise":
                raise AdapterError(f"could not parse encoding method/quality level from {path.name!r}")
            skipped.append(path.name)
            continue
        representations.append(
            representation_from_file(
                path, encoding_method=method, quality_setting=level, fps=fps,
                representation_id=f"{_normalize(object_id)}-{_normalize(method)}_q{level}",
            )
        )

    if not representations:
        raise AdapterError(
            f"no recognizable point-cloud candidate files found in {directory} "
            f"(skipped: {skipped})"
        )

    return SceneObject(
        object_id=object_id,
        label=label or object_id,
        position=position,
        representations=representations,
        importance=importance,
        visibility_policy=VisibilityPolicy(),
    )


# --------------------------------------------------------------------- #
# Assembling a full contract
# --------------------------------------------------------------------- #

def build_contract(
    objects: List[SceneObject],
    bandwidth_mbps: float,
    scene_id: str = "scene",
    viewer_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    control_interval_ms: int = 1000,
    coordinate_system: str = "right-handed-y-up",
    base_urls: Optional[List[str]] = None,
    viewing_conditions: Optional[dict] = None,
    device_state: Optional[dict] = None,
) -> Contract:
    """
    Assemble a ready-to-use, valid Contract from a list of adapter-built
    objects, filling in the header/scene/runtime-context fields a
    hand-written PRCM would otherwise need.
    """
    now = datetime.now(timezone.utc).isoformat()
    return Contract(
        contract="PRCM",
        schema_version="1.0.0",
        document_version=1,
        created_at=now,
        updated_at=now,
        control_interval_ms=control_interval_ms,
        scene={
            "sceneId": scene_id,
            "coordinateSystem": coordinate_system,
            "baseUrls": base_urls or [],
        },
        runtime_context=RuntimeContext(
            bandwidth_mbps=bandwidth_mbps,
            viewer_position=viewer_position,
            viewing_conditions=viewing_conditions or {},
            device_state=device_state or {},
        ),
        objects=objects,
    )

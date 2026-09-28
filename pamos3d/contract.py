"""
pamos3d.contract

PRCM (Perceptual Representation Contract Manifest) data model and
(de)serialization.

The JSON uses camelCase keys; the dataclasses here use snake_case.
Contract.load / Contract.save handle the conversion.

Author: Simone Porcu
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class ContractError(ValueError):
    """Raised when a document is not a structurally valid PRCM."""


# --------------------------------------------------------------------- #
# Resource profile / representation / object / runtime context
# --------------------------------------------------------------------- #

@dataclass
class ResourceProfile:
    """Bitrate and resource cost of one representation."""

    bitrate_bps: int
    decoding_time_ms: Optional[float] = None
    rendering_cost_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    energy_j: Optional[float] = None
    switching_cost_ms: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResourceProfile":
        return cls(
            bitrate_bps=int(d["bitrateBps"]),
            decoding_time_ms=d.get("decodingTimeMs"),
            rendering_cost_ms=d.get("renderingCostMs"),
            memory_mb=d.get("memoryMB"),
            energy_j=d.get("energyJ"),
            switching_cost_ms=float(d.get("switchingCostMs", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bitrateBps": self.bitrate_bps,
            "decodingTimeMs": self.decoding_time_ms,
            "renderingCostMs": self.rendering_cost_ms,
            "memoryMB": self.memory_mb,
            "energyJ": self.energy_j,
            "switchingCostMs": self.switching_cost_ms,
        }


@dataclass
class Representation:
    """One selectable candidate representation of an object."""

    representation_id: str
    format: str
    encoding_method: str
    quality_setting: int
    content_uri: str
    resource_profile: ResourceProfile
    dependencies: List[str] = field(default_factory=list)
    extensions: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Representation":
        return cls(
            representation_id=d["representationId"],
            format=d.get("format", ""),
            encoding_method=d["encodingMethod"],
            quality_setting=int(d["qualitySetting"]),
            content_uri=d.get("contentUri", ""),
            resource_profile=ResourceProfile.from_dict(d["resourceProfile"]),
            dependencies=list(d.get("dependencies", [])),
            extensions=dict(d.get("extensions", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "representationId": self.representation_id,
            "format": self.format,
            "encodingMethod": self.encoding_method,
            "qualitySetting": self.quality_setting,
            "contentUri": self.content_uri,
            "dependencies": list(self.dependencies),
            "resourceProfile": self.resource_profile.to_dict(),
            "extensions": dict(self.extensions),
        }


@dataclass
class VisibilityPolicy:
    """Whether an object is currently visible and how to handle deferral."""

    visible: bool = True
    defer_allowed: bool = True
    defer_utility: float = 0.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "VisibilityPolicy":
        d = d or {}
        return cls(
            visible=bool(d.get("visible", True)),
            defer_allowed=bool(d.get("deferAllowed", True)),
            defer_utility=float(d.get("deferUtility", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visible": self.visible,
            "deferAllowed": self.defer_allowed,
            "deferUtility": self.defer_utility,
        }


@dataclass
class SceneObject:
    """One object in the scene and its candidate representations."""

    object_id: str
    label: str
    position: Tuple[float, float, float]
    representations: List[Representation]
    importance: float = 1.0
    visibility_policy: VisibilityPolicy = field(default_factory=VisibilityPolicy)
    extensions: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneObject":
        pos = d.get("position", [0.0, 0.0, 0.0])
        if len(pos) != 3:
            raise ContractError(f"object {d.get('objectId')!r}: 'position' must have 3 components")
        return cls(
            object_id=d["objectId"],
            label=d.get("label", d["objectId"]),
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            representations=[Representation.from_dict(r) for r in d.get("representations", [])],
            importance=float(d.get("importance", 1.0)),
            visibility_policy=VisibilityPolicy.from_dict(d.get("visibilityPolicy")),
            extensions=dict(d.get("extensions", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objectId": self.object_id,
            "label": self.label,
            "position": list(self.position),
            "importance": self.importance,
            "visibilityPolicy": self.visibility_policy.to_dict(),
            "representations": [r.to_dict() for r in self.representations],
            "extensions": dict(self.extensions),
        }

    def distance_to(self, viewer_position: Tuple[float, float, float]) -> float:
        """Euclidean distance from this object to a viewer position."""
        dx = self.position[0] - viewer_position[0]
        dy = self.position[1] - viewer_position[1]
        dz = self.position[2] - viewer_position[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5


@dataclass
class RuntimeContext:
    """Current bandwidth, viewer position, and device state."""

    bandwidth_mbps: float
    viewer_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    viewing_conditions: Dict[str, Any] = field(default_factory=dict)
    device_state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "RuntimeContext":
        d = d or {}
        pos = d.get("viewerPosition", [0.0, 0.0, 0.0])
        return cls(
            bandwidth_mbps=float(d.get("bandwidthMbps", 0.0)),
            viewer_position=(float(pos[0]), float(pos[1]), float(pos[2])),
            viewing_conditions=dict(d.get("viewingConditions", {})),
            device_state=dict(d.get("deviceState", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bandwidthMbps": self.bandwidth_mbps,
            "viewerPosition": list(self.viewer_position),
            "viewingConditions": dict(self.viewing_conditions),
            "deviceState": dict(self.device_state),
        }


# --------------------------------------------------------------------- #
# The contract itself
# --------------------------------------------------------------------- #

@dataclass
class Contract:
    """
    A PRCM document. Construct with load()/from_dict(); write back out
    with save()/to_dict().
    """

    schema_version: str
    document_version: int
    created_at: str
    updated_at: str
    control_interval_ms: int
    scene: Dict[str, Any]
    runtime_context: RuntimeContext
    objects: List[SceneObject]
    extensions: Dict[str, Any] = field(default_factory=dict)
    contract: str = "PRCM"

    # -- construction -------------------------------------------------- #

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Contract":
        if d.get("contract") != "PRCM":
            raise ContractError(
                f"missing or invalid 'contract' field (expected 'PRCM', got {d.get('contract')!r})"
            )
        if "objects" not in d or not isinstance(d["objects"], list) or not d["objects"]:
            raise ContractError("no 'objects' array found")

        return cls(
            contract="PRCM",
            schema_version=d.get("schemaVersion", "1.0.0"),
            document_version=int(d.get("documentVersion", 1)),
            created_at=d.get("createdAt", ""),
            updated_at=d.get("updatedAt", ""),
            control_interval_ms=int(d.get("controlIntervalMs", 1000)),
            scene=dict(d.get("scene", {})),
            runtime_context=RuntimeContext.from_dict(d.get("runtimeContext")),
            objects=[SceneObject.from_dict(o) for o in d["objects"]],
            extensions=dict(d.get("extensions", {})),
        )

    @classmethod
    def load(cls, path_or_dict: Union[str, Path, Dict[str, Any]]) -> "Contract":
        """Load a PRCM from a file path, a JSON string's parsed dict, or a dict."""
        if isinstance(path_or_dict, dict):
            data = path_or_dict
        else:
            data = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
        contract = cls.from_dict(data)
        contract.validate()
        return contract

    # -- validation ------------------------------------------------------ #

    def validate(self) -> None:
        """Raise :class:`ContractError` if the document is structurally invalid."""
        if self.contract != "PRCM":
            raise ContractError("contract field must be 'PRCM'")
        if not self.objects:
            raise ContractError("a contract must declare at least one object")
        seen_ids = set()
        for obj in self.objects:
            if obj.object_id in seen_ids:
                raise ContractError(f"duplicate objectId: {obj.object_id!r}")
            seen_ids.add(obj.object_id)
            if not obj.representations:
                raise ContractError(f"object {obj.object_id!r} has no representations")
            rep_ids = set()
            for rep in obj.representations:
                if rep.representation_id in rep_ids:
                    raise ContractError(
                        f"object {obj.object_id!r}: duplicate representationId {rep.representation_id!r}"
                    )
                rep_ids.add(rep.representation_id)
                if rep.resource_profile.bitrate_bps < 0:
                    raise ContractError(
                        f"representation {rep.representation_id!r}: bitrateBps must be >= 0"
                    )

    # -- serialization ----------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "schemaVersion": self.schema_version,
            "documentVersion": self.document_version,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "controlIntervalMs": self.control_interval_ms,
            "scene": dict(self.scene),
            "runtimeContext": self.runtime_context.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
            "extensions": dict(self.extensions),
        }

    def save(self, path: Union[str, Path], indent: int = 2) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")

    # -- convenience: building an updated snapshot ------------------------- #

    def with_updated_context(
        self,
        bandwidth_mbps: Optional[float] = None,
        viewer_position: Optional[Tuple[float, float, float]] = None,
        object_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
        visible_object_ids: Optional[set] = None,
    ) -> "Contract":
        """
        Return a new Contract with an updated runtime context and/or
        object positions/visibility. Bumps document_version and updated_at.
        """
        new_ctx = RuntimeContext(
            bandwidth_mbps=self.runtime_context.bandwidth_mbps if bandwidth_mbps is None else bandwidth_mbps,
            viewer_position=self.runtime_context.viewer_position if viewer_position is None else viewer_position,
            viewing_conditions=dict(self.runtime_context.viewing_conditions),
            device_state=dict(self.runtime_context.device_state),
        )
        new_objects = []
        for obj in self.objects:
            pos = obj.position
            if object_positions is not None and obj.object_id in object_positions:
                pos = object_positions[obj.object_id]
            vis = obj.visibility_policy
            if visible_object_ids is not None:
                vis = VisibilityPolicy(
                    visible=obj.object_id in visible_object_ids,
                    defer_allowed=vis.defer_allowed,
                    defer_utility=vis.defer_utility,
                )
            new_objects.append(
                SceneObject(
                    object_id=obj.object_id,
                    label=obj.label,
                    position=pos,
                    representations=obj.representations,
                    importance=obj.importance,
                    visibility_policy=vis,
                    extensions=obj.extensions,
                )
            )
        return Contract(
            contract="PRCM",
            schema_version=self.schema_version,
            document_version=self.document_version + 1,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            control_interval_ms=self.control_interval_ms,
            scene=dict(self.scene),
            runtime_context=new_ctx,
            objects=new_objects,
            extensions=dict(self.extensions),
        )

    def visible_objects(self) -> List[SceneObject]:
        """Objects whose visibility policy currently allows allocation."""
        return [o for o in self.objects if o.visibility_policy.visible]

    def signature(self) -> str:
        """
        A string capturing bandwidth and each visible object's distance.
        Two contracts with the same signature produce the same decision;
        used by PAMOS3DEngine to skip recomputation when nothing changed.
        """
        viewer = self.runtime_context.viewer_position
        parts = [f"{self.runtime_context.bandwidth_mbps:.3f}"]
        for obj in sorted(self.visible_objects(), key=lambda o: o.object_id):
            parts.append(f"{obj.object_id}@{obj.distance_to(viewer):.3f}")
        return "|".join(parts)

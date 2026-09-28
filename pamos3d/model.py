"""
pamos3d.model

D2AN (Distance-conditioned Differentiable Allocator Network): the
point-cloud perceptual-quality model.

Inference only needs numpy. A pretrained bundle packages both the
trained weights and the fitted feature encoders, so from_pretrained()
does not need the original training dataset.

Author: Simone Porcu
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from .contract import Representation


class ModelError(RuntimeError):
    """Raised for bundle-loading or inference errors."""


_BUNDLED_INSTANCES = {
    "pointcloud-v1": "pamos3d_pointcloud_v1_bundle.json",
}


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # weight has PyTorch's (out_features, in_features) layout.
    return x @ weight.T + bias


@dataclass
class D2ANConfig:
    embed_dim: int
    hidden_dim: int
    k: int


def _normalize_encoding_method(raw: str) -> str:
    """
    Map an encodingMethod string onto its canonical name, tolerating
    punctuation/case differences ("G-PCC-Trisoup", "gpcc_trisoup" and
    "G-PCC Trisoup" are the same method).
    """
    n = "".join(ch for ch in raw.lower() if ch.isalnum())
    if n == "vpcc":
        return "V-PCC"
    if n == "gpccoctree":
        return "G-PCC Octree"
    if n == "gpcctrisoup":
        return "G-PCC Trisoup"
    return raw


class D2AN:
    """
    A trained D2AN instance for one 3D-format adapter. This class wraps
    the point-cloud instance.

    Only the utility head's forward pass is used at inference time; the
    ordinal and selector heads are used during training only.
    """

    def __init__(
        self,
        weights: Dict[str, np.ndarray],
        method2id: Dict[str, int],
        encoders: Dict[str, float],
        config: D2ANConfig,
        distance_weight_params: Dict[str, float],
    ):
        self._w = weights
        self._method2id = dict(method2id)
        self._enc = dict(encoders)
        self._config = config
        self._d0 = float(distance_weight_params.get("d0", 1.5))
        self._alpha = float(distance_weight_params.get("alpha", 2.0))

    # -- construction ------------------------------------------------- #

    @classmethod
    def from_pretrained(cls, name: str = "pointcloud-v1") -> "D2AN":
        """Load one of the bundles shipped with this package."""
        if name not in _BUNDLED_INSTANCES:
            raise ModelError(
                f"unknown pretrained instance {name!r}; available: {list(_BUNDLED_INSTANCES)}"
            )
        bundle_path = resources.files(__package__).joinpath(_BUNDLED_INSTANCES[name])
        with bundle_path.open("r", encoding="utf-8") as f:
            return cls.from_bundle_dict(json.load(f))

    @classmethod
    def from_bundle_file(cls, path: Union[str, Path]) -> "D2AN":
        """Load a bundle produced by the training pipeline from an arbitrary path."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_bundle_dict(json.load(f))

    @classmethod
    def from_bundle_dict(cls, bundle: Dict) -> "D2AN":
        if bundle.get("format") != "pamos3d-d2an-bundle":
            raise ModelError("not a pamos3d D2AN bundle (missing/invalid 'format' field)")
        weights = {k: np.array(v, dtype=np.float64) for k, v in bundle["weights"].items()}
        config = D2ANConfig(
            embed_dim=bundle["architecture"]["embed_dim"],
            hidden_dim=bundle["architecture"]["hidden_dim"],
            k=bundle["architecture"]["K"],
        )
        return cls(
            weights=weights,
            method2id=bundle["method2id"],
            encoders=bundle["encoders"],
            config=config,
            distance_weight_params=bundle.get("distance_weight", {}),
        )

    # -- feature encoding ---------------------------------------------- #

    def _encode(self, encoding_method: str, quality_setting: int, bitrate_bps: float, distance_m: float):
        method_name = _normalize_encoding_method(encoding_method)
        if method_name not in self._method2id:
            raise ModelError(
                f"encoding method {encoding_method!r} is not in this D2AN instance's "
                f"calibration domain ({sorted(self._method2id)})"
            )
        method_id = self._method2id[method_name]

        level_min, level_max = self._enc["level_min"], self._enc["level_max"]
        level_norm = (quality_setting - level_min) / max(1e-6, level_max - level_min)

        br_mean = self._enc["br_scaler_mean"][0]
        br_scale = self._enc["br_scaler_scale"][0]
        br_norm = (np.log1p(bitrate_bps) - br_mean) / br_scale

        d_mean = self._enc["dist_scaler_mean"][0]
        d_scale = self._enc["dist_scaler_scale"][0]
        dist_norm = (distance_m - d_mean) / d_scale

        return method_id, level_norm, br_norm, dist_norm

    def _forward_utility(self, method_id: int, level_norm: float, br_norm: float, dist_norm: float) -> float:
        emb = self._w["emb_method.weight"][method_id]
        x = np.concatenate([emb, [level_norm, br_norm, dist_norm]])
        h = _relu(_linear(x, self._w["backbone.0.weight"], self._w["backbone.0.bias"]))
        h = _relu(_linear(h, self._w["backbone.2.weight"], self._w["backbone.2.bias"]))
        hu = _relu(_linear(h, self._w["head_u.0.weight"], self._w["head_u.0.bias"]))
        u = _linear(hu, self._w["head_u.2.weight"], self._w["head_u.2.bias"])[0]
        return float(_sigmoid(u))

    def distance_weight(self, distance_m: float) -> float:
        """Importance weight from viewing distance: closer objects weigh more."""
        return 1.0 / (1.0 + (distance_m / self._d0) ** self._alpha)

    # -- public inference API ------------------------------------------ #

    def predict_mos(self, representation: Representation, distance_m: float) -> float:
        """Continuous MOS in [1, 5] for one candidate at the given viewing distance."""
        method_id, level_norm, br_norm, dist_norm = self._encode(
            representation.encoding_method,
            representation.quality_setting,
            representation.resource_profile.bitrate_bps,
            distance_m,
        )
        u = self._forward_utility(method_id, level_norm, br_norm, dist_norm)
        return 1.0 + 4.0 * u

    def predict_utility(self, representation: Representation, distance_m: float) -> float:
        """Normalized utility in [0, 1], before the MOS rescale."""
        return (self.predict_mos(representation, distance_m) - 1.0) / 4.0

    def predict_batch(self, representations: List[Representation], distance_m: float) -> List[float]:
        """predict_mos() for several candidates of the same object."""
        return [self.predict_mos(r, distance_m) for r in representations]

    def supports(self, encoding_method: str) -> bool:
        """Whether this instance's calibration domain covers a given encoding method."""
        return _normalize_encoding_method(encoding_method) in self._method2id

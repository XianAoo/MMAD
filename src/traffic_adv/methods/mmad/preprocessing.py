from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from traffic_adv.data.schema import TrafficFlow


class MMADPreprocessor:
    def __init__(self, sequence_length: int = 32, quantile: float = 0.99):
        self.sequence_length = int(sequence_length)
        self.quantile = float(quantile)
        self.clip_upper: np.ndarray | None = None
        self.data_min: np.ndarray | None = None
        self.data_max: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, flows: Iterable[TrafficFlow]) -> "MMADPreprocessor":
        values = self._raw_feature_tensor(list(flows))
        if len(values) == 0:
            raise ValueError("Cannot fit MMAD preprocessor on an empty dataset")
        self.clip_upper = np.percentile(values, self.quantile * 100.0, axis=0, keepdims=True)
        clipped = np.minimum(np.maximum(values, 0.0), self.clip_upper)
        logged = np.log1p(clipped)
        self.data_min = logged.min(axis=0, keepdims=True)
        self.data_max = logged.max(axis=0, keepdims=True)
        self.is_fitted = True
        return self

    def transform(self, flows: Iterable[TrafficFlow]) -> np.ndarray:
        self._require_fitted()
        flows = list(flows)
        values = self._raw_feature_tensor(flows)
        clipped = np.minimum(np.maximum(values, 0.0), self.clip_upper)
        logged = np.log1p(clipped)
        denom = np.maximum(self.data_max - self.data_min, 1e-8)
        normalized = np.clip((logged - self.data_min) / denom, 0.0, 1.0)
        mask = self._mask_tensor(flows)
        return np.concatenate([normalized, mask], axis=1).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_fitted()
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        features = np.clip(values[:, :2, :], 0.0, 1.0)
        denom = np.maximum(self.data_max - self.data_min, 1e-8)
        logged = features * denom + self.data_min
        raw = np.expm1(logged)
        raw = np.minimum(np.maximum(raw, 0.0), self.clip_upper)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        masks = np.where(values[:, 2:3, :] > 0.5, 1.0, 0.0)
        return raw[:, 0, :], raw[:, 1, :], masks

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "MMADPreprocessor":
        with Path(path).open("rb") as handle:
            value = _LegacyMMADPreprocessorUnpickler(handle, cls).load()
        if not isinstance(value, cls):
            value = cls._from_legacy_object(value)
        return value

    @classmethod
    def _from_legacy_object(cls, value) -> "MMADPreprocessor":
        required = ("sequence_length", "quantile", "clip_upper", "data_min", "data_max")
        if not all(hasattr(value, name) for name in required):
            fields = sorted(getattr(value, "__dict__", {}).keys())
            raise TypeError(
                f"Unexpected MMAD preprocessor type: {type(value).__name__}; "
                f"available fields: {fields}"
            )
        restored = cls(
            sequence_length=getattr(value, "sequence_length"),
            quantile=getattr(value, "quantile"),
        )
        restored.clip_upper = getattr(value, "clip_upper")
        restored.data_min = getattr(value, "data_min")
        restored.data_max = getattr(value, "data_max")
        restored.is_fitted = bool(getattr(value, "is_fitted", True))
        return restored

    def _raw_feature_tensor(self, flows: list[TrafficFlow]) -> np.ndarray:
        values = np.zeros((len(flows), 2, self.sequence_length), dtype=np.float32)
        for flow_index, flow in enumerate(flows):
            for packet_index, packet in enumerate(flow.packets[: self.sequence_length]):
                values[flow_index, 0, packet_index] = max(float(packet.iat_ms), 0.0)
                values[flow_index, 1, packet_index] = max(float(packet.packet_size), 0.0)
        return values

    def _mask_tensor(self, flows: list[TrafficFlow]) -> np.ndarray:
        masks = np.zeros((len(flows), 1, self.sequence_length), dtype=np.float32)
        for flow_index, flow in enumerate(flows):
            valid_length = min(len(flow.packets), self.sequence_length)
            masks[flow_index, 0, :valid_length] = 1.0
        return masks

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Fit or load the MMAD preprocessor before transforming data")


class _LegacyMMADPreprocessorUnpickler(pickle.Unpickler):
    def __init__(self, handle, preprocessor_cls):
        super().__init__(handle)
        self.preprocessor_cls = preprocessor_cls

    def find_class(self, module: str, name: str):
        if module in {"mmad", "mmad.preprocessing"} and name == "MMADPreprocessor":
            return self.preprocessor_cls
        if module == "mmad" or module.startswith("mmad."):
            return _LegacyMMADPickleObject
        return super().find_class(module, name)


class _LegacyMMADPickleObject:
    pass

import numpy as np
from typing import Any

from logger import logger
from core.sample import EncodedSample
from encoders.arch_encoder import ArchEncoder
from encoders.cross_encoder import CrossEncoder
from encoders.device_encoder import DeviceEncoder


class DatasetBuilder:
    def __init__(
        self,
        arch_encoder: ArchEncoder,
        device_encoder: DeviceEncoder,
        cross_encoder: CrossEncoder,
        log_transform_target: bool = True,
    ) -> None:
        self._arch_enc = arch_encoder
        self._device_enc = device_encoder
        self._cross_enc = cross_encoder
        self._log_target = log_transform_target

    def encode(self, arch_str: str, device_name: str) -> np.ndarray:
        sample = self.encode_sample(arch_str, device_name)
        return sample.feature_vector

    def encode_sample(self, arch_str: str, device_name: str) -> EncodedSample:
        arch_vec = self._arch_enc.encode(arch_str)
        device_vec = self._device_enc.encode(device_name)
        cross_vec = self._cross_enc.encode(arch_vec, device_vec)
        return EncodedSample(
            arch_vec=arch_vec,
            device_vec=device_vec,
            cross_vec=cross_vec,
            arch_str=arch_str,
            device_name=device_name,
        )

    def build(
        self,
        api: Any,
        hw_api: Any,
        device_names: list[str] | None = None,
        dataset: str = "cifar10",
        max_archs: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        registry = self._device_enc._registry
        if device_names is None:
            device_names = registry.names
        else:
            unknown = set(device_names) - set(registry.names)
            if unknown:
                raise ValueError(f"Unknown device(s): {unknown}")

        n_total = len(api) if max_archs is None else min(max_archs, len(api))
        X_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        meta: list[dict] = []

        for idx in range(n_total):
            arch_str = api.arch(idx)

            for dev in device_names:
                latency = hw_api.query_by_index_and_device(idx, dev, dataset)
                if latency is None or latency <= 0:
                    continue

                X_rows.append(self.encode(arch_str, dev))
                y_val = float(np.log1p(latency)) if self._log_target else float(latency)
                y_rows.append(y_val)
                meta.append({"arch_idx": idx, "device": dev, "arch_str": arch_str})

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.float32)
        logger.info(
            f"DatasetBuilder: {len(X_rows)} samples built ({n_total} archs x {len(device_names)} devices)."
        )
        return X, y, meta

    @property
    def feature_names(self) -> list[str]:
        """Human-readable names for all 50 output dimensions."""
        return (
            self._arch_enc.feature_names
            + self._device_enc.feature_names
            + self._cross_enc.feature_names
        )

    def __repr__(self) -> str:
        return (
            f"DatasetBuilder("
            f"arch={self._arch_enc}, "
            f"device={self._device_enc}, "
            f"cross={self._cross_enc}, "
            f"log_transform={self._log_target})"
        )

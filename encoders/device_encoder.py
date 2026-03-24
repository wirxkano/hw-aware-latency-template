import numpy as np
from sklearn.preprocessing import StandardScaler

from constants import DEVICE_CONTINUOUS_IDX, DIM_DEVICE
from registry import DeviceRegistry, DeviceSpec


class DeviceEncoder:
    def __init__(self, registry: DeviceRegistry) -> None:
        self._registry = registry
        self._scaler: StandardScaler | None = None

    def fit(self) -> "DeviceEncoder":
        specs = self._registry.all_specs
        if not specs:
            raise ValueError(
                "Registry is empty - add at least one device before fitting."
            )

        raw_cont = np.array(
            [[spec.features[i] for i in DEVICE_CONTINUOUS_IDX] for spec in specs],
            dtype=np.float64,
        )
        self._scaler = StandardScaler()
        self._scaler.fit(np.log1p(raw_cont))
        return self

    def encode(self, device_name: str) -> np.ndarray:
        self._check_fitted()
        spec = self._registry.get(device_name)
        return self._encode_spec(spec)

    def encode_spec(self, spec: DeviceSpec) -> np.ndarray:
        self._check_fitted()
        return self._encode_spec(spec)

    def encode_all(self) -> dict[str, np.ndarray]:
        return {name: self.encode(name) for name in self._registry.names}

    def update_registry(self, registry: DeviceRegistry) -> "DeviceEncoder":
        self._registry = registry
        return self.fit()

    @property
    def is_fitted(self) -> bool:
        return self._scaler is not None

    @property
    def output_dim(self) -> int:
        return DIM_DEVICE

    @property
    def feature_names(self) -> list[str]:
        return [
            "log_tops",
            "log_bw",
            "log_cache",
            "log_tdp",
            "log_cores",
            "log_clock",
            "is_gpu",
            "is_asic",
            "is_fpga",
            "is_mobile",
        ]

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"DeviceEncoder(status={status}, " f"registry={self._registry})"

    def _check_fitted(self) -> None:
        if self._scaler is None:
            raise RuntimeError("DeviceEncoder is not fitted. Call encoder.fit() first.")

    def _encode_spec(self, spec: DeviceSpec) -> np.ndarray:
        cont_raw = np.array(spec.continuous, dtype=np.float64)
        cont_norm = self._scaler.transform([np.log1p(cont_raw)])[0].astype(np.float32)
        binary = np.array(spec.binary_flags, dtype=np.float32)
        return np.concatenate([cont_norm, binary])

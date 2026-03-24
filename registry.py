from dataclasses import dataclass
from typing import Iterator

from constants import DEVICE_BINARY_IDX, DEVICE_CONTINUOUS_IDX, DEVICE_FEATURE_NAMES


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    features: tuple[float, ...]
    description: str = ""

    @classmethod
    def from_list(
        cls,
        name: str,
        features: list[float],
        description: str = "",
    ) -> "DeviceSpec":
        return cls(name=name, features=tuple(features), description=description)

    def __post_init__(self) -> None:
        if len(self.features) != len(DEVICE_FEATURE_NAMES):
            raise ValueError(
                f"DeviceSpec '{self.name}' must have "
                f"{len(DEVICE_FEATURE_NAMES)} features, "
                f"got {len(self.features)}."
            )
        for idx in DEVICE_BINARY_IDX:
            val = self.features[idx]
            if val not in (0, 1, 0.0, 1.0):
                raise ValueError(
                    f"DeviceSpec '{self.name}': binary flag at index {idx} "
                    f"must be 0 or 1, got {val}."
                )

    @property
    def tops(self) -> float:
        return self.features[0]

    @property
    def bandwidth(self) -> float:
        return self.features[1]

    @property
    def cache_mb(self) -> float:
        return self.features[2]

    @property
    def tdp_w(self) -> float:
        return self.features[3]

    @property
    def num_cores(self) -> float:
        return self.features[4]

    @property
    def clock_ghz(self) -> float:
        return self.features[5]

    @property
    def is_gpu(self) -> bool:
        return bool(self.features[6])

    @property
    def is_asic(self) -> bool:
        return bool(self.features[7])

    @property
    def is_fpga(self) -> bool:
        return bool(self.features[8])

    @property
    def is_mobile(self) -> bool:
        return bool(self.features[9])

    @property
    def continuous(self) -> tuple[float, ...]:
        return tuple(self.features[i] for i in DEVICE_CONTINUOUS_IDX)

    @property
    def binary_flags(self) -> tuple[float, ...]:
        return tuple(self.features[i] for i in DEVICE_BINARY_IDX)

    def __repr__(self) -> str:
        hw_type = (
            "GPU"
            if self.is_gpu
            else "ASIC" if self.is_asic else "FPGA" if self.is_fpga else "CPU/Mobile"
        )
        return (
            f"DeviceSpec(name='{self.name}', type={hw_type}, "
            f"tops={self.tops}, bw={self.bandwidth} GB/s, "
            f"tdp={self.tdp_w} W)"
        )


class DeviceRegistry:
    def __init__(self, initial_devices: list[DeviceSpec] | None = None) -> None:
        self._store: dict[str, DeviceSpec] = {}
        for spec in initial_devices or []:
            self.register(spec)

    def register(self, spec: DeviceSpec) -> None:
        self._store[spec.name] = spec

    def register_from_list(
        self,
        name: str,
        features: list[float],
        description: str = "",
    ) -> DeviceSpec:
        spec = DeviceSpec.from_list(name, features, description)
        self.register(spec)
        return spec

    def remove(self, name: str) -> None:
        if name not in self._store:
            raise KeyError(f"Device '{name}' not in registry.")
        del self._store[name]

    def get(self, name: str) -> DeviceSpec:
        if name not in self._store:
            raise KeyError(
                f"Device '{name}' not in registry. " f"Available: {self.names}"
            )
        return self._store[name]

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[DeviceSpec]:
        return iter(self._store.values())

    @property
    def names(self) -> list[str]:
        return list(self._store.keys())

    @property
    def all_specs(self) -> list[DeviceSpec]:
        return list(self._store.values())

    def __repr__(self) -> str:
        return f"DeviceRegistry({self.names})"

import numpy as np

from constants import DIM_CROSS, SLICE_DERIVED


class CrossEncoder:
    def __init__(
        self,
        arch_param_idx: int = SLICE_DERIVED.start,  # 35
        device_tops_idx: int = 0,
        device_bw_idx: int = 1,
    ) -> None:
        self._arch_param_idx = arch_param_idx
        self._device_tops_idx = device_tops_idx
        self._device_bw_idx = device_bw_idx

    def encode(
        self,
        arch_vec: np.ndarray,
        device_vec: np.ndarray,
    ) -> np.ndarray:
        param_proxy = float(arch_vec[self._arch_param_idx])
        log_tops = float(device_vec[self._device_tops_idx])
        log_bw = float(device_vec[self._device_bw_idx])

        return np.array(
            [param_proxy * log_tops, param_proxy * log_bw],
            dtype=np.float32,
        )

    @property
    def output_dim(self) -> int:
        return DIM_CROSS

    @property
    def feature_names(self) -> list[str]:
        return ["cross_compute_pressure", "cross_memory_pressure"]

    def __repr__(self) -> str:
        return (
            f"CrossEncoder("
            f"arch_param_idx={self._arch_param_idx}, "
            f"device_tops_idx={self._device_tops_idx}, "
            f"device_bw_idx={self._device_bw_idx})"
        )

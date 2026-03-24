import numpy as np
from dataclasses import dataclass

from constants import SLICE_ARCH_ONEHOT, SLICE_DERIVED, SLICE_OP_COUNTS


@dataclass(frozen=True)
class EncodedSample:
    arch_vec: np.ndarray
    device_vec: np.ndarray
    cross_vec: np.ndarray
    arch_str: str
    device_name: str

    @property
    def feature_vector(self) -> np.ndarray:
        return np.concatenate([self.arch_vec, self.device_vec, self.cross_vec])

    @property
    def one_hot(self) -> np.ndarray:
        return self.arch_vec[SLICE_ARCH_ONEHOT]

    @property
    def op_counts(self) -> np.ndarray:
        return self.arch_vec[SLICE_OP_COUNTS]

    @property
    def derived(self) -> np.ndarray:
        return self.arch_vec[SLICE_DERIVED]

    @property
    def device_continuous(self) -> np.ndarray:
        return self.device_vec[:6]

    @property
    def device_binary(self) -> np.ndarray:
        return self.device_vec[6:]

    @property
    def dim(self) -> int:
        return self.feature_vector.shape[0]

    def __repr__(self) -> str:
        return (
            f"EncodedSample("
            f"device='{self.device_name}', "
            f"dim={self.dim}, "
            f"param_proxy={self.derived[0]:.1f}, "
            f"skip_ratio={self.derived[1]:.2f})"
        )

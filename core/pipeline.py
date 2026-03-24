import numpy as np
from typing import Any
from xgboost import XGBRegressor

from constants import DIM_TOTAL
from catalog import build_registry
from registry import DeviceRegistry
from encoders.arch_encoder import ArchEncoder
from encoders.cross_encoder import CrossEncoder
from encoders.device_encoder import DeviceEncoder
from core.dataset import DatasetBuilder
from core.model import ModelFactory
from core.sample import EncodedSample
from logger import Logger

logger = Logger()

class HardwareAwarePipeline:
    def __init__(
        self,
        log_transform_target: bool = True,
        device_registry: DeviceRegistry | None = None,
    ) -> None:
        self._registry = device_registry or build_registry()
        self._arch_enc = ArchEncoder()
        self._device_enc = DeviceEncoder(self._registry)
        self._cross_enc = CrossEncoder()
        self._builder: DatasetBuilder | None = None
        self._log_target = log_transform_target
        self._is_fitted = False

    def fit(self) -> "HardwareAwarePipeline":
        self._device_enc.fit()
        self._builder = DatasetBuilder(
            arch_encoder=self._arch_enc,
            device_encoder=self._device_enc,
            cross_encoder=self._cross_enc,
            log_transform_target=self._log_target,
        )
        self._is_fitted = True
        logger.info(f"Pipeline fitted on devices: {self._registry.names}")
        return self

    def register_device(
        self,
        name: str,
        features: list[float],
        description: str = "",
        refit: bool = True,
    ) -> "HardwareAwarePipeline":
        self._registry.register_from_list(name, features, description)
        if refit:
            self.fit()
        return self

    @property
    def device_names(self) -> list[str]:
        return self._registry.names

    def encode(self, arch_str: str, device_name: str) -> np.ndarray:
        self._check_fitted()
        return self._builder.encode(arch_str, device_name)

    def encode_sample(self, arch_str: str, device_name: str) -> EncodedSample:
        self._check_fitted()
        return self._builder.encode_sample(arch_str, device_name)

    def build_dataset(
        self,
        api: Any,
        hw_api: Any,
        device_names: list[str] | None = None,
        dataset: str = "cifar10",
        max_archs: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        self._check_fitted()
        return self._builder.build(
            api=api,
            hw_api=hw_api,
            device_names=device_names,
            dataset=dataset,
            max_archs=max_archs,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: list[dict],
        verbose: int = 50,
        **model_overrides: Any,
    ) -> tuple[XGBRegressor, dict[str, Any]]:
        return ModelFactory.train(
            X=X,
            y=y,
            meta=meta,
            verbose=verbose,
            **model_overrides,
        )

    @property
    def feature_names(self) -> list[str]:
        self._check_fitted()
        return self._builder.feature_names

    @property
    def output_dim(self) -> int:
        return DIM_TOTAL

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"HardwareAwarePipeline("
            f"status={status}, "
            f"devices={self._registry.names}, "
            f"output_dim={self.output_dim}, "
            f"log_transform={self._log_target})"
        )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call pipeline.fit() first.")

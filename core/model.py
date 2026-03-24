import logging
import numpy as np
import pandas as pd
from typing import Any
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


class ModelFactory:
    _DEFAULTS: dict[str, Any] = dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        objective="reg:squarederror",
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
    )

    @classmethod
    def build(cls, **overrides: Any) -> XGBRegressor:
        params = {**cls._DEFAULTS, **overrides}
        return XGBRegressor(**params)

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        meta: list[dict],
        test_size: float = 0.2,
        verbose: int = 50,
        **model_overrides: Any,
    ) -> tuple[XGBRegressor, dict[str, Any]]:
        arch_groups = [m["arch_idx"] for m in meta]
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=arch_groups))

        model = cls.build(**model_overrides)
        model.fit(
            X[train_idx],
            y[train_idx],
            eval_set=[(X[test_idx], y[test_idx])],
            verbose=verbose,
        )

        metrics = cls.evaluate(model, X[test_idx], y[test_idx], meta, test_idx)
        return model, metrics

    @classmethod
    def evaluate(
        cls,
        model: XGBRegressor,
        X_test: np.ndarray,
        y_test: np.ndarray,
        meta: list[dict],
        test_idx: np.ndarray,
        log_transformed: bool = True,
    ) -> dict[str, Any]:
        raw_preds = model.predict(X_test)
        if log_transformed:
            preds = np.expm1(raw_preds)
            truths = np.expm1(y_test)
        else:
            preds = raw_preds
            truths = y_test

        abs_err = np.abs(preds - truths)
        mae = float(abs_err.mean())
        mape = float((abs_err / (np.abs(truths) + 1e-8)).mean()) * 100.0

        # Per-device breakdown
        df = pd.DataFrame([meta[i] for i in test_idx])
        df["abs_err"] = abs_err
        per_device_mae = df.groupby("device")["abs_err"].mean().to_dict()

        metrics = dict(mae=mae, mape=mape, per_device_mae=per_device_mae)
        logger.info("Evaluation — MAE: %.4f | MAPE: %.2f%%", mae, mape)
        for dev, dev_mae in per_device_mae.items():
            logger.info("  %-12s MAE: %.4f", dev, dev_mae)
        return metrics

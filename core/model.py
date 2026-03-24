import numpy as np
import pandas as pd
from typing import Any
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit

from logger import logger


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
        eval_metric="mae",
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
    )

    @classmethod
    def build(cls, **overrides: Any) -> XGBRegressor:
        params = {**cls._DEFAULTS, **overrides}
        return XGBRegressor(**params)
    
    @classmethod
    def split_3way(cls, X, y, meta, train_ratio=0.7, val_ratio=0.15, random_state=42):
        groups = np.array([m["arch_idx"] for m in meta])
        gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
        train_idx, temp_idx = next(gss1.split(X, y, groups=groups))

        temp_groups = groups[temp_idx]

        val_size = val_ratio / (1.0 - train_ratio)

        gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size, random_state=random_state)
        val_sub_idx, test_sub_idx = next(gss2.split(X[temp_idx], y[temp_idx], groups=temp_groups))

        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        return train_idx, val_idx, test_idx

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        meta: list[dict],
        verbose: int = 50,
        **model_overrides: Any,
    ) -> tuple[XGBRegressor, dict[str, Any]]:
        train_idx, val_idx, test_idx = cls.split_3way(X, y, meta)

        model = cls.build(**model_overrides)
        alpha = 0.1
        weights = alpha * (1 / (y + 1e-6)) + (1 - alpha)
        model.fit(
            X[train_idx],
            y[train_idx],
            eval_set=[
                (X[train_idx], y[train_idx]),
                (X[val_idx], y[val_idx])
            ], # for drawing learning curve
            sample_weight=weights[train_idx],
            sample_weight_eval_set=[weights[train_idx], None],
            verbose=verbose,
        )
        results = model.evals_result()
        model.save_model(f"pretrained/xgb_model_weighted_mse_alpha_{str(alpha)}.json")
        
        train_curve = results["validation_0"]["mae"]
        val_curve = results["validation_1"]["mae"]
        for i, (tr, va) in enumerate(zip(train_curve, val_curve)):
            logger.info(f"[{i}] train-mae:{tr:.5f} | val-mae:{va:.5f}")

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
        raw_preds = model.predict(X_test, iteration_range=(0, model.best_iteration + 1))
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
        logger.info(f"Evaluation — MAE: {mae:.4f} | MAPE: {mape:.2f}%")
        for dev, dev_mae in per_device_mae.items():
            logger.info(f"  {dev} MAE: {dev_mae:.4f}")
        return metrics

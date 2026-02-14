"""
train.py

Entrena el modelo final en dos etapas, clasificación y regresión, usando los datasets
preparados en data/prep.

Entradas
--------
- data/prep/train.parquet
- data/prep/valid.parquet
- data/prep/meta.json

Salidas
-------
- artifacts/model.joblib
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    LGBM_CLASSIFIER_PARAMS,
    LGBM_REGRESSOR_PARAMS,
    LOG_DIR,
    PREP_DIR,
    TARGET_MAX,
    TARGET_MIN,
)
from src.utils.data_validation import require_file
from src.utils.logging_utils import get_logger
from src.utils.metrics import rmse


@dataclass(frozen=True)
class TrainInputs:
    """Datasets y metadata cargados desde data/prep."""

    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    feature_cols: list[str]
    meta: dict[str, Any]


@dataclass(frozen=True)
class DatasetSplit:
    """Matriz X e y para entrenamiento/validación."""

    x: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class TrainContext:
    """Contexto común de entrenamiento."""

    feature_cols: list[str]
    cat_features: list[str]
    logger: logging.Logger


@dataclass(frozen=True)
class ModelBundle:
    """Modelo final de dos etapas."""

    clf: lgb.LGBMClassifier
    reg: lgb.LGBMRegressor
    feature_cols: list[str]
    cat_features: list[str]


def cargar_datasets(prep_dir: Path) -> TrainInputs:
    """
    Carga train/valid parquet y meta.json con la lista de features.
    """
    train_path = prep_dir / "train.parquet"
    valid_path = prep_dir / "valid.parquet"
    meta_path = prep_dir / "meta.json"

    require_file(train_path)
    require_file(valid_path)
    require_file(meta_path)

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)

    with meta_path.open("r", encoding="utf-8") as file:
        meta: dict[str, Any] = json.load(file)

    feature_cols = list(meta["feature_cols"])
    return TrainInputs(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
        meta=meta,
    )


def build_context(feature_cols: list[str], logger: logging.Logger) -> TrainContext:
    """
    Construye contexto de entrenamiento (features y categóricas).
    """
    candidates = ["shop_id", "item_id", "month", "year"]
    cat_features = [col for col in candidates if col in feature_cols]
    return TrainContext(
        feature_cols=feature_cols, cat_features=cat_features, logger=logger
    )


def build_splits(inputs: TrainInputs) -> tuple[DatasetSplit, DatasetSplit]:
    """
    Construye splits X/y para train y valid.
    """
    x_train = inputs.train_df[inputs.feature_cols]
    y_train = inputs.train_df["y"].astype(np.float32)

    x_valid = inputs.valid_df[inputs.feature_cols]
    y_valid = inputs.valid_df["y"].astype(np.float32)

    return DatasetSplit(x=x_train, y=y_train), DatasetSplit(x=x_valid, y=y_valid)


def train_classifier(
    train_split: DatasetSplit, valid_split: DatasetSplit, ctx: TrainContext
) -> tuple[lgb.LGBMClassifier, np.ndarray]:
    """
    Entrena el clasificador (venta vs no venta) y regresa probas en validación.
    """
    logger = ctx.logger
    logger.info("Entrenando clasificador (venta vs no venta).")

    y_train_bin = (train_split.y > 0).astype(int).to_numpy()
    y_valid_bin = (valid_split.y > 0).astype(int).to_numpy()

    classifier = lgb.LGBMClassifier(**LGBM_CLASSIFIER_PARAMS)
    classifier.fit(
        train_split.x,
        y_train_bin,
        eval_set=[(valid_split.x, y_valid_bin)],
        eval_metric="binary_logloss",
        categorical_feature=ctx.cat_features,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    prob_valid = classifier.predict_proba(valid_split.x)[:, 1].astype(np.float32)
    logger.info(
        "Clasificador entrenado. prob_valid_mean=%.4f", float(prob_valid.mean())
    )
    return classifier, prob_valid


def train_regressor(
    train_split: DatasetSplit, valid_split: DatasetSplit, ctx: TrainContext
) -> tuple[lgb.LGBMRegressor, np.ndarray]:
    """
    Entrena el regresor sobre y>0 y regresa mu en validación.
    """
    logger = ctx.logger
    logger.info("Entrenando regresor (unidades | venta).")

    regressor = lgb.LGBMRegressor(**LGBM_REGRESSOR_PARAMS)

    mask_pos_train = train_split.y > 0
    mask_pos_valid = valid_split.y > 0

    regressor.fit(
        train_split.x.loc[mask_pos_train],
        train_split.y.loc[mask_pos_train],
        eval_set=[
            (valid_split.x.loc[mask_pos_valid], valid_split.y.loc[mask_pos_valid])
        ],
        eval_metric="rmse",
        categorical_feature=ctx.cat_features,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    mu_valid = regressor.predict(valid_split.x).astype(np.float32)
    logger.info("Regresor entrenado. mu_valid_mean=%.4f", float(mu_valid.mean()))
    return regressor, mu_valid


def evaluate_valid(
    valid_split: DatasetSplit,
    prob_valid: np.ndarray,
    mu_valid: np.ndarray,
    logger: logging.Logger,
) -> float:
    """
    Evalúa RMSE en validación para pred = clip(p * mu, 0, 20).
    """
    pred_valid = np.clip(prob_valid * mu_valid, TARGET_MIN, TARGET_MAX)
    score = rmse(valid_split.y.to_numpy(), pred_valid)
    logger.info("RMSE en validación: %.6f", score)

    n_peaks_5 = int((valid_split.y >= 5).sum())
    n_peaks_10 = int((valid_split.y >= 10).sum())
    n_peaks_15 = int((valid_split.y >= 15).sum())
    logger.info(
        "Picos en validación: y>=5=%d, y>=10=%d, y>=15=%d",
        n_peaks_5,
        n_peaks_10,
        n_peaks_15,
    )
    return score


def entrenar_modelo_dos_etapas(
    inputs: TrainInputs, logger: logging.Logger
) -> tuple[ModelBundle, float]:
    """
    Entrena el modelo de dos etapas y regresa bundle + RMSE validación.
    """
    ctx = build_context(inputs.feature_cols, logger)
    train_split, valid_split = build_splits(inputs)

    logger.info(
        "Features=%d | cat_features=%d",
        len(ctx.feature_cols),
        len(ctx.cat_features),
    )

    clf, prob_valid = train_classifier(train_split, valid_split, ctx)
    reg, mu_valid = train_regressor(train_split, valid_split, ctx)
    score = evaluate_valid(valid_split, prob_valid, mu_valid, logger)

    bundle = ModelBundle(
        clf=clf,
        reg=reg,
        feature_cols=ctx.feature_cols,
        cat_features=ctx.cat_features,
    )
    return bundle, score


def guardar_modelo(
    artifacts_dir: Path,
    bundle: ModelBundle,
    meta: dict[str, Any],
    logger: logging.Logger,
) -> Path:
    """
    Guarda el bundle del modelo entrenado en artifacts/model.joblib.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.joblib"

    payload = {
        "bundle": {
            "clf": bundle.clf,
            "reg": bundle.reg,
            "feature_cols": bundle.feature_cols,
            "cat_features": bundle.cat_features,
        },
        "meta": meta,
    }

    joblib.dump(payload, model_path)
    logger.info("Modelo guardado en: %s", model_path.as_posix())
    return model_path


def main() -> None:
    """
    Punto de entrada del pipeline de entrenamiento.
    """
    logger = get_logger(__name__, log_dir=LOG_DIR, prefijo_archivo="train")
    start = time.perf_counter()

    logger.info("Iniciando entrenamiento.")

    try:
        inputs = cargar_datasets(PREP_DIR)
        logger.info(
            "Datasets cargados. train_rows=%d valid_rows=%d",
            len(inputs.train_df),
            len(inputs.valid_df),
        )

        bundle, score = entrenar_modelo_dos_etapas(inputs, logger)
        guardar_modelo(ARTIFACTS_DIR, bundle, inputs.meta, logger)

        logger.info("RMSE final (validación): %.6f", score)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Fallo en entrenamiento: %s", str(exc))
        raise

    duration = time.perf_counter() - start
    logger.info("Entrenamiento terminado. duracion_seg=%.2f", duration)


if __name__ == "__main__":
    main()

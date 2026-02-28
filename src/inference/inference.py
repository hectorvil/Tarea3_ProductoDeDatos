"""
inference.py

Ejecuta inferencia batch con el modelo entrenado.

Entradas
--------
- data/inference/test_features.parquet (si no existe,
  se copia desde data/prep/test_features.parquet)
- data/prep/test_pairs.parquet
- data/raw/test.csv
- artifacts/model.joblib

Salida
------
- data/predictions/submission.csv
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    INFERENCE_DIR,
    LOG_DIR,
    PREDICTIONS_DIR,
    PREP_DIR,
    RAW_DIR,
    TARGET_MAX,
    TARGET_MIN,
)
from src.utils.data_validation import require_columns, require_file
from src.utils.logging_utils import get_logger


@dataclass(frozen=True)
class InferenceInputs:
    """Inputs necesarios para inferencia batch."""

    test_features: pd.DataFrame
    test_pairs: pd.DataFrame
    raw_test: pd.DataFrame


def asegurar_features_inferencia(logger: logging.Logger) -> Path:
    """
    Verifica que exista data/inference/test_features.parquet.
    Si no existe, lo copia desde data/prep/test_features.parquet.
    """
    infer_path = INFERENCE_DIR / "test_features.parquet"
    source_path = PREP_DIR / "test_features.parquet"

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    if infer_path.exists():
        logger.info("Features de inferencia encontrados: %s", infer_path.as_posix())
        return infer_path

    require_file(source_path)
    shutil.copyfile(source_path, infer_path)
    logger.info(
        "Copiando features de inferencia desde %s a %s",
        source_path.as_posix(),
        infer_path.as_posix(),
    )
    return infer_path


def cargar_modelo(model_path: Path) -> dict[str, Any]:
    """
    Carga el modelo entrenado desde artifacts/model.joblib.
    """
    require_file(model_path)
    payload: dict[str, Any] = joblib.load(model_path)
    if "bundle" not in payload:
        raise ValueError("Archivo de modelo inválido: se esperaba la llave 'bundle'.")
    return payload


def predecir(payload: dict[str, Any], test_features: pd.DataFrame) -> np.ndarray:
    """
    Genera predicciones usando el modelo en dos etapas.
    """
    bundle = payload["bundle"]
    feature_cols = bundle["feature_cols"]

    x_test = test_features[feature_cols]
    prob = bundle["clf"].predict_proba(x_test)[:, 1].astype(np.float32)
    mu = bundle["reg"].predict(x_test).astype(np.float32)

    return np.clip(prob * mu, TARGET_MIN, TARGET_MAX)


def construir_submission(
    preds: np.ndarray, test_pairs: pd.DataFrame, raw_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Arma submission.csv con columnas ID y item_cnt_month.
    """
    pred_map = pd.DataFrame(
        {
            "shop_id": test_pairs["shop_id"].values,
            "item_id": test_pairs["item_id"].values,
            "item_cnt_month": preds,
        }
    )

    submission = raw_test.merge(pred_map, on=["shop_id", "item_id"], how="left")
    submission["item_cnt_month"] = (
        submission["item_cnt_month"].fillna(0).clip(TARGET_MIN, TARGET_MAX)
    )
    return submission[["ID", "item_cnt_month"]]


def _log_basic_df_info(logger: logging.Logger, name: str, df: pd.DataFrame) -> None:
    """
    Loggea información básica de un DataFrame (shape + NaNs) sin imprimir datos.
    """
    logger.info("%s: filas=%d cols=%d", name, len(df), df.shape[1])
    na_total = int(df.isna().sum().sum())
    if na_total > 0:
        logger.warning("%s: total_NA=%d", name, na_total)


def cargar_inputs_inferencia(logger: logging.Logger) -> InferenceInputs:
    """
    Carga archivos necesarios para inferencia y realiza validaciones básicas.
    """
    features_path = asegurar_features_inferencia(logger)
    test_features = pd.read_parquet(features_path)
    _log_basic_df_info(logger, "test_features", test_features)

    test_pairs_path = PREP_DIR / "test_pairs.parquet"
    require_file(test_pairs_path)
    test_pairs = pd.read_parquet(test_pairs_path)
    _log_basic_df_info(logger, "test_pairs", test_pairs)

    raw_test_path = RAW_DIR / "test.csv"
    require_file(raw_test_path)
    raw_test = pd.read_csv(raw_test_path, encoding="utf-8")
    raw_test.columns = raw_test.columns.str.strip()
    require_columns(raw_test, ["ID", "shop_id", "item_id"], "test.csv")
    _log_basic_df_info(logger, "raw_test", raw_test)

    raw_test["ID"] = pd.to_numeric(raw_test["ID"], errors="coerce").astype(np.int32)
    raw_test["shop_id"] = pd.to_numeric(raw_test["shop_id"], errors="coerce").astype(
        np.int16
    )
    raw_test["item_id"] = pd.to_numeric(raw_test["item_id"], errors="coerce").astype(
        np.int16
    )

    return InferenceInputs(
        test_features=test_features,
        test_pairs=test_pairs,
        raw_test=raw_test,
    )


def run_inference(inputs: InferenceInputs, logger: logging.Logger) -> pd.DataFrame:
    """
    Corre inferencia completa y regresa el DataFrame de submission.
    """
    model_path = ARTIFACTS_DIR / "model.joblib"
    payload = cargar_modelo(model_path)

    bundle = payload["bundle"]
    feature_cols = bundle.get("feature_cols", [])
    missing = [col for col in feature_cols if col not in inputs.test_features.columns]
    if missing:
        logger.warning("Faltan columnas en test_features: %s", missing)

    logger.info("Generando predicciones.")
    preds = predecir(payload, inputs.test_features)
    logger.info(
        "Preds: n=%d min=%.4f p50=%.4f max=%.4f",
        len(preds),
        float(np.min(preds)),
        float(np.median(preds)),
        float(np.max(preds)),
    )

    return construir_submission(preds, inputs.test_pairs, inputs.raw_test)


def guardar_submission(submission: pd.DataFrame, logger: logging.Logger) -> Path:
    """
    Guarda submission.csv en data/predictions/.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / "submission.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8")

    logger.info(
        "Submission guardado: %s filas=%d", out_path.as_posix(), len(submission)
    )
    return out_path


def main() -> None:
    """
    Punto de entrada del pipeline de inferencia batch.
    """
    logger = get_logger(__name__, log_dir=LOG_DIR, prefijo_archivo="inference")
    start = time.perf_counter()
    logger.info("Iniciando inferencia batch.")

    try:
        inputs = cargar_inputs_inferencia(logger)
        submission = run_inference(inputs, logger)
        guardar_submission(submission, logger)

    except FileNotFoundError as exc:
        logger.exception("Archivo requerido no encontrado: %s", str(exc))
        raise
    except (ValueError, KeyError) as exc:
        logger.exception("Error de validación/estructura de datos: %s", str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error inesperado en inferencia: %s", str(exc))
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Inferencia terminada. duracion_seg=%.2f", duration)


if __name__ == "__main__":
    main()

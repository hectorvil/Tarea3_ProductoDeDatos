#!/usr/bin/env python3
from __future__ import annotations

import io
from pathlib import Path

import flask
import joblib
import numpy as np
import pandas as pd

from src.config import TARGET_MIN, TARGET_MAX

MODEL_PATH = Path("/opt/ml/model/model.joblib")
app = flask.Flask(__name__)
_payload = None


def _load_payload():
    global _payload
    if _payload is None:
        _payload = joblib.load(MODEL_PATH)
    return _payload


def _predict(df: pd.DataFrame) -> np.ndarray:
    payload = _load_payload()
    bundle = payload["bundle"]
    feature_cols = bundle["feature_cols"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    x = df[feature_cols]
    prob = bundle["clf"].predict_proba(x)[:, 1].astype(np.float32)
    mu = bundle["reg"].predict(x).astype(np.float32)

    alpha = float(payload.get("alpha", 0.90))
    preds = (prob ** alpha) * mu
    return np.clip(preds, TARGET_MIN, TARGET_MAX)


@app.get("/ping")
def ping():
    try:
        _load_payload()
        return flask.Response(response="\n", status=200)
    except Exception:
        return flask.Response(response="\n", status=404)


@app.post("/invocations")
def invocations():
    ct = flask.request.content_type or ""

    if "application/json" in ct:
        data = flask.request.get_json(force=True)
        if isinstance(data, dict) and "instances" in data:
            df = pd.DataFrame(data["instances"])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return flask.Response(
                "JSON inválido. Usa lista de registros o {'instances': [...]}",
                status=400,
            )

    elif "text/csv" in ct:
        text = flask.request.data.decode("utf-8")
        df = pd.read_csv(io.StringIO(text))

    else:
        return flask.Response("Unsupported Content-Type", status=415)

    try:
        preds = _predict(df)
    except Exception as e:
        return flask.Response(str(e), status=400)

    return flask.jsonify({"predictions": preds.tolist()})
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, List, Optional
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns=[data.target_names[0]])
    y = data.frame[data.target_names[0]]
    y.name = "MedHouseVal"
    return X, y


def build_model(model_type: str, alpha: float, standardize: bool, poly_degree: int) -> Pipeline:
    steps = []
    if poly_degree and poly_degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    if standardize:
        steps.append(("scaler", StandardScaler()))
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=alpha, random_state=42)
    elif model_type == "lasso":
        model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    else:
        raise ValueError("model_type inválido. Usa: linear | ridge | lasso")
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def get_feature_names_from_pipeline(pipeline: Pipeline, original_names: List[str]) -> List[str]:
    names = original_names
    if "poly" in pipeline.named_steps:
        names = pipeline.named_steps["poly"].get_feature_names_out(original_names).tolist()
    return names


# ======== GRÁFICAS (se muestran en pantalla) ========
def plot_parity(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.xlabel("Valor real (y)"); plt.ylabel("Predicción (ŷ)")
    plt.title("Paridad: y vs. ŷ")
    plt.tight_layout()
    plt.show()


def plot_residuals_vs_pred(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(0)
    plt.xlabel("Predicción (ŷ)"); plt.ylabel("Residual (y - ŷ)")
    plt.title("Residuales vs. Predicción")
    plt.tight_layout()
    plt.show()


def plot_residual_hist(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure()
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual (y - ŷ)"); plt.ylabel("Frecuencia")
    plt.title("Histograma de residuales")
    plt.tight_layout()
    plt.show()


def plot_top_coefficients(pipeline: Pipeline, original_feature_names: List[str], k: int = 15):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return
    names = get_feature_names_from_pipeline(pipeline, original_feature_names)
    coef = np.array(model.coef_).ravel()
    if len(coef) != len(names):
        return
    idx = np.argsort(np.abs(coef))[::-1][:min(k, len(coef))]
    top_names = [names[i] for i in idx]
    top_vals = coef[idx]
    plt.figure()
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, top_vals)
    plt.yticks(y_pos, top_names)
    plt.gca().invert_yaxis()
    plt.xlabel("Coeficiente"); plt.title(f"Top {len(top_names)} coeficientes por |peso|")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(X_train, y_train, X_val, y_val, builder_args: dict, train_sizes=np.linspace(0.1, 1.0, 8)):
    n = X_train.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    X_train = X_train[idx]; y_train = y_train[idx]

    rmse_tr, rmse_va = [], []
    for frac in train_sizes:
        m = max(5, int(frac * n))
        pipe = build_model(**builder_args)
        pipe.fit(X_train[:m], y_train[:m])
        yhat_tr = pipe.predict(X_train[:m])
        yhat_va = pipe.predict(X_val)
        rmse_tr.append(np.sqrt(mean_squared_error(y_train[:m], yhat_tr)))
        rmse_va.append(np.sqrt(mean_squared_error(y_val, yhat_va)))

    plt.figure()
    plt.plot(train_sizes * 100, rmse_tr, marker="o", label="Train RMSE")
    plt.plot(train_sizes * 100, rmse_va, marker="s", label="Valid RMSE")
    plt.xlabel("% de datos de entrenamiento"); plt.ylabel("RMSE")
    plt.title("Curva de aprendizaje")
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Regresión Lineal (sklearn) con gráficas mostradas en pantalla (sin archivos).")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--model", type=str, default="linear", choices=["linear", "ridge", "lasso"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--poly-degree", type=int, default=1)
    ap.add_argument("--predict-csv", type=str, default=None, help="CSV con columnas iguales a las features originales (para predecir en el mismo proceso).")
    ap.add_argument("--predict-head", type=int, default=5, help="Cuántas predicciones imprimir.")
    args = ap.parse_args()

    X, y = load_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    builder_args = dict(model_type=args.model, alpha=args.alpha, standardize=args.standardize, poly_degree=args.poly_degree)
    pipeline = build_model(**builder_args)
    pipeline.fit(Xtr, ytr)
    yhat = pipeline.predict(Xte)
    metrics = evaluate(yte.values, yhat)

    print("=== California Housing | Regresión (sklearn) ===")
    print(f"Filas: total={len(X)} | train={len(Xtr)} | test={len(Xte)}")
    print(f"Modelo: {args.model} | alpha={args.alpha} | standardize={args.standardize} | poly_degree={args.poly_degree}")
    print("\nMétricas en TEST:")
    for k, v in metrics.items():
        print(f"- {k}: {v:.6f}")

    # Gráficas en pantalla
    plot_parity(yte.values, yhat)
    plot_residuals_vs_pred(yte.values, yhat)
    plot_residual_hist(yte.values, yhat)
    plot_top_coefficients(pipeline, list(X.columns), k=15)
    plot_learning_curve(Xtr.values, ytr.values, Xte.values, yte.values, builder_args)

    # Predicción inmediata (opcional, en consola)
    if args.predict_csv:
        df_new = pd.read_csv(args.predict_csv)
        missing = [c for c in X.columns if c not in df_new.columns]
        if missing:
            raise ValueError(f"Faltan columnas en el CSV: {missing}")
        preds = pipeline.predict(df_new[X.columns])
        out = pd.DataFrame({"prediction": preds})
        print("\nPredicciones (primeras filas):")
        print(out.head(args.predict_head).to_string(index=False))


if __name__ == "__main__":
    main()

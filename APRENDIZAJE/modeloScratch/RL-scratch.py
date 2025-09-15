#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# ===== Escalado scratch =====
class StandardScalerScratch:
    def __init__(self): self.mean_=None; self.std_=None
    def fit(self, X): self.mean_=X.mean(axis=0); self.std_=X.std(axis=0, ddof=0); self.std_[self.std_==0]=1.0; return self
    def transform(self, X): return (X - self.mean_) / self.std_
    def fit_transform(self, X): return self.fit(X).transform(X)


# ===== Polinomios =====
def polynomial_features(X: np.ndarray, feature_names: List[str], degree: int) -> Tuple[np.ndarray, List[str]]:
    if degree <= 1: return X, feature_names[:]
    cols = [X[:, j] for j in range(X.shape[1])]; names = feature_names[:]
    for deg in range(2, degree+1):
        for comb in itertools.combinations_with_replacement(range(X.shape[1]), deg):
            X = np.column_stack([X, np.prod([cols[j] for j in comb], axis=0)])
            names.append("*".join([feature_names[j] for j in comb]))
    return X, names


def train_test_split_scratch(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state); n = X.shape[0]
    idx = np.arange(n); rng.shuffle(idx)
    n_test = int(np.floor(test_size*n))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


# ===== Modelo lineal scratch =====
class LinearRegressionScratch:
    def __init__(self, solver="normal_equation", lr=1e-2, epochs=1000, tol=1e-8, verbose=False):
        self.solver, self.lr, self.epochs, self.tol, self.verbose = solver, lr, epochs, tol, verbose
        self.theta_ = None
    @staticmethod
    def _add_bias(X): return np.column_stack([np.ones((X.shape[0],1)), X])
    def fit(self, X, y):
        Xb = self._add_bias(X)
        if self.solver == "normal_equation":
            self.theta_ = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
        elif self.solver == "gd":
            rng = np.random.default_rng(42); self.theta_ = rng.normal(0,0.01,size=Xb.shape[1]); prev = np.inf
            for it in range(self.epochs):
                r = Xb@self.theta_ - y; loss = (r@r)/Xb.shape[0]
                grad = (2.0/Xb.shape[0])*(Xb.T@r); self.theta_ -= self.lr*grad
                if self.verbose and it % max(1,self.epochs//10)==0: print(f"[GD] it={it} MSE={loss:.6f}")
                if abs(prev-loss) < self.tol: break; prev=loss
        else: raise ValueError("solver inválido")
        return self
    def predict(self, X): return self._add_bias(X) @ self.theta_
    def coef_(self): return self.theta_[1:]
    def intercept_(self): return float(self.theta_[0])


# ===== Métricas =====
def evaluate(y, yhat):
    mse = float(np.mean((y-yhat)**2)); rmse=float(np.sqrt(mse)); mae=float(np.mean(np.abs(y-yhat)))
    ss_res=float(np.sum((y-yhat)**2)); ss_tot=float(np.sum((y-np.mean(y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else float("nan")
    return {"MSE":mse,"RMSE":rmse,"MAE":mae,"R2":r2}


# ===== Gráficas (show) =====
def plot_parity(y, yhat):
    plt.figure(); plt.scatter(y, yhat, alpha=0.4)
    lims=[min(y.min(),yhat.min()), max(y.max(),yhat.max())]; plt.plot(lims, lims)
    plt.xlabel("Valor real (y)"); plt.ylabel("Predicción (ŷ)"); plt.title("Paridad: y vs. ŷ"); plt.tight_layout(); plt.show()

def plot_residuals_vs_pred(y, yhat):
    r=y-yhat; plt.figure(); plt.scatter(yhat, r, alpha=0.4); plt.axhline(0)
    plt.xlabel("Predicción (ŷ)"); plt.ylabel("Residual (y-ŷ)"); plt.title("Residuales vs. Predicción"); plt.tight_layout(); plt.show()

def plot_residual_hist(y, yhat):
    r=y-yhat; plt.figure(); plt.hist(r,bins=40); plt.xlabel("Residual (y-ŷ)"); plt.ylabel("Frecuencia")
    plt.title("Histograma de residuales"); plt.tight_layout(); plt.show()

def plot_top_coefficients(coef, names, k=15):
    idx=np.argsort(np.abs(coef))[::-1][:min(k,len(coef))]; top_names=[names[i] for i in idx]; top_vals=coef[idx]
    plt.figure(); y_pos=np.arange(len(top_names)); plt.barh(y_pos, top_vals); plt.yticks(y_pos, top_names); plt.gca().invert_yaxis()
    plt.xlabel("Coeficiente"); plt.title(f"Top {len(top_names)} coeficientes por |peso|"); plt.tight_layout(); plt.show()

def plot_learning_curve_scratch(Xtr, ytr, Xte, yte, solver, lr, epochs, tol, train_sizes=np.linspace(0.1,1.0,8)):
    n=Xtr.shape[0]; idx=np.arange(n); rng=np.random.default_rng(42); rng.shuffle(idx)
    Xtr, ytr = Xtr[idx], ytr[idx]
    rmse_tr, rmse_va = [], []
    for frac in train_sizes:
        m=max(5,int(frac*n))
        model=LinearRegressionScratch(solver=solver, lr=lr, epochs=epochs, tol=tol)
        model.fit(Xtr[:m], ytr[:m])
        yhat_tr=model.predict(Xtr[:m]); yhat_va=model.predict(Xte)
        rmse_tr.append(np.sqrt(np.mean((ytr[:m]-yhat_tr)**2)))
        rmse_va.append(np.sqrt(np.mean((yte-yhat_va)**2)))
    plt.figure(); plt.plot(train_sizes*100, rmse_tr, marker="o", label="Train RMSE")
    plt.plot(train_sizes*100, rmse_va, marker="s", label="Valid RMSE")
    plt.xlabel("% de datos de entrenamiento"); plt.ylabel("RMSE"); plt.title("Curva de aprendizaje (scratch)")
    plt.legend(); plt.grid(True, alpha=0.2); plt.tight_layout(); plt.show()


# ===== Datos =====
def load_california_df():
    try:
        from sklearn.datasets import fetch_california_housing
    except Exception as e:
        raise RuntimeError("Instala scikit-learn para --use-sklearn-data") from e
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy(); df.rename(columns={data.target_names[0]: "MedHouseVal"}, inplace=True)
    return df

def load_Xy_from_df(df: pd.DataFrame, target: str):
    if target not in df.columns: raise ValueError(f"Target '{target}' no existe.")
    y = df[target].to_numpy(dtype=float)
    X_df = df.drop(columns=[target])
    return X_df.to_numpy(dtype=float), y, X_df.columns.tolist()


def main():
    ap = argparse.ArgumentParser(description="Regresión Lineal FROM SCRATCH (NumPy) con gráficas en pantalla (sin archivos).")
    ap.add_argument("--csv", type=str, default=None, help="Ruta a CSV que incluya el target.")
    ap.add_argument("--use-sklearn-data", action="store_true", help="Usar California Housing (solo para datos).")
    ap.add_argument("--target", type=str, default="MedHouseVal")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--poly-degree", type=int, default=1)
    ap.add_argument("--solver", type=str, default="normal_equation", choices=["normal_equation","gd"])
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--predict-csv", type=str, default=None, help="CSV para predecir en el mismo proceso.")
    ap.add_argument("--predict-head", type=int, default=5)
    args = ap.parse_args()

    # Carga datos
    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.use_sklearn_data:
        df = load_california_df(); args.target = "MedHouseVal"
    else:
        raise ValueError("Usa --csv <ruta> o --use-sklearn-data")

    X, y, feat_names = load_Xy_from_df(df, args.target)

    # Polinomios
    X_poly, poly_names = polynomial_features(X, feat_names, degree=args.poly_degree)

    # Split
    Xtr, Xte, ytr, yte = train_test_split_scratch(X_poly, y, test_size=args.test_size, random_state=args.random_state)

    # Escalar
    scaler = None
    if args.standardize:
        scaler = StandardScalerScratch()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

    # Entrenar
    model = LinearRegressionScratch(solver=args.solver, lr=args.lr, epochs=args.epochs, tol=args.tol, verbose=args.verbose)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    metrics = evaluate(yte, yhat)

    print("=== Linear Regression (Scratch) ===")
    print(f"Filas: total={len(df)} | train={len(Xtr)} | test={len(Xte)}")
    print(f"Target: {args.target}")
    print(f"Solver: {args.solver} | lr={args.lr} | epochs={args.epochs} | tol={args.tol}")
    print(f"Standardize={args.standardize} | poly_degree={args.poly_degree}")
    print("\nMétricas TEST:")
    for k,v in metrics.items(): print(f"- {k}: {v:.6f}")

    # Gráficas
    plot_parity(yte, yhat)
    plot_residuals_vs_pred(yte, yhat)
    plot_residual_hist(yte, yhat)
    plot_top_coefficients(model.coef_(), poly_names, k=15)
    plot_learning_curve_scratch(Xtr, ytr, Xte, yte, solver=args.solver, lr=args.lr, epochs=args.epochs, tol=args.tol)

    # Predicción inmediata (opcional, en consola)
    if args.predict_csv:
        new_df = pd.read_csv(args.predict_csv)
        missing = [c for c in feat_names if c not in new_df.columns]
        if missing: raise ValueError(f"Faltan columnas en el CSV: {missing}")
        X_new = new_df[feat_names].to_numpy(dtype=float)
        # aplicar las mismas transf.
        X_new, _names_new = polynomial_features(X_new, feat_names, degree=args.poly_degree)
        if args.standardize: X_new = (X_new - scaler.mean_) / scaler.std_
        preds = model.predict(X_new)
        out = pd.DataFrame({"prediction": preds})
        print("\nPredicciones (primeras filas):")
        print(out.head(args.predict_head).to_string(index=False))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regresión lineal multivariable DESDE CERO (sin numpy/pandas/ML)
- Descenso de gradiente con L2 opcional
- Estandarización manual de features numéricas
- Carga CSV del dataset COVID-19 (Gobierno de México)
- Objetivo: riesgo continuo basado en DATA_DIED (1 si != '9999-99-99', 0 en otro caso)
- Métricas: MAE, RMSE, R^2
- Uso:
    python3 regresion_lineal_desde_cero.py --csv Covid-Data.csv --test-size 0.2 \
        --lr 0.05 --epochs 80 --l2 0.0 --seed 42 --save-preds preds.csv
"""
import csv, math, random, argparse, sys, os

# -------------------- Utilidades simples --------------------
def sigmoid(x):  # por si quieres activar salida, pero por defecto NO la uso
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

def stdev(xs):
    m = mean(xs)
    var = sum((x-m)*(x-m) for x in xs) / (len(xs)-1 if len(xs)>1 else 1)
    return math.sqrt(var)

def train_test_split(X, y, test_size=0.2, seed=42):
    rnd = random.Random(seed)
    idx = list(range(len(X)))
    rnd.shuffle(idx)
    n_test = int(len(X)*test_size)
    test_idx = set(idx[:n_test])
    Xtr, ytr, Xte, yte = [], [], [], []
    for i in range(len(X)):
        if i in test_idx:
            Xte.append(X[i]); yte.append(y[i])
        else:
            Xtr.append(X[i]); ytr.append(y[i])
    return Xtr, ytr, Xte, yte

# -------------------- Preprocesamiento --------------------
BOOL_MAYBE = {"1":1.0, 1:1.0, "2":0.0, 2:0.0, "97":0.0, 97:0.0, "98":0.0, 98:0.0, "99":0.0, 99:0.0, "":0.0}
INT_OR_ZERO = lambda v: float(v) if isinstance(v,(int,float)) or (isinstance(v,str) and v.isdigit()) else 0.0

def parse_bool(v):
    return BOOL_MAYBE.get(v, BOOL_MAYBE.get(str(v), 0.0))

def parse_age(v):
    try:
        a = float(v)
        return a if a>=0 and a<=120 else 0.0
    except:
        return 0.0

def died_from_date(date_died):
    # 1 si falleció (cualquier fecha distinta a 9999-99-99), 0 si no
    return 0.0 if (date_died is None or str(date_died).strip()=="9999-99-99") else 1.0

def load_csv_build_xy(path, max_rows=None):
    """
    Devuelve X (lista de listas) e y (lista), usando columnas comunes del dataset.
    No usa numpy/pandas.
    """
    if not os.path.exists(path):
        print(f"[ERROR] No se encontró el archivo: {path}")
        sys.exit(1)

    # columnas típicas del dataset; si alguna no existe se ignora silenciosamente
    wanted_cols = [
        "sex","age","patient_type","pneumonia","pregnancy","diabetes","copd","asthma",
        "inmsupr","hypertension","cardiovascular","renal_chronic","other_disease",
        "obesity","tobacco","medical_unit","intubed","icu","classification"
    ]
    target_col = "date_died"  # en varios CSV aparece con este nombre exacto

    X, y = [], []
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            # objetivo desde DATA_DIED/date_died
            if target_col not in row:
                # a veces viene como DATA_DIED (mayúsculas)
                if "DATA_DIED" in row:
                    row[target_col] = row["DATA_DIED"]
                else:
                    # si no existe, saltamos
                    continue
            yi = died_from_date(row[target_col])
            # features
            feats = []
            for c in wanted_cols:
                if c not in row:
                    feats.append(0.0); continue
                val = row[c]

                if c == "age":
                    feats.append(parse_age(val))
                elif c in ("patient_type","medical_unit","classification"):
                    # valores categóricos con números pequeños -> lo dejamos como número crudo
                    try:
                        feats.append(float(val))
                    except:
                        feats.append(0.0)
                else:
                    feats.append(parse_bool(val))
            X.append(feats); y.append(yi)
            count += 1
            if max_rows is not None and count >= max_rows:
                break
    return X, y, wanted_cols

def standardize_train(X):
    """
    Estandariza columnas: z = (x - media)/std. Devuelve Xstd, medias, stds
    """
    if not X: return [], [], []
    n = len(X); d = len(X[0])
    means = [0.0]*d; stds = [1.0]*d

    # medias
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += X[i][j]
        means[j] = s / n
    # stds
    for j in range(d):
        var = 0.0
        for i in range(n):
            diff = X[i][j] - means[j]
            var += diff*diff
        var /= (n-1) if n>1 else 1
        stds[j] = math.sqrt(var) if var>1e-12 else 1.0
    # aplicar
    Xs = []
    for i in range(n):
        row = [(X[i][j] - means[j]) / stds[j] for j in range(d)]
        Xs.append(row)
    return Xs, means, stds

def standardize_apply(X, means, stds):
    Xs = []
    for i in range(len(X)):
        row = [(X[i][j]-means[j])/stds[j] for j in range(len(means))]
        Xs.append(row)
    return Xs

# -------------------- Modelo: Regresión Lineal (GD) --------------------
class LinearRegressionGD:
    def __init__(self, lr=0.05, epochs=100, l2=0.0, seed=42, use_bias=True):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.use_bias = use_bias
        self.rnd = random.Random(seed)
        self.w = None   # pesos (incluye bias como w[-1] si use_bias)

    def _init_weights(self, d):
        self.w = [self.rnd.uniform(-0.5,0.5) for _ in range(d + (1 if self.use_bias else 0))]

    def _dot(self, w, x):
        s = 0.0
        D = len(x)
        for j in range(D):
            s += w[j]*x[j]
        if self.use_bias:
            s += w[-1]  # bias
        return s

    def predict_row(self, x):
        return self._dot(self.w, x)

    def predict(self, X):
        return [self.predict_row(x) for x in X]

    def fit(self, X, y, verbose=True):
        n = len(X)
        if n == 0:
            raise ValueError("X vacío")
        d = len(X[0])
        self._init_weights(d)

        for ep in range(1, self.epochs+1):
            # gradientes acumulados
            grad = [0.0]*d
            grad_b = 0.0

            # MSE + L2
            mse_sum = 0.0
            for i in range(n):
                yhat = self._dot(self.w, X[i])
                err = yhat - y[i]
                mse_sum += err*err

                # gradientes de w_j
                for j in range(d):
                    grad[j] += (2.0/n) * err * X[i][j]
                if self.use_bias:
                    grad_b += (2.0/n) * err

            # L2
            if self.l2 > 0.0:
                for j in range(d):
                    grad[j] += (2.0*self.l2/n) * self.w[j]

            # actualización
            for j in range(d):
                self.w[j] -= self.lr * grad[j]
            if self.use_bias:
                self.w[-1] -= self.lr * grad_b

            if verbose and (ep % max(1,self.epochs//10) == 0 or ep==1):
                mse = mse_sum/n
                print(f"[Epoch {ep:4d}] MSE={mse:.6f}")

# -------------------- Métricas --------------------
def mae(y, yhat):
    return sum(abs(a-b) for a,b in zip(y,yhat))/len(y) if y else 0.0

def rmse(y, yhat):
    return math.sqrt(sum((a-b)*(a-b) for a,b in zip(y,yhat))/len(y)) if y else 0.0

def r2(y, yhat):
    if not y: return 0.0
    ym = mean(y)
    ss_res = sum((a-b)*(a-b) for a,b in zip(y,yhat))
    ss_tot = sum((a-ym)*(a-ym) for a in y)
    return 1.0 - (ss_res/(ss_tot if ss_tot>1e-12 else 1.0))

# -------------------- Guardado de predicciones --------------------
def save_predictions(path, yhat):
    with open(path, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["y_hat"])
        for v in yhat:
            w.writerow([f"{v:.6f}"])

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Regresión lineal desde cero para COVID (riesgo continuo a partir de DATA_DIED).")
    ap.add_argument("--csv", required=False, help="Ruta al CSV del dataset.")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-rows", type=int, default=None, help="Opcional: recorta filas para pruebas rápidas.")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-preds", default=None, help="Ruta para guardar y_hat del test en CSV.")
    ap.add_argument("--no-verbose", action="store_true")
    args = ap.parse_args()

    if not args.csv:
        print("[AVISO] No se proporcionó --csv. Crearé un dataset de juguete para demostrar el pipeline.")
        # mini dataset sintético (age y un par de flags)
        X = [
            [1, 70, 2, 1, 0, 1, 0, 0, 1, 0,0,0,0,1,0, 3,1,1,3],
            [2, 25, 1, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0, 3,0,0,2],
            [1, 55, 2, 1, 0, 0, 0, 0, 1, 0,0,0,0,0,0, 6,0,0,3],
            [2, 80, 2, 1, 0, 1, 0, 0, 1, 1,1,0,0,0,0, 3,1,1,3],
            [1, 45, 1, 0, 0, 0, 0, 1, 0, 0,0,0,0,0,1, 2,0,0,1],
        ]
        y = [1,0,0,1,0]
        headers = ["sex","age","patient_type","pneumonia","pregnancy","diabetes","copd","asthma",
                   "inmsupr","hypertension","cardiovascular","renal_chronic","other_disease",
                   "obesity","tobacco","medical_unit","intubed","icu","classification"]
    else:
        print("[INFO] Cargando CSV…")
        X, y, headers = load_csv_build_xy(args.csv, max_rows=args.max_rows)
        print(f"[INFO] Filas válidas: {len(X)} | Features: {len(headers)}")

    Xtr, ytr, Xte, yte = train_test_split(X, y, test_size=args.test_size, seed=args.seed)
    print(f"[INFO] Split -> Train: {len(Xtr)} | Test: {len(Xte)}")

    # estandarizamos TODO (todas las columnas en este ejemplo)
    Xtr_std, means, stds = standardize_train(Xtr)
    Xte_std = standardize_apply(Xte, means, stds)

    model = LinearRegressionGD(lr=args.lr, epochs=args.epochs, l2=args.l2, seed=args.seed, use_bias=True)
    model.fit(Xtr_std, ytr, verbose=(not args.no_verbose))

    yhat_tr = model.predict(Xtr_std)
    yhat_te = model.predict(Xte_std)

    print("\n===== MÉTRICAS (Train) =====")
    print(f"MAE={mae(ytr,yhat_tr):.4f} | RMSE={rmse(ytr,yhat_tr):.4f} | R2={r2(ytr,yhat_tr):.4f}")

    print("\n===== MÉTRICAS (Test) =====")
    print(f"MAE={mae(yte,yhat_te):.4f} | RMSE={rmse(yte,yhat_te):.4f} | R2={r2(yte,yhat_te):.4f}")

    # Mostrar algunas predicciones como "riesgo continuo"
    print("\nEjemplos de predicción (y_real -> y_hat):")
    for i in range(min(10, len(yte))):
        print(f"{yte[i]:.0f} -> {yhat_te[i]:.4f}")

    if args.save_preds:
        save_predictions(args.save_preds, yhat_te)
        print(f"\n[OK] Predicciones guardadas en: {args.save_preds}")

if __name__ == "__main__":
    main()

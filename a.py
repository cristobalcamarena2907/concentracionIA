#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
covid_rf_balanced.py — Random Forest desde cero (sin librerías), con:
- Carga de CSV y mapeo de columnas por alias
- Transformación DATE_DIED -> target binario (1=muerto, 0=vivo)
- Bootstrap balanceado por árbol
- Gini con pesos de clase
- Probabilidades + búsqueda de umbral óptimo (F1 o Recall)
- Límite de thresholds por feature (K) para acelerar

Ejecuta:
  python3 covid_rf_balanced.py
"""

import csv, math, random, sys, time
from typing import List, Dict, Any, Tuple
from multiprocessing import Pool

# ========= CONFIG GENERAL =========
CSV_PATH   = "Covid Data.csv"      # <--- RUTA A TU CSV
TEST_RATIO = 0.2

# Submuestreo rápido opcional para prototipar (None = usar todo)
HEAD_N = None  # por ej. 200000 para pruebas rápidas

# ========= CONFIG RANDOM FOREST =========
N_TREES      = 50
MAX_DEPTH    = 10
MIN_SIZE     = 15
SAMPLE_SIZE  = 0.6     # fracción de datos por árbol
MTRY_MODE    = "log2"  # "sqrt", "log2" o int
K_THRESHOLDS = 16      # máx thresholds por feature en cada split (acelera)

# Bootstrap balanceado por árbol (50/50 aprox. de cada clase)
USE_BALANCED_BOOTSTRAP = True

# Gini con pesos de clase (cost-sensitive).  0: vivos, 1: muertos
CLASS_WEIGHTS = {0: 1.0, 1: 5.0}  # prueba 1:4, 1:5, 1:6…

# Paralelizar árboles (usa procesos del stdlib)
USE_PARALLEL = False  # True para acelerar si tienes varios núcleos

# Optimización de umbral en un set de validación
THRESHOLD_TARGET = "f1"  # "f1" o "recall"


# ========= ALIASES DE COLUMNAS =========
ALIASES: Dict[str, List[str]] = {
    "DATE_DIED": ["date_died","DATE_DIED"],
    "SEX": ["sex","SEX"],
    "AGE": ["age","AGE"],
    "PATIENT_TYPE": ["patient_type","PATIENT_TYPE"],
    "PNEUMONIA": ["pneumonia","PNEUMONIA"],
    "PREGNANT": ["pregnant","PREGNANT","pregnancy","PREGNANCY"],
    "DIABETES": ["diabetes","DIABETES"],
    "COPD": ["copd","COPD"],
    "ASTHMA": ["asthma","ASTHMA"],
    "INMSUPR": ["inmsupr","INMSUPR","immunosuppressed","IMMUNOSUPR"],
    "HYPERTENSION": ["hypertension","HYPERTENSION","hipertension","HIPERTENSION"],
    "CARDIOVASCULAR": ["cardiovascular","CARDIOVASCULAR"],
    "RENAL_CHRONIC": ["renal_chronic","RENAL_CHRONIC","chronic_renal","CHRONIC_RENAL"],
    "OTHER_DISEASE": ["other_disease","OTHER_DISEASE","other_diseases","OTHER_DISEASES"],
    "OBESITY": ["obesity","OBESITY"],
    "TOBACCO": ["tobacco","TOBACCO"],
    "USMER": ["usmer","USMER","usmr","USMR"],
    "MEDICAL_UNIT": ["medical_unit","MEDICAL_UNIT"],
    "INTUBED": ["intubed","INTUBED"],
    "ICU": ["icu","ICU"],
    "CLASSIFICATION_FINAL": [
        "classification_final","CLASSIFICATION_FINAL",
        "classification","CLASSIFICATION","clasiffication_final","CLASIFFICATION_FINAL"
    ],
}

DEFAULT_FEATURES = [
    "SEX","AGE","PATIENT_TYPE","PNEUMONIA","PREGNANT","DIABETES","COPD","ASTHMA",
    "INMSUPR","HYPERTENSION","CARDIOVASCULAR","RENAL_CHRONIC","OTHER_DISEASE",
    "OBESITY","TOBACCO","USMER","MEDICAL_UNIT","INTUBED","ICU","CLASSIFICATION_FINAL"
]


# ========= UTILIDADES DE CARGA =========
def try_open_csv(path: str):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            f = open(path, newline="", encoding=enc)
            _ = f.readline(); f.seek(0)
            return f
        except Exception:
            pass
    raise RuntimeError("No se pudo abrir el CSV con utf-8/latin-1.")

def build_header_map(reader: csv.DictReader) -> Dict[str, str]:
    actual = [h.strip() for h in (reader.fieldnames or [])]
    low2real = {h.lower(): h for h in actual}
    mapping = {}
    for canon, alist in ALIASES.items():
        for a in alist:
            if a.lower() in low2real:
                mapping[canon] = low2real[a.lower()]
                break
    return mapping

def parse_float(x: str, default: float = 0.0) -> float:
    x = x.strip()
    if x == "" or x.upper() == "NA": return default
    try: return float(int(x))
    except: 
        try: return float(x)
        except: return default

def preprocess_date_died(v: str) -> int:
    return 0 if v.strip() == "9999-99-99" else 1

def load_dataset(path: str, features_canon: List[str]) -> Tuple[List[List[float]], List[int]]:
    f = try_open_csv(path)
    reader = csv.DictReader(f)
    header_map = build_header_map(reader)

    print("\n== Mapeo de columnas detectado ==")
    for c in sorted(header_map): print(f"{c:22s} <- {header_map[c]}")
    missing = [c for c in features_canon + ["DATE_DIED"] if c not in header_map]
    if missing:
        print("\n[ADVERTENCIA] Faltan columnas (ajusta ALIASES/DEFAULT_FEATURES):")
        for m in missing: print("  -", m)
        print()

    X, y = [], []
    for row in reader:
        try:
            if "DATE_DIED" not in header_map: continue
            tgt = preprocess_date_died(row[header_map["DATE_DIED"]])
            feats = []
            ok = True
            for c in features_canon:
                if c not in header_map: ok = False; break
                feats.append(parse_float(row[header_map[c]], 0.0))
            if ok: X.append(feats); y.append(tgt)
        except Exception:
            pass
    f.close()
    return X, y

def stratified_head(X, y, n):
    if n is None or n >= len(X): return X, y
    pos = [i for i, t in enumerate(y) if t==1]
    neg = [i for i, t in enumerate(y) if t==0]
    random.shuffle(pos); random.shuffle(neg)
    k_pos = min(len(pos), n//5)  # ~20% positivos si hay
    k_neg = min(len(neg), n - k_pos)
    take = pos[:k_pos] + neg[:k_neg]
    random.shuffle(take)
    return [X[i] for i in take], [y[i] for i in take]


# ========= SPLIT & MÉTRICAS =========
def train_test_split(X, y, test_ratio=0.2):
    idx = list(range(len(X))); random.shuffle(idx)
    n_test = int(len(X) * test_ratio)
    test_idx = set(idx[:n_test])
    Xtr, ytr, Xte, yte = [], [], [], []
    for i in range(len(X)):
        (Xte if i in test_idx else Xtr).append(X[i])
        (yte if i in test_idx else ytr).append(y[i])
    return Xtr, ytr, Xte, yte

def metrics(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==0)
    acc = (tp+tn)/max(1,(tp+tn+fp+fn))
    prec = tp/max(1,(tp+fp))
    rec = tp/max(1,(tp+fn))
    f1 = 2*prec*rec/max(1e-12,(prec+rec))
    return acc, prec, rec, f1, (tp, fp, fn, tn)


# ========= RANDOM FOREST CORE =========
def choose_mtry(n_features: int) -> int:
    if isinstance(MTRY_MODE, int): return max(1, min(n_features, MTRY_MODE))
    mode = str(MTRY_MODE).lower()
    if mode == "sqrt": return max(1, int(math.sqrt(n_features)))
    if mode == "log2": return max(1, int(math.log2(n_features)))
    return max(1, int(math.sqrt(n_features)))

def test_split(index: int, value: float, dataset):
    left, right = [], []
    for row in dataset:
        (left if row[index] < value else right).append(row)
    return left, right

def gini_index(groups, classes):
    # versión ponderada por CLASS_WEIGHTS
    total_w = sum(sum(CLASS_WEIGHTS[row[-1]] for row in g) for g in groups)
    gini = 0.0
    for group in groups:
        wsize = sum(CLASS_WEIGHTS[row[-1]] for row in group)
        if wsize == 0: continue
        score = 0.0
        for c in classes:
            wc = sum(CLASS_WEIGHTS[row[-1]] for row in group if row[-1]==c)
            p = wc / wsize
            score += p*p
        gini += (1.0 - score) * (wsize / max(1e-12,total_w))
    return gini

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    n_features = len(dataset[0]) - 1
    mtry = choose_mtry(n_features)
    features = random.sample(list(range(n_features)), mtry)

    b_index, b_value, b_score, b_groups = None, None, float("inf"), None
    for index in features:
        values = sorted(set(row[index] for row in dataset))
        # limitar thresholds (K) para acelerar
        if len(values) > K_THRESHOLDS:
            step = len(values) / (K_THRESHOLDS + 1)
            cand = [values[int((i+1)*step)] for i in range(K_THRESHOLDS)]
        else:
            cand = values
        for v in cand:
            groups = test_split(index, v, dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, v, gini, groups
    return {"index": b_index, "value": b_value, "groups": b_groups}

def to_terminal(group):
    labels = [row[-1] for row in group]
    return max(set(labels), key=labels.count)

def split(node, max_depth, min_size, depth):
    left, right = node["groups"]; node.pop("groups", None)
    if not left or not right:
        t = to_terminal(left + right)
        node["left"] = t; node["right"] = t; return
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right); return
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_split(left)
        split(node["left"], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_split(right)
        split(node["right"], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

def predict_tree(node, row_features):
    if row_features[node["index"]] < node["value"]:
        return predict_tree(node["left"], row_features) if isinstance(node["left"], dict) else node["left"]
    else:
        return predict_tree(node["right"], row_features) if isinstance(node["right"], dict) else node["right"]

def subsample(dataset, ratio):
    sample, n = [], max(1, int(round(len(dataset) * ratio)))
    for _ in range(n): sample.append(random.choice(dataset))
    return sample

def subsample_balanced(dataset, ratio=1.0, pos_label=1):
    pos = [r for r in dataset if r[-1]==pos_label]
    neg = [r for r in dataset if r[-1]!=pos_label]
    n_total = max(2, int(round(len(dataset)*ratio)))
    n_pos = min(len(pos), n_total//2)
    n_neg = min(len(neg), n_total - n_pos)
    if n_pos == 0: n_pos = min(1, len(pos))
    if n_neg == 0: n_neg = min(1, len(neg))
    samp = random.sample(pos, n_pos) + random.sample(neg, n_neg)
    random.shuffle(samp)
    return samp

def _train_one_tree(args):
    dataset, max_depth, min_size, sample_size, seed, balanced = args
    random.seed(seed)
    sample = subsample_balanced(dataset, sample_size) if balanced else subsample(dataset, sample_size)
    return build_tree(sample, max_depth, min_size)

def random_forest_train(trainX, trainy, n_trees, max_depth, min_size, sample_size, balanced=True):
    dataset = [x + [int(y)] for x, y in zip(trainX, trainy)]
    trees = []
    t0 = time.perf_counter()
    for i in range(n_trees):
        t1 = time.perf_counter()
        sample = subsample_balanced(dataset, sample_size) if balanced else subsample(dataset, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
        print(f"[RF] Árbol {i+1}/{n_trees} en {time.perf_counter()-t1:.2f}s "
              f"(acum {time.perf_counter()-t0:.2f}s)")
    return trees

def random_forest_train_parallel(trainX, trainy, n_trees, max_depth, min_size, sample_size, balanced=True):
    dataset = [x + [int(y)] for x, y in zip(trainX, trainy)]
    args = [(dataset, max_depth, min_size, sample_size, 1337+i, balanced) for i in range(n_trees)]
    with Pool() as pool:
        trees = pool.map(_train_one_tree, args)
    return trees

def rf_predict_proba(trees, row):
    votes_pos = 0
    for t in trees:
        votes_pos += 1 if predict_tree(t, row) == 1 else 0
    return votes_pos / max(1, len(trees))


# ========= UMBRAL ÓPTIMO =========
def find_best_threshold(trees, Xval, yval, target="f1"):
    best_thr, best_val = 0.5, -1.0
    for k in range(5, 96):  # 0.05..0.95
        thr = k / 100.0
        yhat = [1 if rf_predict_proba(trees, x) >= thr else 0 for x in Xval]
        acc, prec, rec, f1, _ = metrics(yval, yhat)
        score = {"f1": f1, "recall": rec}.get(target, f1)
        if score > best_val:
            best_val, best_thr = score, thr
    return best_thr, best_val


# ========= MAIN =========
def main():
    print("Cargando datos…")
    X, y = load_dataset(CSV_PATH, DEFAULT_FEATURES)

    if HEAD_N is not None:
        X, y = stratified_head(X, y, HEAD_N)
        print(f"[INFO] Usando subset de {len(X)} filas para prototipo.")

    pos = sum(y); neg = len(y)-pos
    print(f"\nTotal filas válidas: {len(X)} | Positivos (muertes): {pos} | Negativos (vivos): {neg}")

    if len(X) == 0:
        print("[ERROR] No se cargaron filas. Revisa el mapeo/CSV_PATH."); sys.exit(1)

    # train / test y dentro del train separamos validación
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_ratio=TEST_RATIO)
    Xtr2, ytr2, Xval, yval = train_test_split(Xtr, ytr, test_ratio=0.12)  # ~10% del total como val

    print(f"Train: {len(Xtr2)} | Val: {len(Xval)} | Test: {len(Xte)}")
    print("Entrenando Random Forest…")

    if USE_PARALLEL:
        trees = random_forest_train_parallel(Xtr2, ytr2, N_TREES, MAX_DEPTH, MIN_SIZE, SAMPLE_SIZE, USE_BALANCED_BOOTSTRAP)
    else:
        trees = random_forest_train(Xtr2, ytr2, N_TREES, MAX_DEPTH, MIN_SIZE, SAMPLE_SIZE, USE_BALANCED_BOOTSTRAP)

    thr, best = find_best_threshold(trees, Xval, yval, target=THRESHOLD_TARGET)
    print(f"\nUmbral óptimo ({THRESHOLD_TARGET.upper()}): {thr:.2f}  (score={best:.4f})")

    # evaluar en test con ese umbral
    yhat = [1 if rf_predict_proba(trees, x) >= thr else 0 for x in Xte]
    acc, prec, rec, f1, (tp, fp, fn, tn) = metrics(yte, yhat)

    print("\n=== Resultados en TEST ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nMatriz de confusión:")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

if __name__ == "__main__":
    main()
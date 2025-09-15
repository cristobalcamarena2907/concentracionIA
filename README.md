# 📘 Regresión Lineal con y sin librerías

Este proyecto implementa un modelo de **Regresión Lineal** para el dataset **California Housing**, usando dos enfoques:

1. **Con librerías (scikit-learn)** → más práctico y eficiente.  
2. **From scratch (NumPy puro)** → implementado manualmente para entender la matemática detrás.

Ambos modelos muestran métricas de evaluación y gráficas de diagnóstico.

---

## ⚙️ Requisitos

Antes de correr los programas, instala las dependencias en un entorno virtual (opcional pero recomendado):

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# En Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install numpy pandas matplotlib scikit-learn
```
1. Modelo con librerías

```
python sklearn-model.py --model ridge --alpha 5.0 --standardize
```

2. Modelo sin librerías
```
python scratch-model.py --use-sklearn-data --solver normal_equation --standardize
```

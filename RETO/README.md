# 🚀 Módulo Reto - Detección de Fraudes

Este módulo corresponde al **reto integrador del semestre**, enfocado en el desarrollo de un sistema de **detección automática de fraudes financieros** mediante algoritmos de **aprendizaje supervisado**.  

El objetivo principal fue diseñar, implementar y evaluar un modelo de machine learning capaz de **maximizar el recall en la clase minoritaria (fraudes)**, manteniendo un equilibrio con la precisión para reducir falsos positivos.

---

## 📂 Contenido

- **Implementacion/**
  - `fraud_detection.py` → Script principal con el pipeline de entrenamiento y predicción.  
  - `outputs_20250918-121632/` → Resultados y métricas generadas durante la ejecución.  

- **Reporte Reto.pdf** → Documento detallado con introducción, metodología, experimentación y resultados.  
- **requirements.txt** → Dependencias necesarias para ejecutar el código.

---

## Reflexión

El sistema actual cumple con el objetivo principal de alcanzar un alto recall en la clase fraudulenta, lo cual es crucial en escenarios donde es preferible detectar más casos de fraude aunque implique algunos falsos positivos. Sin embargo, la precisión relativamente baja indica que aún existen falsos positivos significativos.

- **Posibles mejoras/**
  - Validación más robusta → Aplicar validación cruzada estratificada y técnicas de calibración de probabilidades.
  - Modelos más complejos → Implementar algoritmos como XGBoost, CatBoost o LightGBM para mejorar la separación de clases.
  - Balanceo avanzado de datos → Uso de técnicas como SMOTE combinado con undersampling adaptativo.

---

## 🛠️ Requisitos

Instalar las librerías necesarias:
```bash
pip install -r requirements.txt

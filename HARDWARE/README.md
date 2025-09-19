# ⚙️ Módulo de Hardware: Filtrado y Digitalización

---

## 📂 Contenido de la Carpeta
- **filtrado.py** → Script en Python con la implementación de filtrado (Kalman y Luenberger).  
- **mit-bih-arrhythmia-database-1.0.0.zip** → Dataset utilizado (MIT-BIH Arrhythmia Database).  
- **Reporte Hardware.pdf** → Documento con el análisis, resultados y discusión. 

---

## 📊 Descripción del Proyecto

### 🔹 Dataset
Se empleó el **registro 209 del MIT-BIH Arrhythmia Database**, con frecuencia de muestreo original de **360 Hz** y un tamaño de ~650,000 muestras.  
Las señales corresponden a **ECG** en el rango de **0.5–40 Hz**, donde se encuentra la información fisiológicamente relevante.

### 🔹 Digitalización
1. **Submuestreo** → reducción de la frecuencia de 360 Hz a 60 Hz (Nyquist = 30 Hz).  
2. **Cuantización** → discretización de amplitudes en menos niveles respecto a la señal original.  

### 🔹 Filtros implementados
- **Filtro de Kalman**: estimador recursivo óptimo bajo supuestos de linealidad y ruido gaussiano.  
- **Observador de Luenberger**: estimador determinista basado en realimentación de error.  

---

## 📈 Resultados 
- **Kalman** preserva mejor la morfología de la señal ECG, con menor error y mayor relación señal/ruido.  
- **Luenberger** es más simple computacionalmente, pero amplifica oscilaciones y presenta errores en picos rápidos.  

### Métricas cuantitativas:
- **MSE Kalman**: 0.0484  
- **MSE Luenberger**: 0.0533  
- **SNR Kalman**: 2.72 dB  
- **SNR Luenberger**: 2.31 dB  
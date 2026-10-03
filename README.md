# Minería de Datos – Práctica Experimental 1: Predicción de calidad de vinos (Wine Reviews)

Aplicación del proceso de minería de datos (definición del problema → datos → preprocesamiento → modelado → evaluación) sobre el dataset [Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews) (130 k reseñas de WineEnthusiast).

**Problema:** predecir si un vino obtendrá **≥ 90 puntos** ("alta calidad") a partir de precio, origen, variedad, antigüedad, catador y texto de la reseña; y segmentar el catálogo con K-Means.

## Resultados principales (conjunto de prueba, 23 998 reseñas)

| Modelo | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Baseline (clase mayoritaria) | 0.620 | 0.000 | 0.500 |
| Árbol de decisión | 0.794 | 0.724 | 0.868 |
| Regresión logística (tabular) | 0.806 | 0.734 | 0.885 |
| Random Forest | 0.810 | 0.741 | 0.890 |
| Gradient Boosting (ajustado) | 0.818 | 0.758 | 0.900 |
| **Regresión logística + TF-IDF** | **0.865** | **0.820** | **0.942** |
| Stacking (texto + tabular) | 0.867 | 0.821 | 0.943 |

El texto de la reseña aporta más que cambiar de algoritmo; la longitud de la reseña y el precio son las variables tabulares más influyentes.

## Estructura

```
notebooks/Practica1_Wine_Reviews.ipynb   Cuaderno de Google Colab (todas las fases, con comentarios y análisis)
src/common.py                            Rutas, constantes y estilo
src/01_eda.py                            Exploración y visualizaciones
src/02_preprocess.py                     Limpieza e ingeniería de variables
src/03_clustering.py                     K-Means (modelo descriptivo)
src/pipelines.py                         Preprocesadores y modelos candidatos
src/04_modeling.py                       Validación cruzada, GridSearch y stacking
src/05_evaluation.py                     Métricas en test, curvas, importancia, bootstrap
reports/figures/                         Figuras generadas
results/                                 Métricas y tablas (CSV/JSON)
```

## Reproducir

1. Descargar `winemag-data-130k-v2.csv` desde Kaggle y colocarlo en `data/raw/`.
2. Instalar dependencias y ejecutar en orden:

```bash
pip install -r requirements.txt
python src/01_eda.py
python src/02_preprocess.py
python src/03_clustering.py
python src/04_modeling.py     # ~12 min (incluye stacking)
python src/05_evaluation.py
```

Todo usa `random_state = 42`. En Google Colab basta con abrir `notebooks/Practica1_Wine_Reviews.ipynb` y ejecutar (descarga el dataset con `kagglehub`).

## Decisiones metodológicas

- Se eliminan 9 983 reseñas duplicadas antes de dividir, para evitar fuga entre entrenamiento y prueba.
- Imputación, escalado, codificación y TF-IDF se ajustan solo con entrenamiento (dentro de `Pipeline`).
- Hold-out 80/20 estratificado + validación cruzada de 5 folds; intervalos bootstrap para AUC y F1.
- Limitación: la reseña se escribe después de catar el vino, por lo que el modelo sirve para priorizar y analizar, no para estimar la calidad sin reseña.

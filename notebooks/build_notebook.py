"""Genera notebooks/Practica1_Wine_Reviews.ipynb (formato Google Colab) a partir de las celdas definidas aquí."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# ------------------------------------------------------------------ PORTADA
md("""
# PRÁCTICA EXPERIMENTAL 1 – MINERÍA DE DATOS
## Predicción de la calidad de un vino a partir de sus reseñas (Wine Reviews)

Dataset: https://www.kaggle.com/datasets/zynicide/wine-reviews

**Contenido del cuaderno**
1. Definición del problema
2. Datos: carga, radiografía y análisis exploratorio
3. Preprocesamiento
4. Modelo descriptivo: segmentación con K-Means
5. Modelo predictivo: clasificación (comparación de algoritmos)
6. Evaluación y validación de resultados
7. Conclusiones
""")

# ------------------------------------------------------------------ FASE 1
md("""
# 1. DEFINICIÓN DEL PROBLEMA

**Problema.** Una tienda o distribuidora de vinos recibe miles de referencias y necesita decidir cuáles destacar o comprar. Las reseñas de catadores profesionales (WineEnthusiast) asignan un puntaje de 80 a 100, pero no existen para todos los vinos nuevos.

**Pregunta de investigación.** ¿Se puede predecir si un vino obtendrá **90 puntos o más** ("alta calidad") usando su precio, origen, variedad, antigüedad, catador y el texto de la reseña?

**Objetivos**
- *General:* construir y evaluar modelos de minería de datos que describan y predigan la calidad de los vinos.
- *Específicos:* (1) explorar y depurar el dataset; (2) segmentar los vinos con un modelo descriptivo (K-Means); (3) comparar al menos cinco algoritmos de clasificación; (4) medir qué variables y qué palabras explican la calidad.

**Alcance.** Reseñas de WineEnthusiast hasta 2017 (~130 000 registros, 44 países). Variable objetivo binaria: `high_quality = 1` si `points >= 90`. El modelo no es un sustituto del catador: es una herramienta de priorización.
""")

md("# CONFIGURACIÓN")
code("""
import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

SEED = 42
sns.set_theme(style="darkgrid")
pd.set_option("display.max_columns", 30)
""")

# ------------------------------------------------------------------ FASE 2: DATOS
md("# 2. DATOS\n\nCARGA DEL DATASET DESDE KAGGLE")
code("""
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "winemag-data-130k-v2.csv"

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "zynicide/wine-reviews",
  file_path,
  pandas_kwargs={"index_col": 0},
)

print("Primeros 5 registros:")
df.head()
""")

md("REALIZAR LA RADIOGRAFIA DE MIS DATOS CARGADOS")
code("df.info()")
md("""
análisis El dataset tiene 129 971 filas y 13 columnas. Solo `points` es entero y `price` es numérica (float); el resto son texto. `description` está completa y es la variable con más información (texto libre). Las columnas `region_2`, `taster_twitter_handle`, `taster_name`, `designation`, `region_1` y `price` tienen valores nulos.
""")

code("df.describe()")
md("""
análisis `points` va de 80 a 100 con media 88.45 y desviación 3.04: los puntajes se concentran en la parte baja del rango (mediana 88, 75 % de los vinos ≤ 91). `price` va de 4 a 3 300 USD con media 35.4 y mediana 25: la media es mayor que la mediana, es decir, hay **sesgo fuerte a la derecha** causado por pocos vinos muy caros. Eso justifica transformar el precio con logaritmo más adelante.
""")

md("VALORES FALTANTES POR COLUMNA")
code("""
faltantes = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
faltantes[faltantes > 0]
""")
md("""
análisis `region_2` falta en el 61 % de los registros, por lo que se descartará. `taster_twitter_handle` es solo un alias del catador. `designation` (29 %) y `region_1` (16 %) se convertirán en indicadores binarios (existe / no existe). `price` falta en el 6.9 %: se imputará con la mediana y se agregará un indicador de ausencia, porque que un vino no tenga precio puede ser informativo.
""")

md("REGISTROS DUPLICADOS")
code("""
dup = df.duplicated(subset=["description", "title"]).sum()
print(f"Reseñas duplicadas (misma descripción y título): {dup} ({dup/len(df):.1%})")
""")
md("""
análisis Hay 9 983 reseñas repetidas (7.7 %). Si se dejaran, el mismo vino podría quedar en entrenamiento y en prueba a la vez y los resultados saldrían inflados. Se eliminarán antes de dividir los datos.
""")

md("DISTRIBUCIÓN DEL PUNTAJE Y DE LA VARIABLE OBJETIVO")
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
sns.countplot(x="points", data=df, color="#7B1E3A", ax=ax[0])
ax[0].set_title("Distribución de puntajes")
ax[0].tick_params(axis="x", labelrotation=90)
ax[0].axvline(9.5, color="gold", ls="--", label="umbral 90")
ax[0].legend()

obj = (df.points >= 90).value_counts(normalize=True).rename({False: "< 90", True: ">= 90"})
sns.barplot(x=obj.index, y=obj.values, color="#7B1E3A", ax=ax[1])
ax[1].set_title("Proporción por clase (points >= 90)")
for i, v in enumerate(obj.values):
    ax[1].text(i, v + 0.01, f"{v:.1%}", ha="center")
plt.show()
""")
md("""
análisis La distribución del puntaje es casi simétrica con centro en 87-88 y cola larga hacia 100. Con el umbral de 90 puntos la clase "alta calidad" representa ~37.7 % (≈ 38 % tras depurar duplicados): el problema está **moderadamente desbalanceado**, no es extremo, así que bastan métricas como F1 y AUC además de la exactitud. Un clasificador que siempre diga "< 90" acertaría 62 %: ese es el piso (baseline) que debemos superar.
""")

md("RELACIÓN ENTRE PRECIO Y PUNTAJE")
code("""
d = df.dropna(subset=["price"])
rho, p = stats.spearmanr(d.price, d.points)
print(f"Correlación de Spearman precio-puntos: rho = {rho:.3f} (p = {p:.1e})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(np.log10(d.price), bins=50, color="#7B1E3A", ax=ax[0])
ax[0].set_title("Distribución del precio (escala log10)")
hb = ax[1].hexbin(np.log10(d.price), d.points, gridsize=40, cmap="RdPu", mincnt=1, bins="log")
plt.colorbar(hb, ax=ax[1], label="log10(reseñas)")
ax[1].set_xlabel("log10(precio USD)"); ax[1].set_ylabel("Puntos"); ax[1].set_title("Precio vs. puntaje")
plt.show()
""")
md("""
análisis En escala logarítmica el precio se vuelve casi simétrico. La correlación de Spearman es **0.61**: moderada-fuerte y significativa; los vinos caros tienden a puntuar más, pero hay mucha dispersión (se ven vinos de 15 USD con 92 puntos y vinos de 100 USD con 85). Por eso el precio ayuda pero no alcanza por sí solo para predecir la calidad.
""")

md("PUNTAJE POR PAÍS")
code("""
top_paises = df.country.value_counts().head(10).index
print(df.country.value_counts().head(10))

plt.figure(figsize=(8, 4.5))
orden = df[df.country.isin(top_paises)].groupby("country").points.median().sort_values().index
sns.boxplot(data=df[df.country.isin(top_paises)], y="country", x="points", order=orden, color="#E4B7C3", fliersize=1)
plt.title("Puntaje por país (top 10 por número de reseñas)")
plt.show()

H, p = stats.kruskal(*[df.loc[df.country == c, "points"] for c in top_paises])
print(f"Kruskal-Wallis puntos ~ país: H = {H:.0f}, p = {p:.2e}")
""")
md("""
análisis Estados Unidos concentra el 42 % de las reseñas (54 504), seguido de Francia e Italia. Las medianas difieren entre países (Austria y Alemania con mediana 90 arriba; Argentina y Chile con 86 y España con 87 abajo) y la prueba de Kruskal-Wallis rechaza que sean iguales (p ≈ 0). El país aporta información, aunque con mucho solapamiento entre cajas.
""")

md("PUNTAJE POR CATADOR (POSIBLE SESGO DEL EVALUADOR)")
code("""
tm = df.groupby("taster_name").points.agg(["mean", "count"]).sort_values("mean")
plt.figure(figsize=(7, 5))
sns.barplot(x=tm["mean"], y=tm.index, color="#C9A227")
plt.xlim(84, 91.5); plt.xlabel("Puntos (media)"); plt.ylabel("")
plt.title("Puntaje medio por catador")
plt.show()

H, p = stats.kruskal(*[g.points.values for _, g in df.dropna(subset=["taster_name"]).groupby("taster_name")])
print(f"Kruskal-Wallis puntos ~ catador: H = {H:.0f}, p = {p:.2e}")
""")
md("""
análisis Hay 19 catadores y su puntaje medio va de 85.9 (Alexander Peartree) a 90.6 (Anne Krebiehl MW), casi 5 puntos de diferencia. Parte de eso refleja los vinos que le tocó catar a cada uno, pero también estilos de puntuación distintos. Esto es una advertencia de **sesgo del evaluador**; más adelante se mide cuánto depende el modelo de esta variable.
""")

# ------------------------------------------------------------------ FASE 3: PREPROCESAMIENTO
md("# 3. PREPROCESAMIENTO\n\nELIMINAR DUPLICADOS Y COLUMNAS SIN VALOR")
code("""
n0 = len(df)
data = df.drop_duplicates(subset=["description", "title"]).reset_index(drop=True)
data = data.drop(columns=["taster_twitter_handle", "region_2"])
print(f"Filas: {n0} -> {len(data)} ({n0 - len(data)} duplicadas eliminadas)")
""")
md("""
análisis Se eliminaron 9 983 duplicados (quedan 119 988 reseñas) y dos columnas: `region_2` (61 % nulos) y `taster_twitter_handle` (redundante con `taster_name`).
""")

md("TRATAMIENTO DE FALTANTES CATEGÓRICOS")
code("""
for c in ["country", "province", "taster_name"]:
    data[c] = data[c].fillna("Unknown")
data["description"] = data["description"].str.replace(r"\\s+", " ", regex=True).str.strip()
data[["country", "province", "taster_name"]].isna().sum()
""")
md("""
análisis Los pocos nulos de país/provincia y el 20 % de reseñas sin catador se agrupan en la categoría `Unknown`, para no perder esas filas.
""")

md("PRECIO: RECORTE DE VALORES EXTREMOS, INDICADOR DE AUSENCIA Y LOGARITMO")
code("""
cap = data.price.quantile(0.999)
print(f"Percentil 99.9 del precio: {cap:.0f} USD  |  filas recortadas: {(data.price > cap).sum()}")

data["price_missing"] = data.price.isna().astype(int)
data["log_price"] = np.log1p(data.price.clip(upper=cap))   # la imputación de la mediana se hace dentro del pipeline

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].hist(data.price.dropna().clip(upper=300), bins=50, color="grey"); ax[0].set_title("Precio original (recortado a 300)")
ax[1].hist(data.log_price.dropna(), bins=50, color="#7B1E3A"); ax[1].set_title("log(1 + precio)")
plt.show()
""")
md("""
análisis Solo 111 vinos superan los 476 USD (percentil 99.9); se recortan para que no dominen la escala. El logaritmo convierte la distribución sesgada en una campana casi simétrica, adecuada para modelos lineales y distancias (K-Means). La imputación de la mediana no se hace aquí sino dentro del pipeline de entrenamiento, para evitar **fuga de información** del conjunto de prueba.
""")

md("EDAD DEL VINO: EXTRAER LA COSECHA DESDE EL TÍTULO")
code("""
def cosecha(titulo):
    anios = re.findall(r"\\b(19[5-9]\\d|20[01]\\d)\\b", titulo)
    return int(anios[-1]) if anios else np.nan

REF_YEAR = 2017  # último año de cosecha presente en el dataset
data["year"] = data.title.map(cosecha)
data.loc[data.year > REF_YEAR, "year"] = np.nan
data["age_missing"] = data.year.isna().astype(int)
data["age"] = REF_YEAR - data.year

print(f"Cosecha extraída en el {100 * (1 - data.age_missing.mean()):.1f}% de los vinos")
data.age.describe()
""")
md("""
análisis Se obtuvo la cosecha en el 96.4 % de los títulos mediante una expresión regular. Con ella se creó la variable `age` (2017 − cosecha): la mayoría de vinos tiene pocos años y unos pocos superan los 30, lo que da una variable nueva y útil que no existía en el dataset original.
""")

md("VARIABLES DERIVADAS DEL TEXTO Y DEL METADATO + VARIABLE OBJETIVO")
code("""
data["desc_len_chars"] = data.description.str.len()
data["desc_words"] = data.description.str.split().str.len()
data["has_designation"] = data.designation.notna().astype(int)
data["has_region"] = data.region_1.notna().astype(int)
data["high_quality"] = (data.points >= 90).astype(int)

print(data.high_quality.value_counts(normalize=True).round(3))
print()
print(data.groupby("high_quality")[["desc_len_chars", "desc_words", "price"]].median())
""")
md("""
análisis Tras depurar, la clase positiva (≥ 90 puntos) es el **38.0 %**. Hay un hallazgo importante: los vinos de alta calidad tienen reseñas más largas (mediana de palabras claramente mayor). Los catadores escriben más cuando el vino los impresiona; la longitud de la reseña se convierte en una de las variables más predictivas (se confirma en la evaluación).
""")

# ------------------------------------------------------------------ FASE 4: CLUSTERING
md("# 4. MODELO DESCRIPTIVO: SEGMENTACIÓN CON K-MEANS\n\nELEGIR EL NÚMERO DE CLUSTERS (CODO Y SILUETA)")
code("""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

feats_km = ["points", "log_price", "age", "desc_words"]
Xk = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(data[feats_km]))
idx = np.random.RandomState(SEED).choice(len(Xk), 15000, replace=False)

ks, inercia, sil = range(2, 9), [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=SEED).fit(Xk)
    inercia.append(km.inertia_)
    sil.append(silhouette_score(Xk[idx], km.labels_[idx]))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(ks, inercia, "o-", color="#7B1E3A"); ax[0].set_title("Método del codo"); ax[0].set_xlabel("k")
ax[1].plot(ks, sil, "o-", color="#C9A227"); ax[1].set_title("Coeficiente de silueta"); ax[1].set_xlabel("k")
plt.show()
print({k: round(s, 3) for k, s in zip(ks, sil)})
""")
md("""
análisis La silueta es máxima con k = 2 (0.29) y decae después; con k = 4 vale 0.195 (estructura moderada, los grupos se solapan porque el vino es un continuo de calidad y no categorías separadas). Se elige **k = 4** porque a partir de ahí la silueta ya no cae de forma marcada (0.195 → 0.187) y permite una interpretación comercial más rica que solo "bueno / malo". Es una decisión de interpretabilidad, no de máxima silueta.
""")

md("PERFIL DE CADA CLUSTER")
code("""
km = KMeans(4, n_init=20, random_state=SEED).fit(Xk)
data["cluster"] = km.labels_

perfil = data.groupby("cluster").agg(
    n=("points", "size"), puntos=("points", "mean"), precio_mediano=("price", "median"),
    edad=("age", "mean"), palabras=("desc_words", "mean"), prop_alta_calidad=("high_quality", "mean"),
).round(2)
perfil["pais_top"] = data.groupby("cluster").country.agg(lambda s: s.value_counts().index[0])
perfil["variedad_top"] = data.groupby("cluster").variety.agg(lambda s: s.value_counts().index[0])
perfil["%"] = (100 * perfil.n / len(data)).round(1)
perfil.sort_values("puntos", ascending=False)
""")
md("""
análisis Los cuatro segmentos son interpretables:
- **Premium** (≈ 20 %): 92.2 puntos, precio mediano 60 USD, reseñas largas (52 palabras), 93 % de alta calidad, dominado por Pinot Noir.
- **Buena relación calidad-precio** (≈ 34 %): 89.3 puntos y 27 USD; 46 % supera los 90.
- **Vinos maduros** (≈ 16 %): edad media de 11.9 años, 87.6 puntos, mayormente Cabernet Sauvignon; solo 23 % alta calidad.
- **Económicos** (≈ 30 %): 85.4 puntos, 15 USD y reseñas cortas (31 palabras), 0 % de alta calidad (Chardonnay).

El precio y el largo de la reseña crecen junto con el puntaje, lo que anticipa lo que verá el modelo predictivo.
""")

md("VISUALIZACIÓN DE LOS CLUSTERS (PCA)")
code("""
pca = PCA(2, random_state=SEED).fit(Xk)
Z = pca.transform(Xk[idx])
plt.figure(figsize=(6.4, 4.6))
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=km.labels_[idx], palette="viridis", s=6, alpha=.6, linewidth=0)
plt.xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.0f}% var.)"); plt.ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.0f}% var.)")
plt.title("Clusters K-Means (proyección PCA)"); plt.legend(title="Cluster", markerscale=3)
plt.show()
""")
md("""
análisis En el plano de los dos primeros componentes los clusters forman una banda continua ordenada por calidad, con fronteras que se tocan. Confirma la silueta moderada: la segmentación es útil para describir el catálogo, no para separar grupos "naturales".
""")

# ------------------------------------------------------------------ FASE 5: MODELADO
md("# 5. MODELO PREDICTIVO: CLASIFICACIÓN DE ALTA CALIDAD (≥ 90 PUNTOS)\n\nDIVISIÓN ENTRENAMIENTO / PRUEBA (80 / 20, ESTRATIFICADA)")
code("""
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

NUM = ["log_price", "age", "desc_len_chars", "desc_words"]
BIN = ["price_missing", "age_missing", "has_designation", "has_region"]
CAT = ["country", "province", "variety", "taster_name"]
TEXT = "description"
TARGET = "high_quality"

X, y = data[NUM + BIN + CAT + [TEXT]], data[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print("Entrenamiento:", Xtr.shape, "| Prueba:", Xte.shape)
print("Proporción de positivos:", round(ytr.mean(), 3), "/", round(yte.mean(), 3))
""")
md("""
análisis Se reservó el 20 % (23 998 reseñas) para la prueba final, que no se toca hasta evaluar. La estratificación mantiene 38 % de positivos en ambos conjuntos. Todo el preprocesamiento que aprende de los datos (mediana, escalado, categorías, TF-IDF) se ajusta únicamente con el conjunto de entrenamiento dentro de los pipelines.
""")

md("PIPELINES Y MODELOS CANDIDATOS")
code("""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier

def prep_linear(text=False):
    t = [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
         ("bin", "passthrough", BIN),
         ("cat", OneHotEncoder(min_frequency=200, handle_unknown="infrequent_if_exist"), CAT)]
    if text:
        t.append(("txt", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5,
                                         sublinear_tf=True, stop_words="english"), TEXT))
    return ColumnTransformer(t)

prep_gb = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                           min_frequency=200, encoded_missing_value=np.nan), CAT),
    ("num", "passthrough", NUM), ("bin", "passthrough", BIN)])

modelos = {
    "Baseline (clase mayoritaria)": Pipeline([("prep", prep_linear()), ("clf", DummyClassifier(strategy="most_frequent"))]),
    "Regresión logística (tabular)": Pipeline([("prep", prep_linear()), ("clf", LogisticRegression(max_iter=1000))]),
    "Árbol de decisión": Pipeline([("prep", prep_linear()), ("clf", DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=50, random_state=SEED))]),
    "Random Forest": Pipeline([("prep", prep_linear()), ("clf", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED))]),
    # hiperparámetros obtenidos con GridSearchCV (ver src/04_modeling.py)
    "Gradient Boosting (ajustado)": Pipeline([("prep", prep_gb), ("clf", HistGradientBoostingClassifier(
        learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0, max_iter=300, early_stopping=True,
        validation_fraction=0.1, categorical_features=[0, 1, 2, 3], random_state=SEED))]),
    "Regresión logística + TF-IDF": Pipeline([("prep", prep_linear(text=True)), ("clf", LogisticRegression(
        max_iter=2000, C=1.0))]),
}
list(modelos)
""")
md("""
análisis Se comparan seis enfoques de complejidad creciente: un *baseline* que siempre predice la clase mayoritaria, un modelo lineal, un árbol, un bosque aleatorio, *gradient boosting* (con categorías nativas) y una regresión logística que además lee el texto de la reseña con TF-IDF (unigramas y bigramas). Los hiperparámetros de boosting y de la regresión con texto se eligieron con `GridSearchCV` (3 folds, AUC) en el script `src/04_modeling.py`.
""")

md("VALIDACIÓN CRUZADA (5 FOLDS) SOBRE EL CONJUNTO DE ENTRENAMIENTO")
code("""
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}

filas = []
for nombre, m in modelos.items():
    r = cross_validate(m, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1)
    filas.append({"modelo": nombre,
                  **{f"{k}": f"{r[f'test_{k}'].mean():.4f} ± {r[f'test_{k}'].std():.4f}" for k in scoring},
                  "auc_media": r["test_roc_auc"].mean()})
    print(f"{nombre:34s} AUC = {r['test_roc_auc'].mean():.4f}", flush=True)

cv_df = pd.DataFrame(filas).sort_values("auc_media", ascending=False).drop(columns="auc_media")
cv_df
""")
md("""
análisis Resultados en validación cruzada (AUC): baseline 0.500 → árbol 0.870 → regresión logística 0.886 → Random Forest 0.893 → Gradient Boosting 0.902 → **regresión logística + TF-IDF 0.943**. Las desviaciones entre folds son pequeñas (≈ 0.002), así que las diferencias son estables. Lo más importante: **añadir el texto sube el AUC en ~4 puntos**, más que cualquier cambio de algoritmo sobre las variables tabulares. En este problema la información está en lo que dice la reseña, no en el modelo usado.
""")

# ------------------------------------------------------------------ FASE 6: EVALUACION
md("# 6. EVALUACIÓN\n\nENTRENAR CON TODO EL ENTRENAMIENTO Y MEDIR EN EL CONJUNTO DE PRUEBA")
code("""
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve)

proba, filas = {}, []
for nombre, m in modelos.items():
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    proba[nombre] = p
    pred = (p >= 0.5).astype(int)
    filas.append({"modelo": nombre, "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred, zero_division=0),
                  "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "mcc": matthews_corrcoef(yte, pred),
                  "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p)})

test_df = pd.DataFrame(filas).sort_values("roc_auc", ascending=False).round(4)
test_df
""")
md("""
análisis En el conjunto de prueba (23 998 reseñas nunca vistas) el mejor modelo es la **regresión logística con TF-IDF: exactitud 86.5 %, F1 0.820, AUC 0.942**, frente a 62 % de exactitud del baseline. Los resultados de prueba coinciden con los de validación cruzada (no hay sobreajuste). Con precisión 0.83 y recall 0.81, de cada 100 vinos que el modelo marca como "alta calidad" 83 lo son, y detecta 81 de cada 100 que realmente lo son. Los modelos solo tabulares quedan en AUC 0.87-0.90. Un ensamble por *stacking* (texto + boosting, ver `src/04_modeling.py`) alcanza AUC 0.9425, una mejora de apenas 0.0002, es decir, no justifica su costo.
""")

md("CURVAS ROC Y PRECISIÓN-RECALL")
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for nombre in ["Regresión logística (tabular)", "Random Forest", "Gradient Boosting (ajustado)", "Regresión logística + TF-IDF"]:
    fpr, tpr, _ = roc_curve(yte, proba[nombre]); ax[0].plot(fpr, tpr, label=f"{nombre} ({roc_auc_score(yte, proba[nombre]):.3f})")
    pr, rc, _ = precision_recall_curve(yte, proba[nombre]); ax[1].plot(rc, pr, label=nombre)
ax[0].plot([0, 1], [0, 1], "--", color="lightgrey"); ax[0].set_title("Curvas ROC (test)"); ax[0].legend(fontsize=7, loc="lower right")
ax[0].set_xlabel("Tasa de falsos positivos"); ax[0].set_ylabel("Tasa de verdaderos positivos")
ax[1].axhline(yte.mean(), ls="--", color="lightgrey"); ax[1].set_title("Precisión-Recall (test)")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precisión")
plt.show()
""")
md("""
análisis La curva del modelo con texto domina a las demás en todo el rango: para cualquier tasa de falsos positivos, detecta más vinos buenos. En Precisión-Recall (más informativa con clases desbalanceadas) mantiene precisión 0.90 con recall 0.70 y 0.84 con recall 0.80, muy por encima de la línea base de 0.38. Por ejemplo, con 5 % de falsos positivos detecta 70 % de los vinos buenos, frente a 54 % del boosting tabular.
""")

md("MATRIZ DE CONFUSIÓN DEL MEJOR MODELO")
code("""
mejor = "Regresión logística + TF-IDF"
cm = confusion_matrix(yte, (proba[mejor] >= 0.5).astype(int))
plt.figure(figsize=(4.4, 3.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", cbar=False, xticklabels=["< 90", ">= 90"], yticklabels=["< 90", ">= 90"])
plt.xlabel("Predicho"); plt.ylabel("Real"); plt.title(mejor)
plt.show()
""")
md("""
análisis El modelo comete más errores del tipo "falso negativo" (vinos de ≥ 90 clasificados como < 90) que "falso positivo", coherente con un recall (0.81) algo menor que la precisión (0.83). Hay 1 773 falsos negativos frente a 1 458 falsos positivos. Los errores se concentran en los vinos cercanos al umbral: el 72 % de los errores corresponde a vinos de 89, 90 o 91 puntos (tasa de error de 33 % en los de 89 y 43 % en los de 90, frente a menos de 3 % por debajo de 86), donde incluso dos catadores podrían discrepar.
""")

md("QUÉ PALABRAS DISTINGUEN UN VINO DE ≥ 90 PUNTOS")
code("""
lr = modelos[mejor]
nombres = lr.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=nombres)
coef = coef[coef.index.str.startswith("txt__")]; coef.index = coef.index.str[5:]

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
coef.nlargest(15)[::-1].plot.barh(color="#7B1E3A", ax=ax[0]); ax[0].set_title("Términos asociados a >= 90 puntos")
coef.nsmallest(15)[::-1].plot.barh(color="grey", ax=ax[1]); ax[1].set_title("Términos asociados a < 90 puntos")
plt.show()
""")
md("""
análisis El modelo aprendió el vocabulario del catador. Suman hacia alta calidad: *complex, beautiful, delicious, long, impressive, elegant, rich, gorgeous, lush*. Restan: *simple, lacks, straightforward, somewhat, bit, little, easy, short, astringent*. Es lógico: adjetivos de riqueza y complejidad vs. calificativos de ausencia o atenuación. Los años 2020 y 2022 también suman: son ventanas de consumo ("drink through 2022"), propias de vinos con potencial de guarda. Se verificó que los números 80-100 aparecen en solo 3.3 % de las descripciones y casi siempre son porcentajes de mezcla de uvas (no el puntaje), por lo que **no hay fuga directa de la etiqueta**.
""")

md("IMPORTANCIA DE LAS VARIABLES TABULARES (PERMUTACIÓN)")
code("""
from sklearn.inspection import permutation_importance

gb = modelos["Gradient Boosting (ajustado)"]
muestra = np.random.RandomState(SEED).choice(len(Xte), 6000, replace=False)
feats = NUM + BIN + CAT
pi = permutation_importance(gb, Xte.iloc[muestra][feats + [TEXT]], yte.iloc[muestra], scoring="roc_auc",
                            n_repeats=5, random_state=SEED, n_jobs=1)
imp = pd.Series(pi.importances_mean, index=feats + [TEXT]).drop(TEXT).sort_values()

plt.figure(figsize=(6.4, 4.4))
imp.plot.barh(color="#7B1E3A")
plt.xlabel("Caída del AUC al permutar la variable"); plt.title("Importancia por permutación (Gradient Boosting)")
plt.show()
""")
md("""
análisis La variable más influyente es la **longitud de la reseña** (`desc_len_chars`: el AUC cae 0.23 al permutarla), seguida del **precio** (0.16). Muy por detrás: catador (0.024), provincia (0.017), edad (0.014) y variedad (0.014). El país casi no aporta una vez conocidos precio y provincia. Un modelo de boosting entrenado **sin** el catador pierde solo 0.003 de AUC (0.8998 → 0.8965): el modelo no depende de quién cató el vino, lo que atenúa la preocupación por el sesgo del evaluador.
""")

md("ANÁLISIS DE ERRORES SEGÚN RANGO DE PRECIO")
code("""
e = Xte.copy()
e["y"] = yte.values
e["pred"] = (proba[mejor] >= 0.5).astype(int)
e["error"] = (e.y != e.pred).astype(int)
e["rango_precio"] = pd.cut(np.expm1(e.log_price), [0, 15, 30, 60, 120, 1e5], labels=["<15", "15-30", "30-60", "60-120", ">120"])
print("Tasa de error por rango de precio (USD):")
print(e.groupby("rango_precio", observed=True).error.mean().round(3))
print("\\nError en vinos sin precio:", round(e[e.price_missing == 1].error.mean(), 3))
""")
md("""
análisis El modelo falla poco en los extremos (4 % de error en vinos < 15 USD, que casi nunca llegan a 90; 7.5 % en > 120 USD, que casi siempre sí) y falla más en la zona media de 15 a 60 USD (16-18 %), donde conviven vinos buenos y regulares. Es el comportamiento esperado: la incertidumbre está en el medio.
""")

# ------------------------------------------------------------------ CONCLUSIONES
md("""
# 7. CONCLUSIONES

1. **Es posible predecir la alta calidad de un vino con buena precisión**: el mejor modelo (regresión logística + TF-IDF) logra 86.5 % de exactitud, F1 0.82 y AUC 0.94 en datos no vistos, frente a 62 % del baseline.
2. **El texto de la reseña aporta más que cualquier algoritmo sofisticado**: pasar de variables tabulares a texto mejora el AUC de ~0.90 a ~0.94; cambiar de árbol a boosting solo aporta ~0.03.
3. **Las variables más influyentes son la longitud de la reseña y el precio**; el catador pesa poco una vez conocidas las demás variables.
4. **La segmentación K-Means** produce cuatro perfiles interpretables (Premium, Calidad-precio, Maduros, Económicos), aunque con fronteras difusas (silueta 0.195).
5. **Limitaciones:** el dataset solo cubre reseñas de una revista hasta 2017; el puntaje es subjetivo; la reseña se escribe *después* de catar el vino, por lo que el modelo sirve para analizar y priorizar, no para predecir la calidad de un vino sin reseña; y el umbral de 90 puntos es una decisión de negocio.
6. **Trabajo futuro:** embeddings de lenguaje (BERT), regresión sobre el puntaje exacto y validación con reseñas posteriores a 2017.

Código completo y reproducible en el repositorio de GitHub (carpeta `src/`).
""")

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path(__file__).parent / "Practica1_Wine_Reviews.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("escrito", out, len(cells), "celdas")

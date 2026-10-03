"""Fase 3 - Preprocesamiento: limpieza, transformación e ingeniería de variables.

La imputación, el escalado y la codificación se hacen dentro de los pipelines de
scikit-learn (fit solo sobre entrenamiento) para evitar fuga de información.
"""
import re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from common import *

set_style()
raw = pd.read_csv(RAW, index_col=0)
log = {"rows_raw": len(raw)}

df = raw.copy()
# 1) Duplicados: la misma reseña (descripción + título) aparece varias veces
df = df.drop_duplicates(subset=["description", "title"]).reset_index(drop=True)
log["rows_after_dedup"] = len(df); log["duplicates_removed"] = log["rows_raw"] - len(df)

# 2) Columnas descartadas: sin valor predictivo o con >60% de faltantes / fuga de identidad
df = df.drop(columns=["taster_twitter_handle", "region_2"])

# 3) Texto: normalización básica de espacios
for c in ["country", "province", "variety", "taster_name", "designation", "region_1", "winery"]:
    df[c] = df[c].astype("string").str.strip()
df["description"] = df["description"].str.replace(r"\s+", " ", regex=True).str.strip()

# 4) Faltantes categóricos
df["country"] = df["country"].fillna("Unknown")
df["province"] = df["province"].fillna("Unknown")
df["taster_name"] = df["taster_name"].fillna("Unknown")

# 5) Precio: outliers extremos recortados al percentil 99.9 antes del log
cap = df.price.quantile(0.999)
log["price_cap_p999"] = float(cap)
log["price_capped_rows"] = int((df.price > cap).sum())
df["price_missing"] = df.price.isna().astype(int)
df["log_price"] = np.log1p(df.price.clip(upper=cap))

# 6) Año de cosecha desde el título
def vintage(t):
    m = re.findall(r"\b(19[5-9]\d|20[01]\d)\b", t)
    return int(m[-1]) if m else np.nan
df["year"] = df.title.map(vintage)
df.loc[df.year > REF_YEAR, "year"] = np.nan
df["age_missing"] = df.year.isna().astype(int)
df["age"] = REF_YEAR - df.year

# 7) Variables derivadas del texto y metadatos
df["desc_len_chars"] = df.description.str.len()
df["desc_words"] = df.description.str.split().str.len()
df["has_designation"] = df.designation.notna().astype(int)
df["has_region"] = df.region_1.notna().astype(int)

# 8) Variable objetivo
df[TARGET] = (df.points >= THRESHOLD).astype(int)

keep = ["title", "description", "points", TARGET, "price", *NUM, *BIN, *CAT]
df = df[keep]
CLEAN.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN, index=False)

log.update(rows_final=len(df), share_high_quality=float(df[TARGET].mean()),
           missing_after={c: float(df[c].isna().mean()) for c in ["log_price", "age"]},
           age_range=[float(df.age.min()), float(df.age.max())],
           desc_words_mean=float(df.desc_words.mean()),
           year_extracted_pct=float(100 * (1 - df.age_missing.mean())))
save_json(log, "preprocess.json")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].hist(raw.price.dropna().clip(upper=300), bins=50, color=GREY); ax[0].set_title("Precio original (recortado a 300)")
ax[1].hist(df.log_price.dropna(), bins=50, color=WINE); ax[1].set_title("log(1 + precio) tras recorte")
ax[2].hist(df.age.dropna(), bins=range(0, 40), color=GOLD); ax[2].set_title("Edad del vino (2017 - cosecha)")
save_fig(fig, "06_preprocess")
print(log)

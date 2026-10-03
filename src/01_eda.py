"""Fase 2 - Datos: exploración, estadísticos descriptivos y visualizaciones."""
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from scipy import stats
from common import *

set_style()
df = pd.read_csv(RAW, index_col=0)
out = {"shape": list(df.shape)}

# --- calidad de datos
miss = df.isna().mean().sort_values(ascending=False) * 100
out["missing_pct"] = miss.round(2).to_dict()
out["duplicates_desc_title"] = int(df.duplicated(subset=["description", "title"]).sum())
out["n_countries"] = int(df.country.nunique()); out["n_varieties"] = int(df.variety.nunique())
out["n_provinces"] = int(df.province.nunique()); out["n_tasters"] = int(df.taster_name.nunique())
out["points"] = df.points.describe().round(3).to_dict()
out["price"] = df.price.describe().round(3).to_dict()
out["share_high_quality_raw"] = float((df.points >= THRESHOLD).mean())
out["top_countries"] = df.country.value_counts().head(10).to_dict()
out["top_varieties"] = df.variety.value_counts().head(10).to_dict()

d = df.dropna(subset=["price"])
rho, p = stats.spearmanr(d.price, d.points)
out["spearman_price_points"] = {"rho": rho, "p": p}
rho2, _ = stats.spearmanr(np.log(d.price), d.points)
out["price_q"] = d.price.quantile([.5, .9, .99, .999]).to_dict()
# Kruskal-Wallis puntos ~ país (top 10)
top = df.country.value_counts().head(10).index
kw = stats.kruskal(*[df.loc[df.country == c, "points"] for c in top])
out["kruskal_points_country"] = {"H": kw.statistic, "p": kw.pvalue}
tk = df.taster_name.dropna().unique()
kw2 = stats.kruskal(*[df.loc[df.taster_name == t, "points"] for t in tk])
out["kruskal_points_taster"] = {"H": kw2.statistic, "p": kw2.pvalue}
out["taster_mean_points"] = df.groupby("taster_name").points.mean().round(2).sort_values().to_dict()
save_json(out, "eda.json")

# --- figuras
fig, ax = plt.subplots(figsize=(7, 3.6))
m = miss[miss > 0].sort_values()
ax.barh(m.index, m.values, color=WINE); ax.set_xlabel("% de valores faltantes")
ax.set_title("Valores faltantes por variable")
for i, v in enumerate(m.values): ax.text(v + .5, i, f"{v:.1f}%", va="center", fontsize=8)
save_fig(fig, "01_missing")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
sns.countplot(x="points", data=df, color=WINE, ax=ax[0]); ax[0].set_title("Distribución de puntajes")
ax[0].set_xticks(range(0, 21, 2)); ax[0].set_xticklabels(range(80, 101, 2)); ax[0].axvline(9.5, color=GOLD, ls="--", label="umbral 90")
ax[0].legend(); ax[0].set_ylabel("Reseñas")
sns.histplot(np.log10(d.price), bins=50, color=WINE, ax=ax[1]); ax[1].set_title("Distribución del precio (escala log)")
ax[1].set_xlabel("log10(precio USD)")
save_fig(fig, "02_points_price_dist")

fig, ax = plt.subplots(figsize=(6.2, 4.4))
hb = ax.hexbin(np.log10(d.price), d.points, gridsize=40, cmap="RdPu", mincnt=1, bins="log")
plt.colorbar(hb, label="log10(reseñas)"); ax.set_xlabel("log10(precio USD)"); ax.set_ylabel("Puntos")
ax.set_title(f"Precio vs. puntaje (Spearman ρ = {rho:.2f})")
save_fig(fig, "03_price_vs_points")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cc = df.country.value_counts().head(10)
sns.barplot(x=cc.values, y=cc.index, color=WINE, ax=ax[0]); ax[0].set_title("Reseñas por país (top 10)"); ax[0].set_xlabel("Reseñas")
order = df[df.country.isin(top)].groupby("country").points.median().sort_values().index
sns.boxplot(data=df[df.country.isin(top)], y="country", x="points", order=order, color="#E4B7C3", fliersize=1, ax=ax[1])
ax[1].set_title("Puntaje por país (top 10)"); ax[1].set_ylabel("")
save_fig(fig, "04_countries")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
vv = df.variety.value_counts().head(12)
sns.barplot(x=vv.values, y=vv.index, color=WINE, ax=ax[0]); ax[0].set_title("Variedades más frecuentes"); ax[0].set_xlabel("Reseñas")
tm = df.groupby("taster_name").points.agg(["mean", "count"]).sort_values("mean")
sns.barplot(x=tm["mean"], y=tm.index, color=GOLD, ax=ax[1]); ax[1].set_xlim(84, 91.5)
ax[1].set_title("Puntaje medio por catador"); ax[1].set_xlabel("Puntos (media)"); ax[1].set_ylabel("")
save_fig(fig, "05_varieties_tasters")
print("EDA ok", out["shape"], "dups", out["duplicates_desc_title"], "rho", round(rho, 3))

"""Modelo descriptivo: segmentación de vinos con K-Means."""
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from common import *

set_style()
df = pd.read_csv(CLEAN)
feats = ["points", "log_price", "age", "desc_words"]
X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[feats]))
rng = np.random.RandomState(SEED)
idx = rng.choice(len(X), 15000, replace=False)

ks = range(2, 9); inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=SEED).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X[idx], km.labels_[idx]))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(ks, inertia, "o-", color=WINE); ax[0].set_title("Método del codo"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inercia")
ax[1].plot(ks, sil, "o-", color=GOLD); ax[1].set_title("Coeficiente de silueta"); ax[1].set_xlabel("k")
save_fig(fig, "07_kmeans_selection")

K = 4
km = KMeans(K, n_init=20, random_state=SEED).fit(X)
df["cluster"] = km.labels_
prof = df.groupby("cluster").agg(n=("points", "size"), points=("points", "mean"), price_median=("price", "median"),
                                 age=("age", "mean"), desc_words=("desc_words", "mean"), high_quality=(TARGET, "mean")).round(2)
prof["country_top"] = df.groupby("cluster").country.agg(lambda s: s.value_counts().index[0])
prof["variety_top"] = df.groupby("cluster").variety.agg(lambda s: s.value_counts().index[0])
prof["share_pct"] = (100 * prof.n / len(df)).round(1)
# etiquetas ordenadas por puntaje
prof = prof.sort_values("points", ascending=False)
prof.to_csv(RESULTS / "cluster_profiles.csv")
save_json({"k": K, "silhouette_by_k": dict(zip(map(int, ks), map(float, sil))), "silhouette_final": float(sil[K - 2]),
           "profiles": prof.reset_index().to_dict(orient="records")}, "clustering.json")

pca = PCA(2, random_state=SEED).fit(X)
Z = pca.transform(X[idx])
fig, ax = plt.subplots(figsize=(6.4, 4.6))
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=km.labels_[idx], palette=PALETTE[:K], s=6, alpha=.6, linewidth=0, ax=ax)
ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.0f}% var.)"); ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.0f}% var.)")
ax.set_title("Clusters K-Means (proyección PCA)"); ax.legend(title="Cluster", markerscale=3)
save_fig(fig, "08_kmeans_pca")
print(prof.to_string()); print("silhouette", sil)

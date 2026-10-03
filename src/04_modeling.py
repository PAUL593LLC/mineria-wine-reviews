"""Fase 4 - Modelado: comparación por validación cruzada, ajuste de hiperparámetros y ensamble."""
import time, joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from common import *
from pipelines import *

set_style()
MODELS.mkdir(exist_ok=True)
df = pd.read_csv(CLEAN)
FEATS = NUM + BIN + CAT + [TEXT]
X, y = df[FEATS], df[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
joblib.dump((Xtr, Xte, ytr, yte), MODELS / "split.joblib")
print("train", Xtr.shape, "test", Xte.shape, "pos rate", round(ytr.mean(), 3), round(yte.mean(), 3))

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
rows = []


def run_cv(name, model):
    t = time.time()
    r = cross_validate(model, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1)
    rows.append({"modelo": name, **{f"{m}_mean": r[f"test_{m}"].mean() for m in scoring},
                 **{f"{m}_std": r[f"test_{m}"].std() for m in scoring}, "fit_s": r["fit_time"].mean()})
    print(f"{name:42s} acc={rows[-1]['accuracy_mean']:.4f} f1={rows[-1]['f1_mean']:.4f} "
          f"auc={rows[-1]['roc_auc_mean']:.4f} ({time.time()-t:.0f}s)", flush=True)


# 1) Comparación base
for name, m in candidates().items():
    run_cv(name, m)

# 2) Ajuste de hiperparámetros (GridSearch 3-fold) de los dos mejores enfoques
t = time.time()
gs_hgb = GridSearchCV(hgb(), {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31, 63],
                             "clf__l2_regularization": [0.0, 1.0]}, scoring="roc_auc", cv=3, n_jobs=1, refit=False)
gs_hgb.fit(Xtr, ytr)
print("HGB best", gs_hgb.best_params_, round(gs_hgb.best_score_, 4), f"{time.time()-t:.0f}s", flush=True)
t = time.time()
gs_lr = GridSearchCV(Pipeline([("prep", prep_linear(text=True)), ("clf", LogisticRegression(max_iter=2000))]),
                     {"clf__C": [0.25, 1.0, 4.0]}, scoring="roc_auc", cv=3, n_jobs=1, refit=False)
gs_lr.fit(Xtr, ytr)
print("LR+TFIDF best", gs_lr.best_params_, round(gs_lr.best_score_, 4), f"{time.time()-t:.0f}s", flush=True)
save_json({"hgb": {"params": gs_hgb.best_params_, "cv_auc": gs_hgb.best_score_},
           "lr_tfidf": {"params": gs_lr.best_params_, "cv_auc": gs_lr.best_score_}}, "tuning.json")

bp = gs_hgb.best_params_
hp = dict(learning_rate=bp["clf__learning_rate"], max_leaf_nodes=bp["clf__max_leaf_nodes"],
          l2_regularization=bp["clf__l2_regularization"])
cat_nt = [c for c in CAT if c != "taster_name"]


def lr_best():
    return Pipeline([("prep", prep_linear(text=True)),
                     ("clf", LogisticRegression(max_iter=2000, C=gs_lr.best_params_["clf__C"]))])


run_cv("Gradient Boosting (ajustado)", hgb(**hp))
run_cv("Regresión logística + TF-IDF (ajustada)", lr_best())
# 3) Ablación: sin el catador
run_cv("Gradient Boosting (ajustado, sin catador)", hgb(cat_nt, **hp))


# 4) Ensamble por stacking: texto (LR) + tabular (GB) -> meta-clasificador LR
def make_stack():
    return StackingClassifier(estimators=[("texto", lr_best()), ("tabular", hgb(**hp))],
                              final_estimator=LogisticRegression(max_iter=1000), stack_method="predict_proba", cv=5)


run_cv("Stacking (texto + tabular)", make_stack())

cvdf = pd.DataFrame(rows)
cvdf.to_csv(RESULTS / "cv_results.csv", index=False)

# Reentrenar los candidatos finales sobre todo el conjunto de entrenamiento
cands = candidates()
final = {"Baseline": cands["Baseline (clase mayoritaria)"],
         "Regresión logística (tabular)": cands["Regresión logística (tabular)"],
         "Árbol de decisión": cands["Árbol de decisión"], "Random Forest": cands["Random Forest"],
         "Gradient Boosting (ajustado)": hgb(**hp), "Regresión logística + TF-IDF (ajustada)": lr_best(),
         "Gradient Boosting (sin catador)": hgb(cat_nt, **hp), "Stacking (texto + tabular)": make_stack()}
for n, m in final.items():
    m.fit(Xtr, ytr)
joblib.dump(final, MODELS / "final_models.joblib")

fig, ax = plt.subplots(figsize=(8.5, 4.6))
d = cvdf.sort_values("roc_auc_mean")
ax.barh(d.modelo, d.roc_auc_mean, xerr=d.roc_auc_std, color=WINE, ecolor=GREY)
ax.set_xlim(0.45, 1.0)
ax.set_xlabel("AUC-ROC (media ± desv., CV 5-fold)")
ax.set_title("Comparación de modelos en validación cruzada")
for i, v in enumerate(d.roc_auc_mean):
    ax.text(v + .012, i, f"{v:.3f}", va="center", fontsize=8)
save_fig(fig, "09_cv_comparison")
print("done")

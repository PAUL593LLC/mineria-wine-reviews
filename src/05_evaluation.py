"""Fase 5 - Evaluación: métricas en test, curvas, matriz de confusión, importancia e intervalos bootstrap."""
import joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss, matthews_corrcoef)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from common import *

set_style()
Xtr, Xte, ytr, yte = joblib.load(MODELS / "split.joblib")
models = joblib.load(MODELS / "final_models.joblib")
proba = {n: m.predict_proba(Xte)[:, 1] for n, m in models.items()}

rows = []
for n, p in proba.items():
    pred = (p >= 0.5).astype(int)
    rows.append({"modelo": n, "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred, zero_division=0),
                 "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "mcc": matthews_corrcoef(yte, pred),
                 "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p), "brier": brier_score_loss(yte, p)})
res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
res.to_csv(RESULTS / "test_metrics.csv", index=False)
print(res.round(4).to_string(index=False))
best = "Regresión logística + TF-IDF (ajustada)"   # modelo final adoptado (AUC ≈ stacking, mucho más simple)
best_gb = "Gradient Boosting (ajustado)"

# --- curvas ROC y PR
show = ["Regresión logística (tabular)", "Random Forest", best_gb, "Regresión logística + TF-IDF (ajustada)", "Stacking (texto + tabular)"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for n, c in zip(show, PALETTE):
    fpr, tpr, _ = roc_curve(yte, proba[n]); ax[0].plot(fpr, tpr, color=c, label=f"{n} ({roc_auc_score(yte, proba[n]):.3f})")
    pr, rc, _ = precision_recall_curve(yte, proba[n]); ax[1].plot(rc, pr, color=c, label=n)
ax[0].plot([0, 1], [0, 1], "--", color="lightgrey"); ax[0].set_xlabel("Tasa de falsos positivos"); ax[0].set_ylabel("Tasa de verdaderos positivos")
ax[0].set_title("Curvas ROC (test)"); ax[0].legend(fontsize=7, loc="lower right")
ax[1].axhline(yte.mean(), ls="--", color="lightgrey"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precisión"); ax[1].set_title("Curvas Precisión-Recall (test)")
save_fig(fig, "10_roc_pr")

# --- matriz de confusión del mejor modelo
cm = confusion_matrix(yte, (proba[best] >= 0.5).astype(int))
fig, ax = plt.subplots(figsize=(4.4, 3.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", cbar=False, xticklabels=["< 90", "≥ 90"], yticklabels=["< 90", "≥ 90"], ax=ax)
ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(f"Matriz de confusión\n{best}", fontsize=9)
save_fig(fig, "11_confusion")

# --- calibración
fig, ax = plt.subplots(figsize=(4.8, 4))
for n, c in zip([best_gb, "Regresión logística + TF-IDF (ajustada)", "Stacking (texto + tabular)"], PALETTE[2:]):
    fp, mp = calibration_curve(yte, proba[n], n_bins=10); ax.plot(mp, fp, "o-", color=c, label=n, ms=4)
ax.plot([0, 1], [0, 1], "--", color="lightgrey"); ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Fracción de positivos")
ax.set_title("Curva de calibración"); ax.legend(fontsize=6.5)
save_fig(fig, "12_calibration")

# --- importancia por permutación (variables originales) del Gradient Boosting
idx = np.random.RandomState(SEED).choice(len(Xte), 6000, replace=False)
feats = NUM + BIN + CAT
pi = permutation_importance(models[best_gb], Xte.iloc[idx][feats + [TEXT]], yte.iloc[idx], scoring="roc_auc", n_repeats=5,
                            random_state=SEED, n_jobs=1)
imp = pd.Series(pi.importances_mean, index=feats + [TEXT]).drop(TEXT).sort_values()
imp.to_csv(RESULTS / "permutation_importance.csv", header=["delta_auc"])
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.barh(imp.index, imp.values, color=WINE); ax.set_xlabel("Caída del AUC al permutar la variable")
ax.set_title("Importancia por permutación (Gradient Boosting)")
save_fig(fig, "13_permutation_importance")

# --- palabras más influyentes en la regresión logística con TF-IDF
lr = models["Regresión logística + TF-IDF (ajustada)"]
names = lr.named_steps["prep"].get_feature_names_out(); coef = lr.named_steps["clf"].coef_[0]
txt = pd.Series(coef, index=names); txt = txt[txt.index.str.startswith("txt__")]; txt.index = txt.index.str[5:]
top_pos, top_neg = txt.nlargest(15), txt.nsmallest(15)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ax[0].barh(top_pos.index[::-1], top_pos.values[::-1], color=WINE); ax[0].set_title("Términos asociados a ≥ 90 puntos")
ax[1].barh(top_neg.index[::-1], top_neg.values[::-1], color=GREY); ax[1].set_title("Términos asociados a < 90 puntos")
for a in ax: a.set_xlabel("Coeficiente (regresión logística)")
save_fig(fig, "14_text_terms")

# --- bootstrap del AUC y F1 del mejor modelo; diferencia pareada GB vs texto
rng = np.random.RandomState(SEED); yv = yte.values; n = len(yv)
def boot(fn, B=300):
    out = []
    for _ in range(B):
        i = rng.randint(0, n, n); out.append(fn(i))
    return np.percentile(out, [2.5, 97.5]).tolist()
pb, pg, pt = proba[best], proba[best_gb], proba["Regresión logística + TF-IDF (ajustada)"]
ci = {"best": best,
      "auc_ci95": boot(lambda i: roc_auc_score(yv[i], pb[i])),
      "f1_ci95": boot(lambda i: f1_score(yv[i], (pb[i] >= .5).astype(int))),
      "auc_diff_final_minus_gb_ci95": boot(lambda i: roc_auc_score(yv[i], pb[i]) - roc_auc_score(yv[i], pg[i])),
      "auc_diff_gb_minus_text_ci95": boot(lambda i: roc_auc_score(yv[i], pg[i]) - roc_auc_score(yv[i], pt[i]))}

# --- análisis de errores: error por rango de precio y por catador
e = Xte.copy(); e["y"] = yte.values; e["pred"] = (pb >= .5).astype(int); e["err"] = (e.y != e.pred).astype(int)
e["price_band"] = pd.cut(np.expm1(e.log_price), [0, 15, 30, 60, 120, 1e5], labels=["<15", "15-30", "30-60", "60-120", ">120"])
ci["error_by_price_band"] = e.groupby("price_band", observed=True).err.mean().round(3).to_dict()
ci["error_price_missing"] = float(e[e.price_missing == 1].err.mean())
ci["error_by_taster"] = e.groupby("taster_name").err.mean().round(3).sort_values().to_dict()
ci["top_text_terms_pos"] = top_pos.round(2).to_dict(); ci["top_text_terms_neg"] = top_neg.round(2).to_dict()
ci["perm_importance"] = imp.round(4).sort_values(ascending=False).to_dict()
save_json(ci, "evaluation.json")
print(ci["auc_ci95"], ci["f1_ci95"], ci["auc_diff_final_minus_gb_ci95"], ci["auc_diff_gb_minus_text_ci95"])
print(ci["perm_importance"])

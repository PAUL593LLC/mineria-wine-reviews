"""Definición de preprocesadores y modelos candidatos."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from common import *


def _num():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])


def _ohe():
    return OneHotEncoder(min_frequency=200, handle_unknown="infrequent_if_exist")


def prep_linear(text=False, cat=CAT):
    """Numéricas imputadas+escaladas, binarias, categóricas one-hot (+ TF-IDF opcional)."""
    t = [("num", _num(), NUM), ("bin", "passthrough", BIN), ("cat", _ohe(), cat)]
    if text:
        t.append(("txt", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5, sublinear_tf=True,
                                         stop_words="english"), TEXT))
    return ColumnTransformer(t)


def prep_hgb(cat=CAT):
    """Para boosting: categóricas ordinales tratadas como categóricas nativas."""
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, min_frequency=200,
                        encoded_missing_value=np.nan)
    return ColumnTransformer([("cat", oe, cat), ("num", "passthrough", NUM), ("bin", "passthrough", BIN)])


def hgb(cat=CAT, **kw):
    params = dict(learning_rate=0.1, max_leaf_nodes=31, l2_regularization=0.0, max_iter=300,
                  early_stopping=True, validation_fraction=0.1, random_state=SEED,
                  categorical_features=list(range(len(cat))))
    params.update(kw)
    return Pipeline([("prep", prep_hgb(cat)), ("clf", HistGradientBoostingClassifier(**params))])


def candidates():
    return {
        "Baseline (clase mayoritaria)": Pipeline([("prep", prep_linear()), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "Regresión logística (tabular)": Pipeline([("prep", prep_linear()), ("clf", LogisticRegression(max_iter=1000, C=1.0))]),
        "Árbol de decisión": Pipeline([("prep", prep_linear()), ("clf", DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, random_state=SEED))]),
        "Random Forest": Pipeline([("prep", prep_linear()), ("clf", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED))]),
        "Gradient Boosting (tabular)": hgb(),
        "Regresión logística + TF-IDF": Pipeline([("prep", prep_linear(text=True)), ("clf", LogisticRegression(
            max_iter=2000, C=1.0))]),
    }

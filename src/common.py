"""Rutas, constantes y utilidades compartidas por todas las fases."""
from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "winemag-data-130k-v2.csv"
CLEAN = ROOT / "data" / "processed" / "wines_clean.csv"
RESULTS = ROOT / "results"
FIGS = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

SEED = 42
THRESHOLD = 90          # puntos >= 90 -> "alta calidad"
REF_YEAR = 2017         # último año de cosecha presente en el dataset

WINE = "#7B1E3A"        # color principal (vino tinto)
GOLD = "#C9A227"
GREY = "#6B7280"
PALETTE = [WINE, "#2F6F8F", GOLD, "#4C8C4A", "#8E5BA8", "#D2691E", GREY]

NUM = ["log_price", "age", "desc_len_chars", "desc_words"]
BIN = ["price_missing", "age_missing", "has_designation", "has_region"]
CAT = ["country", "province", "variety", "taster_name"]
TEXT = "description"
TARGET = "high_quality"


def set_style():
    sns.set_theme(style="whitegrid", context="notebook", palette=PALETTE)
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False})


def save_fig(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def save_json(obj, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))

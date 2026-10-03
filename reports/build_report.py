"""Genera reports/Informe_Practica1_Wine_Reviews.pdf: documento integrado + artículo IEEE (2 columnas)."""
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, NextPageTemplate, PageBreak, Paragraph, Spacer,
                                Image, Table, TableStyle, KeepTogether, FrameBreak)

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from common import RESULTS, FIGS, load_json  # noqa: E402

F = "C:/Windows/Fonts/"
for name, f in [("Ar", "arial"), ("Ar-B", "arialbd"), ("Ar-I", "ariali"), ("Ar-BI", "arialbi"),
                ("Ti", "times"), ("Ti-B", "timesbd"), ("Ti-I", "timesi"), ("Ti-BI", "timesbi")]:
    pdfmetrics.registerFont(TTFont(name, F + f + ".ttf"))
pdfmetrics.registerFontFamily("Ar", normal="Ar", bold="Ar-B", italic="Ar-I", boldItalic="Ar-BI")
pdfmetrics.registerFontFamily("Ti", normal="Ti", bold="Ti-B", italic="Ti-I", boldItalic="Ti-BI")

WINE = colors.HexColor("#7B1E3A")
eda, pre, clu = load_json("eda.json"), load_json("preprocess.json"), load_json("clustering.json")
ev, tun = load_json("evaluation.json"), load_json("tuning.json")
cv = pd.read_csv(RESULTS / "cv_results.csv")
te = pd.read_csv(RESULTS / "test_metrics.csv")
prof = pd.read_csv(RESULTS / "cluster_profiles.csv")

# ---------------------------------------------------------------- estilos
H1 = ParagraphStyle("H1", fontName="Ar-B", fontSize=15, textColor=WINE, spaceBefore=14, spaceAfter=8)
H2 = ParagraphStyle("H2", fontName="Ar-B", fontSize=11.5, textColor=colors.HexColor("#333333"), spaceBefore=9, spaceAfter=4)
B = ParagraphStyle("B", fontName="Ar", fontSize=9.8, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
BL = ParagraphStyle("BL", parent=B, leftIndent=14, bulletIndent=2, spaceAfter=2)
CAP = ParagraphStyle("CAP", fontName="Ar-I", fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=10)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=8, leading=10, backColor=colors.HexColor("#F3F3F3"), borderPadding=4, spaceAfter=8)
TITLE = ParagraphStyle("T", fontName="Ar-B", fontSize=22, leading=27, textColor=WINE, alignment=TA_CENTER, spaceAfter=10)
SUB = ParagraphStyle("S", fontName="Ar", fontSize=12, leading=16, alignment=TA_CENTER, spaceAfter=6)

# estilos IEEE
IT = ParagraphStyle("IT", fontName="Ti-B", fontSize=17, leading=20, alignment=TA_CENTER, spaceAfter=6)
IA = ParagraphStyle("IA", fontName="Ti", fontSize=10, leading=12, alignment=TA_CENTER, spaceAfter=8)
IB = ParagraphStyle("IB", fontName="Ti", fontSize=9, leading=10.6, alignment=TA_JUSTIFY, spaceAfter=3, firstLineIndent=10)
IABS = ParagraphStyle("IABS", parent=IB, fontName="Ti-B", firstLineIndent=0, spaceAfter=6)
IH = ParagraphStyle("IH", fontName="Ti", fontSize=9, leading=11, alignment=TA_CENTER, spaceBefore=7, spaceAfter=3)
IH2 = ParagraphStyle("IH2", fontName="Ti-I", fontSize=9, leading=11, spaceBefore=4, spaceAfter=2)
ICAP = ParagraphStyle("ICAP", fontName="Ti", fontSize=8, leading=9.5, alignment=TA_CENTER, spaceAfter=5)
IREF = ParagraphStyle("IREF", fontName="Ti", fontSize=8, leading=9.4, alignment=TA_JUSTIFY, leftIndent=16, firstLineIndent=-16, spaceAfter=2)


def P(t, s=B): return Paragraph(t, s)
def bullets(items): return [Paragraph(i, BL, bulletText="•") for i in items]


def fig(name, width_cm, caption, style=CAP, cell=False):
    from PIL import Image as PI
    w, h = PI.open(FIGS / f"{name}.png").size
    img = Image(str(FIGS / f"{name}.png"), width=width_cm * cm, height=width_cm * cm * h / w)
    parts = [img, Paragraph(caption, style)]
    return parts if cell else KeepTogether(parts)


def table(data, widths, font="Ar", size=8, head=WINE, zebra=True, align_left_cols=(0,)):
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), size),
          ("BACKGROUND", (0, 0), (-1, 0), head), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("FONTNAME", (0, 0), (-1, 0), font + "-B"), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BBBBBB")),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if zebra:
        st += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7EEF1")])]
    t.setStyle(TableStyle(st))
    return t


def n(x): return f"{x:,.0f}".replace(",", " ")


# ---------------------------------------------------------------- datos derivados
short = {"Baseline": "Baseline", "Regresión logística (tabular)": "Reg. logística (tabular)", "Árbol de decisión": "Árbol de decisión",
         "Random Forest": "Random Forest", "Gradient Boosting (ajustado)": "Gradient Boosting",
         "Gradient Boosting (sin catador)": "GB sin catador", "Regresión logística + TF-IDF (ajustada)": "Reg. logística + TF-IDF",
         "Stacking (texto + tabular)": "Stacking texto+tabular"}
te = te.copy(); te["m"] = te.modelo.map(short)
tdict = te.set_index("modelo")
LRT, GBM, STK = "Regresión logística + TF-IDF (ajustada)", "Gradient Boosting (ajustado)", "Stacking (texto + tabular)"
imp = ev["perm_importance"]

story = []
# =================================================================== PORTADA
story += [Spacer(1, 3.5 * cm), P("Guía de Práctica Experimental 1 – Minería de Datos", SUB),
          P("Aplicación de técnicas de data mining en un caso de estudio", SUB), Spacer(1, 1 * cm),
          P("Predicción de la calidad de vinos a partir de reseñas de catadores", TITLE),
          P("Dataset Wine Reviews (Kaggle / WineEnthusiast)", SUB), Spacer(1, 2 * cm),
          P("<b>Estudiante(s):</b> Edison Paul Llerena Cuzco", SUB), P("<b>Universidad:</b> Universidad Estatal Amazónica", SUB), P("<b>Correo:</b> ep.llerenac@uea.edu.ec", SUB), P("<b>Asignatura:</b> Minería de Datos", SUB),
          P("<b>Entrega:</b> Semana 16", SUB), Spacer(1, 1.5 * cm),
          P("<b>Repositorio:</b> https://github.com/PAUL593LLC/mineria-wine-reviews", SUB),
          P("<b>Cuaderno:</b> notebooks/Practica1_Wine_Reviews.ipynb (Google Colab)", SUB), PageBreak()]

# =================================================================== 1 PROBLEMA
story += [P("1. Definición del problema", H1),
          P("Una distribuidora de vinos maneja miles de referencias y necesita priorizar cuáles destacar o comprar. Las reseñas de catadores profesionales "
            "asignan un puntaje de 80 a 100, pero no existen para todos los vinos. Este estudio plantea la pregunta: <b>¿puede predecirse si un vino "
            "obtendrá 90 puntos o más (“alta calidad”) usando precio, origen, variedad, antigüedad, catador y el texto de la reseña?</b>"),
          P("1.1 Objetivos", H2),
          P("<b>General.</b> Construir y evaluar modelos de minería de datos descriptivos y predictivos sobre la calidad de los vinos."),
          *bullets(["Explorar y depurar el dataset (calidad de datos, distribuciones, relaciones).",
                    "Segmentar el catálogo con un modelo descriptivo (K-Means).",
                    "Comparar al menos cinco algoritmos de clasificación con validación cruzada y prueba final.",
                    "Identificar qué variables y qué palabras explican la calidad."]),
          P("1.2 Alcance", H2),
          P(f"Reseñas de WineEnthusiast hasta 2017 ({n(eda['shape'][0])} registros, {eda['n_countries']} países, {n(eda['n_varieties'])} variedades). "
            "Variable objetivo binaria: <i>high_quality</i> = 1 si <i>points</i> ≥ 90. El modelo es una herramienta de priorización y análisis, no sustituye al catador."),
          P("1.3 Metodología", H2),
          P("Se siguió el proceso de minería de datos en cinco fases (problema, datos, preprocesamiento, modelado, evaluación), con código reproducible en Python "
            "(pandas, scikit-learn, seaborn) y semilla fija (42).")]

# =================================================================== 2 DATOS
mp = eda["missing_pct"]
story += [P("2. Datos: recopilación y exploración", H1),
          P("El dataset <i>winemag-data-130k-v2</i> proviene de Kaggle (zynicide/wine-reviews). Cada fila es una reseña con: país, provincia, regiones, variedad, "
            "bodega, título, designación, precio, catador, <i>descripción</i> (texto libre) y <i>puntos</i>."),
          P("2.1 Calidad de los datos", H2),
          table([["Característica", "Valor"],
                 ["Filas × columnas", f"{n(eda['shape'][0])} × {eda['shape'][1]}"],
                 ["Reseñas duplicadas (descripción + título)", f"{n(eda['duplicates_desc_title'])} (7.7 %)"],
                 ["Faltantes: region_2 / taster_twitter / taster_name", f"{mp['region_2']:.1f} % / {mp['taster_twitter_handle']:.1f} % / {mp['taster_name']:.1f} %"],
                 ["Faltantes: designation / region_1 / price", f"{mp['designation']:.1f} % / {mp['region_1']:.1f} % / {mp['price']:.1f} %"],
                 ["Puntos: media ± desv. (mín–máx)", f"{eda['points']['mean']:.2f} ± {eda['points']['std']:.2f} (80–100)"],
                 ["Precio: mediana / media / máx (USD)", f"{eda['price']['50%']:.0f} / {eda['price']['mean']:.1f} / {n(eda['price']['max'])}"],
                 ["Proporción de reseñas con ≥ 90 puntos", f"{eda['share_high_quality_raw']:.1%}"]], [9.5 * cm, 6.5 * cm]),
          Spacer(1, 4),
          fig("01_missing", 10.5, "Figura 1. Porcentaje de valores faltantes por variable."),
          P("2.2 Análisis descriptivo y visualizaciones", H2),
          P(f"La distribución del puntaje es casi simétrica con centro en 87–88 y cola hacia 100; con el umbral de 90 la clase positiva es ≈ 38 %, de modo que el problema "
            f"está moderadamente desbalanceado (un clasificador trivial acierta 62 %). El precio presenta fuerte sesgo a la derecha (media {eda['price']['mean']:.1f} > "
            f"mediana {eda['price']['50%']:.0f}), por lo que se trabaja en escala logarítmica."),
          fig("02_points_price_dist", 15, "Figura 2. Distribución del puntaje (izquierda) y del precio en escala log (derecha)."),
          P(f"El precio y el puntaje se correlacionan de forma moderada-fuerte (Spearman ρ = {eda['spearman_price_points']['rho']:.2f}, p &lt; 0.001), con mucha dispersión: "
            "existen vinos de 15 USD con 92 puntos y vinos de 100 USD con 85."),
          fig("03_price_vs_points", 9.5, "Figura 3. Densidad de reseñas por precio (log) y puntaje."),
          P(f"Estados Unidos concentra el 42 % de las reseñas ({n(eda['top_countries']['US'])}), seguido de Francia e Italia. Las medianas difieren entre países "
            f"(Kruskal-Wallis H = {eda['kruskal_points_country']['H']:.0f}, p ≈ 0): Austria y Alemania con mediana 90; Argentina y Chile con 86. "
            "También hay diferencias entre los 19 catadores: la media va de 85.9 a 90.6 puntos "
            f"(H = {eda['kruskal_points_taster']['H']:.0f}, p ≈ 0), lo que advierte de un posible sesgo del evaluador."),
          fig("04_countries", 15, "Figura 4. Reseñas y puntaje por país (top 10)."),
          fig("05_varieties_tasters", 15, "Figura 5. Variedades más frecuentes y puntaje medio por catador.")]

# =================================================================== 3 PREPROCESAMIENTO
story += [P("3. Preprocesamiento", H1),
          P("Las transformaciones sin aprendizaje se aplican una vez sobre todo el dataset (script <i>02_preprocess.py</i>); las que aprenden parámetros de los datos "
            "(mediana, escalado, categorías, TF-IDF) se ajustan solo con el conjunto de entrenamiento dentro de un <i>Pipeline</i> de scikit-learn para evitar fuga de información."),
          table([["Paso", "Acción", "Efecto"],
                 ["Duplicados", "Eliminar reseñas repetidas (descripción + título)", f"{n(pre['rows_raw'])} → {n(pre['rows_final'])} filas"],
                 ["Columnas", "Descartar region_2 (61 % nulos) y taster_twitter_handle (redundante)", "−2 columnas"],
                 ["Faltantes categóricos", "country, province, taster_name → “Unknown”", "0 nulos"],
                 ["Precio", f"Recorte al percentil 99.9 ({pre['price_cap_p999']:.0f} USD), indicador de ausencia, log(1+x); mediana en el pipeline",
                  f"{pre['price_capped_rows']} filas recortadas"],
                 ["Año de cosecha", "Regex sobre el título → age = 2017 − año; indicador age_missing", f"{pre['year_extracted_pct']:.1f} % extraído"],
                 ["Texto / metadatos", "desc_len_chars, desc_words, has_designation, has_region", "+4 variables"],
                 ["Codificación", "One-hot con categorías raras agrupadas (&lt; 200 casos); TF-IDF 1–2 gramas (20 000 términos)", "dentro del pipeline"],
                 ["Objetivo", "high_quality = 1 si points ≥ 90", f"{pre['share_high_quality']:.1%} positivos"]],
                [3.2 * cm, 9.3 * cm, 3.6 * cm], size=7.8),
          Spacer(1, 6),
          fig("06_preprocess", 15, "Figura 6. Efecto del recorte/transformación del precio y distribución de la edad del vino."),
          P("Un hallazgo del feature engineering: los vinos de alta calidad tienen reseñas más largas (mediana de 45 palabras frente a 36), lo que luego resulta decisivo en el modelado.")]

# =================================================================== 4 MODELADO
story += [P("4. Modelado", H1), P("4.1 Modelo descriptivo: segmentación con K-Means", H2),
          P(f"Se agruparon los vinos según puntaje, log-precio, edad y longitud de reseña (estandarizados). La silueta es máxima con k = 2 ({clu['silhouette_by_k']['2']:.2f}) y baja con k = 4 "
            f"({clu['silhouette_final']:.2f}); se eligió <b>k = 4</b> por interpretabilidad, a sabiendas de que los grupos se solapan (el vino es un continuo de calidad)."),
          fig("07_kmeans_selection", 11, "Figura 7. Método del codo y coeficiente de silueta."), ]
names = {0: "Calidad-precio", 1: "Premium", 2: "Maduros", 3: "Económicos"}
rows = [["Segmento", "% vinos", "Puntos", "Precio med. (USD)", "Edad", "Palabras", "≥ 90 pts", "Variedad top"]]
for _, r in prof.iterrows():
    rows.append([names[int(r.cluster)], f"{r.share_pct:.1f}", f"{r.points:.1f}", f"{r.price_median:.0f}", f"{r.age:.1f}",
                 f"{r.desc_words:.0f}", f"{r.high_quality:.0%}", r.variety_top])
story += [table(rows, [2.7 * cm, 1.6 * cm, 1.5 * cm, 2.7 * cm, 1.3 * cm, 1.7 * cm, 1.6 * cm, 3.0 * cm], size=7.8), Spacer(1, 6),
          P("Los segmentos son interpretables: <b>Premium</b> (92 puntos, 60 USD, 93 % de alta calidad), <b>Calidad-precio</b> (89 puntos, 27 USD), "
            "<b>Maduros</b> (12 años de edad media, 87.6 puntos) y <b>Económicos</b> (85 puntos, 15 USD, reseñas cortas, ningún vino ≥ 90)."),
          fig("08_kmeans_pca", 9, "Figura 8. Clusters en el plano de los dos primeros componentes principales."),
          P("4.2 Modelos predictivos", H2),
          P("Se partió el dataset en 80 % entrenamiento (95 990) y 20 % prueba (23 998), estratificado. Se compararon seis enfoques bajo validación cruzada estratificada de 5 folds "
            "(AUC-ROC, F1, exactitud):"),
          *bullets(["<b>Baseline</b>: clase mayoritaria.", "<b>Regresión logística</b> sobre variables tabulares.", "<b>Árbol de decisión</b> (profundidad 8).",
                    "<b>Random Forest</b> (200 árboles).", "<b>Gradient Boosting</b> (HistGradientBoosting, categorías nativas).",
                    "<b>Regresión logística + TF-IDF</b> del texto de la reseña (unigramas y bigramas).",
                    "<b>Stacking</b> (texto + boosting, meta-clasificador logístico) y una ablación de boosting <b>sin catador</b>."]),
          P(f"Los hiperparámetros se ajustaron con GridSearchCV (3 folds, AUC): boosting con tasa de aprendizaje {tun['hgb']['params']['clf__learning_rate']}, "
            f"{tun['hgb']['params']['clf__max_leaf_nodes']} hojas y L2 = {tun['hgb']['params']['clf__l2_regularization']}; regresión con texto con C = {tun['lr_tfidf']['params']['clf__C']}.")]
rows = [["Modelo (CV 5 folds, entrenamiento)", "Accuracy", "F1", "AUC-ROC"]]
for _, r in cv.sort_values("roc_auc_mean", ascending=False).iterrows():
    if r.modelo == "Gradient Boosting (tabular)":
        continue
    rows.append([r.modelo, f"{r.accuracy_mean:.4f} ± {r.accuracy_std:.4f}", f"{r.f1_mean:.4f} ± {r.f1_std:.4f}", f"{r.roc_auc_mean:.4f} ± {r.roc_auc_std:.4f}"])
story += [Spacer(1, 4), table(rows, [7.4 * cm, 3.1 * cm, 3.1 * cm, 3.1 * cm], size=7.8), Spacer(1, 6),
          fig("09_cv_comparison", 12.5, "Figura 9. AUC-ROC en validación cruzada (media ± desviación).")]

# =================================================================== 5 EVALUACION
rows = [["Modelo (conjunto de prueba)", "Accuracy", "Precisión", "Recall", "F1", "MCC", "AUC-ROC", "PR-AUC"]]
for _, r in te.iterrows():
    rows.append([r.m, f"{r.accuracy:.3f}", f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.f1:.3f}", f"{r.mcc:.3f}", f"{r.roc_auc:.3f}", f"{r.pr_auc:.3f}"])
bp = ev["error_by_price_band"]
story += [P("5. Evaluación y validación de resultados", H1),
          P("5.1 Desempeño en el conjunto de prueba", H2),
          table(rows, [4.4 * cm, 1.7 * cm, 1.8 * cm, 1.6 * cm, 1.4 * cm, 1.5 * cm, 1.9 * cm, 1.7 * cm], size=7.6), Spacer(1, 6),
          P(f"El mejor modelo por AUC es el <i>stacking</i> ({tdict.loc[STK,'roc_auc']:.4f}), pero la regresión logística con TF-IDF ({tdict.loc[LRT,'roc_auc']:.4f}) es prácticamente "
            f"equivalente y mucho más simple, por lo que se adopta como modelo final: <b>exactitud {tdict.loc[LRT,'accuracy']:.1%}, F1 {tdict.loc[LRT,'f1']:.3f}, AUC {tdict.loc[LRT,'roc_auc']:.3f}</b> frente a "
            f"62 % del baseline. Los resultados de prueba coinciden con los de validación cruzada (sin sobreajuste). Mediante bootstrap (300 réplicas) el intervalo de confianza al 95 % del AUC del "
            f"modelo final es [{ev['auc_ci95'][0]:.3f}, {ev['auc_ci95'][1]:.3f}] y el de F1 [{ev['f1_ci95'][0]:.3f}, {ev['f1_ci95'][1]:.3f}]. "
            f"La ventaja del modelo con texto sobre el boosting tabular es significativa: la diferencia de AUC (boosting − texto) tiene IC95 % [{ev['auc_diff_gb_minus_text_ci95'][0]:.3f}, {ev['auc_diff_gb_minus_text_ci95'][1]:.3f}], que no incluye el cero."),
          fig("10_roc_pr", 15.5, "Figura 10. Curvas ROC y Precisión-Recall en el conjunto de prueba."),
          P("5.2 Matriz de confusión y calibración", H2),
          Table([[fig("11_confusion", 6.8, "Figura 11. Matriz de confusión (modelo final: regresión logística + TF-IDF).", cell=True), fig("12_calibration", 7.4, "Figura 12. Curvas de calibración.", cell=True)]],
                colWidths=[8 * cm, 8 * cm]),
          P("Hay más falsos negativos (1 773) que falsos positivos (1 458); casi todos los errores ocurren en vinos de 89 a 91 puntos, junto al umbral (72 % de los errores; tasa de error de 33 % en los de 89 y 43 % en los de 90). "
            f"Por rango de precio el error es de {bp['<15']:.1%} en vinos &lt; 15 USD, {bp['15-30']:.1%} en 15–30, {bp['30-60']:.1%} en 30–60, {bp['60-120']:.1%} en 60–120 y {bp['>120']:.1%} en &gt; 120 USD: "
            "la incertidumbre se concentra en la zona media de precios."),
          P("5.3 Interpretación: ¿qué explica la calidad?", H2),
          P(f"Con importancia por permutación sobre el boosting, la variable más influyente es la <b>longitud de la reseña</b> (caída de AUC de {imp['desc_len_chars']:.3f}), seguida del "
            f"<b>precio</b> ({imp['log_price']:.3f}); catador ({imp['taster_name']:.3f}), provincia ({imp['province']:.3f}), edad ({imp['age']:.3f}) y variedad ({imp['variety']:.3f}) pesan mucho menos. "
            f"Un boosting entrenado <b>sin</b> el catador pierde apenas {tdict.loc[GBM,'roc_auc']-tdict.loc['Gradient Boosting (sin catador)','roc_auc']:.3f} de AUC, así que el modelo no depende del sesgo del evaluador."),
          fig("13_permutation_importance", 9.5, "Figura 13. Importancia por permutación (caída del AUC)."),
          P("En el texto, los términos que más suman a la alta calidad son <i>complex, beautiful, delicious, long, impressive, elegant, rich</i>; los que más restan, "
            "<i>simple, lacks, straightforward, somewhat, bit, little, easy</i>. Se verificó que solo el 3.3 % de las descripciones contienen números entre 80 y 100 "
            "(casi siempre porcentajes de mezcla de uvas) y apenas el 2 % de ellos coincide con el puntaje, por lo que no hay fuga directa de la etiqueta."),
          fig("14_text_terms", 15, "Figura 14. Términos con mayor coeficiente en la regresión logística."),
          P("5.4 Conclusiones y limitaciones", H2),
          *bullets([f"Es posible predecir la alta calidad con buena precisión (AUC ≈ {tdict.loc[LRT,'roc_auc']:.2f}).",
                    "El texto de la reseña aporta más que cualquier algoritmo más sofisticado (≈ +4 puntos de AUC frente a +3 por pasar de árbol a boosting).",
                    "La longitud de la reseña y el precio son las variables tabulares dominantes; el catador influye poco.",
                    "K-Means entrega cuatro perfiles de negocio interpretables, con fronteras difusas (silueta 0.195).",
                    "Limitaciones: datos de una sola revista hasta 2017; puntaje subjetivo; la reseña se escribe después de catar el vino, "
                    "por lo que el modelo sirve para priorizar y analizar, no para estimar la calidad de un vino sin reseña; el umbral de 90 es una decisión de negocio.",
                    "Trabajo futuro: embeddings de lenguaje, regresión del puntaje exacto y validación temporal con reseñas posteriores."]),
          P("6. Repositorio y reproducibilidad", H1),
          P("El código completo (scripts por fase, cuaderno de Colab, figuras y resultados) está en el repositorio de GitHub indicado en la portada. Para reproducir: "),
          P("pip install -r requirements.txt<br/>python src/01_eda.py &nbsp;&amp;&amp;&nbsp; python src/02_preprocess.py &nbsp;&amp;&amp;&nbsp; python src/03_clustering.py<br/>"
            "python src/04_modeling.py &nbsp;&amp;&amp;&nbsp; python src/05_evaluation.py", CODE)]

# =================================================================== ARTÍCULO IEEE
story += [NextPageTemplate("ieee"), PageBreak()]
story += [P("Anexo – Artículo técnico (formato IEEE short paper)", ParagraphStyle("an", fontName="Ar-I", fontSize=8, textColor=colors.grey)), Spacer(1, 2),
          P("Predicting Wine Quality from Expert Reviews: A Data Mining Case Study on the Wine Reviews Dataset".replace("Predicting Wine Quality from Expert Reviews: A Data Mining Case Study on the Wine Reviews Dataset",
                                                                                                                           "Predicción de la calidad de vinos a partir de reseñas de expertos: un caso de estudio de minería de datos"), IT),
          P("Edison Paul Llerena Cuzco<br/><i>Universidad Estatal Amazónica, Asignatura de Minería de Datos</i><br/>ep.llerenac@uea.edu.ec", IA),
          FrameBreak()]
L = tdict.loc[LRT]; G = tdict.loc[GBM]
story += [P("<b><i>Resumen</i>—Se estudia si la calidad de un vino (≥ 90 puntos) puede predecirse a partir de metadatos y del texto de la reseña del catador, usando 119 988 reseñas "
            "únicas de WineEnthusiast. Se compararon seis clasificadores con validación cruzada y un conjunto de prueba independiente, y se segmentó el catálogo con K-Means. "
            f"Una regresión logística sobre TF-IDF alcanzó AUC = {L.roc_auc:.3f} y F1 = {L.f1:.3f}, frente a AUC = {G.roc_auc:.3f} del mejor modelo basado solo en variables tabulares. "
            "La longitud de la reseña y el precio resultaron las variables tabulares más influyentes.</b>", IABS),
          P("<b><i>Términos clave</i>—minería de datos, clasificación, TF-IDF, regresión logística, gradient boosting, K-Means, vinos.</b>", IABS),

          P("I. INTRODUCCIÓN", IH),
          P("Los puntajes de catadores profesionales influyen en el precio y la demanda del vino, pero no están disponibles para todos los productos. Predecir si un vino superará un umbral de calidad "
            "permite priorizar catas y compras. Trabajos previos sobre calidad de vinos usan propiedades fisicoquímicas [1]; aquí se emplean en cambio metadatos comerciales y el texto de la reseña [2]. "
            "El objetivo es cuantificar cuánta información aporta cada fuente y qué algoritmo la explota mejor."),
          P("II. DATOS Y PREPROCESAMIENTO", IH),
          P(f"Se utilizó el conjunto <i>Wine Reviews</i> de Kaggle ({n(pre['rows_raw'])} reseñas, 13 variables). Se eliminaron {n(pre['duplicates_removed'])} duplicados y las columnas "
            "<i>region_2</i> (61 % nulos) y <i>taster_twitter_handle</i>. El precio se recortó al percentil 99.9, se imputó con la mediana y se transformó con logaritmo; la cosecha se extrajo del título "
            "mediante expresiones regulares (96.4 %); se derivaron la longitud de la reseña y dos indicadores binarios. La variable objetivo es <i>points</i> ≥ 90 (38.0 % de positivos). "
            "La imputación, el escalado, la codificación y el TF-IDF (unigramas y bigramas, 20 000 términos) se ajustaron solo con el conjunto de entrenamiento mediante <i>pipelines</i>."),
          P("III. METODOLOGÍA", IH),
          P("Se dividió el conjunto en 80 % entrenamiento y 20 % prueba (estratificado, semilla 42). Los modelos comparados fueron: línea base (clase mayoritaria), regresión logística, árbol de decisión, "
            "<i>Random Forest</i>, <i>gradient boosting</i> con histogramas y regresión logística con TF-IDF, además de un ensamble <i>stacking</i>. Los hiperparámetros se ajustaron con búsqueda en rejilla "
            "(3 folds, AUC) y la comparación se hizo con validación cruzada de 5 folds. Como modelo descriptivo se aplicó K-Means (k = 4) sobre puntaje, log-precio, edad y longitud de la reseña. "
            "Se reportan exactitud, precisión, <i>recall</i>, F1, MCC, AUC-ROC y PR-AUC, con intervalos bootstrap e importancia por permutación."),
          P("IV. RESULTADOS", IH), P("A. Comparación de modelos", IH2),
          P("La Tabla I resume el conjunto de prueba. Los resultados de validación cruzada son consistentes (desviación típica ≤ 0.003 en AUC), lo que descarta sobreajuste.")]
rows = [["Modelo", "Acc.", "F1", "AUC"]]
for key in ["Baseline", "Árbol de decisión", "Regresión logística (tabular)", "Random Forest", GBM, LRT, STK]:
    r = tdict.loc[key]; rows.append([short[key], f"{r.accuracy:.3f}", f"{r.f1:.3f}", f"{r.roc_auc:.3f}"])
story += [P("TABLA I. DESEMPEÑO EN EL CONJUNTO DE PRUEBA", ICAP), table(rows, [4.2 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm], font="Ti", size=7.5), Spacer(1, 4),
          P(f"El texto es el factor decisivo: añadirlo eleva el AUC de {G.roc_auc:.3f} a {L.roc_auc:.3f} (diferencia con IC95 % [{abs(ev['auc_diff_gb_minus_text_ci95'][1]):.3f}, {abs(ev['auc_diff_gb_minus_text_ci95'][0]):.3f}]), "
            f"más que pasar de un árbol a <i>boosting</i> (+{G.roc_auc - tdict.loc['Árbol de decisión'].roc_auc:.3f}). El <i>stacking</i> mejora solo 0.0002 en AUC sobre la regresión con texto, sin justificar su costo."),
          fig("10_roc_pr", 8.6, "Fig. 1. Curvas ROC y Precisión-Recall (prueba).", ICAP),
          P("B. Variables influyentes", IH2),
          P(f"La importancia por permutación (Fig. 2) sitúa en primer lugar la longitud de la reseña (ΔAUC = {imp['desc_len_chars']:.3f}) y luego el precio ({imp['log_price']:.3f}). El catador pesa poco "
            f"({imp['taster_name']:.3f}); sin él, el AUC del <i>boosting</i> baja solo {G.roc_auc - tdict.loc['Gradient Boosting (sin catador)'].roc_auc:.3f}. En el texto, términos como <i>complex</i>, <i>beautiful</i>, "
            "<i>elegant</i> y <i>rich</i> elevan la probabilidad de alta calidad, mientras que <i>simple</i>, <i>lacks</i> y <i>straightforward</i> la reducen."),
          fig("13_permutation_importance", 7.6, "Fig. 2. Importancia por permutación (ΔAUC).", ICAP),
          P("C. Segmentación", IH2),
          P("K-Means identificó cuatro segmentos: <i>Premium</i> (20 %, 92.2 puntos, 60 USD), <i>Calidad-precio</i> (34 %, 89.3 puntos, 27 USD), <i>Maduros</i> (16 %, edad media 11.9 años) y "
            f"<i>Económicos</i> (30 %, 85.4 puntos, 15 USD). La silueta es moderada ({clu['silhouette_final']:.2f}), reflejo de que la calidad es un continuo."),
          P("D. Análisis de errores", IH2),
          P("El 72 % de los errores corresponde a vinos de 89–91 puntos, junto al umbral de decisión; el error es bajo en vinos muy baratos (4 %) o muy caros (7.5 %) y mayor entre 15 y 60 USD (16–18 %)."),
          P("V. DISCUSIÓN Y LIMITACIONES", IH),
          P("Las reseñas se redactan tras catar el vino, por lo que el modelo no estima la calidad <i>ex ante</i>, sino que permite priorizar y auditar puntajes. Los datos provienen de una sola revista hasta 2017 y el puntaje es subjetivo. "
            "Solo el 3.3 % de las descripciones contiene números entre 80 y 100 (casi siempre porcentajes de uva), lo que descarta fuga directa de la etiqueta. El umbral de 90 puntos es una decisión de negocio."),
          P("VI. CONCLUSIONES", IH),
          P(f"Con precio, origen y texto de la reseña es posible predecir la alta calidad de un vino con exactitud de {L.accuracy:.1%} y AUC de {L.roc_auc:.3f}. La información textual supera en aporte a la sofisticación del algoritmo. "
            "Como trabajo futuro se proponen embeddings de lenguaje, regresión del puntaje exacto y validación temporal."),
          P("REFERENCIAS", IH),
          P("[1] P. Cortez, A. Cerdeira, F. Almeida, T. Matos y J. Reis, “Modeling wine preferences by data mining from physicochemical properties,” <i>Decision Support Systems</i>, vol. 47, n.º 4, pp. 547–553, 2009.", IREF),
          P("[2] Z. Thoutt, “Wine Reviews,” Kaggle, 2017. [En línea]. Disponible: https://www.kaggle.com/datasets/zynicide/wine-reviews", IREF),
          P("[3] F. Pedregosa <i>et al.</i>, “Scikit-learn: Machine learning in Python,” <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, 2011.", IREF),
          P("[4] T. Hastie, R. Tibshirani y J. Friedman, <i>The Elements of Statistical Learning</i>, 2.ª ed. Springer, 2009.", IREF)]


# =================================================================== DOCUMENTO
def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Ar", 8); canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"Minería de Datos – Práctica 1 · Wine Reviews · pág. {doc.page}")
    canvas.restoreState()


def footer_ieee(canvas, doc):
    canvas.saveState(); canvas.setFont("Ti", 8); canvas.setFillColor(colors.grey)
    canvas.drawCentredString(letter[0] / 2, 1.0 * cm, f"Artículo IEEE short paper · pág. {doc.page}")
    canvas.restoreState()


out = Path(__file__).parent / "Informe_Practica1_Wine_Reviews.pdf"
doc = BaseDocTemplate(str(out), pagesize=A4, title="Práctica 1 – Minería de Datos: Wine Reviews", author="Edison Paul Llerena Cuzco",
                      leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
main = Frame(2.2 * cm, 2 * cm, A4[0] - 4.4 * cm, A4[1] - 4 * cm, id="main")
LW, LH = letter; m = 1.7 * cm; gap = 0.6 * cm; cw = (LW - 2 * m - gap) / 2
top_h = 4.2 * cm
frames_ieee = [Frame(m, LH - m - top_h, LW - 2 * m, top_h, id="title", leftPadding=0, rightPadding=0),
               Frame(m, 1.8 * cm, cw, LH - m - top_h - 1.8 * cm - 0.2 * cm, id="c1", leftPadding=0, rightPadding=0),
               Frame(m + cw + gap, 1.8 * cm, cw, LH - m - top_h - 1.8 * cm - 0.2 * cm, id="c2", leftPadding=0, rightPadding=0)]
frames_ieee2 = [Frame(m, 1.8 * cm, cw, LH - m - 1.8 * cm, id="d1", leftPadding=0, rightPadding=0),
                Frame(m + cw + gap, 1.8 * cm, cw, LH - m - 1.8 * cm, id="d2", leftPadding=0, rightPadding=0)]
doc.addPageTemplates([PageTemplate("main", frames=[main], onPage=footer, pagesize=A4),
                      PageTemplate("ieee", frames=frames_ieee, onPage=footer_ieee, pagesize=letter, autoNextPageTemplate="ieee2"),
                      PageTemplate("ieee2", frames=frames_ieee2, onPage=footer_ieee, pagesize=letter)])
doc.build(story)
print("PDF:", out)

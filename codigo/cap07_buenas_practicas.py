# -*- coding: utf-8 -*-
"""
Capítulo 7 · Buenas prácticas y flujo de trabajo científico

Script extraído del notebook cap07_buenas_practicas.ipynb
Libro: "Inteligencia artificial para estudiantes de ciencias"
Jorge Bravo Abad — Ediciones Pirámide, 2026
"""


# ======================================================================
# Capitulo 7 - Buenas practicas y flujo de trabajo cientifico
# ======================================================================

# Instalacion de dependencias (descomentar en Colab/Kaggle si es necesario)
# !pip install scikit-optimize shap

import sys, platform
import numpy as np
import pandas as pd
import sklearn

print("Python     :", sys.version.split()[0])
print("numpy      :", np.__version__)
print("pandas     :", pd.__version__)
print("scikit-learn:", sklearn.__version__)

import random
random.seed(42)
np.random.seed(42)   # reproducibilidad

# ======================================================================
# SECCION 1 - Construccion del conjunto de datos a partir de propiedades atomicas
# ======================================================================

# Propiedades atomicas de los elementos relevantes en catodos de litio
# (masa_atomica, electronegatividad_Pauling, radio_ionico_pm, oxidacion, ion_primaria_eV)
PROP_ATOMICAS = {
    "Li": (6.94,  0.98,  76,  1,  5.39),
    "Co": (58.93, 1.88,  65,  3,  7.88),
    "Ni": (58.69, 1.91,  69,  2,  7.64),
    "Mn": (54.94, 1.55,  67,  3,  7.43),
    "Fe": (55.85, 1.83,  78,  2,  7.90),
    "Al": (26.98, 1.61,  54,  3,  5.99),
    "Ti": (47.87, 1.54,  61,  4,  6.83),
    "V":  (50.94, 1.63,  64,  3,  6.75),
    "P":  (30.97, 2.19,  17,  5, 10.49),
    "O":  (16.00, 3.44, 140, -2, 13.62),
}
NOMBRES_PROP = ["masa", "electroneg", "radio", "oxidacion", "ion_primaria"]

# Familias de catodos reales con su composicion (subindices por formula unidad)
FAMILIAS = {
    "LiCoO2":     {"Li": 1, "Co": 1, "O": 2},
    "LiNiO2":     {"Li": 1, "Ni": 1, "O": 2},
    "LiMn2O4":    {"Li": 1, "Mn": 2, "O": 4},
    "LiFePO4":    {"Li": 1, "Fe": 1, "P": 1, "O": 4},
    "NMC":        {"Li": 1, "Ni": 0.6, "Mn": 0.2, "Co": 0.2, "O": 2},
    "LiNiCoAlO2": {"Li": 1, "Ni": 0.8, "Co": 0.15, "Al": 0.05, "O": 2},
    "LiTiO2":     {"Li": 1, "Ti": 1, "O": 2},
    "LiVO2":      {"Li": 1, "V": 1, "O": 2},
}
print("Elementos:", list(PROP_ATOMICAS))
print("Familias de catodo:", list(FAMILIAS))

# ======================================================================
# Ingenieria de caracteristicas
# ======================================================================

def caracteristicas_desde_composicion(comp, rng):
    """Transforma una composicion {elemento: subindice} en un vector de caracteristicas."""
    total = sum(comp.values())

    def media_ponderada(idx):
        return sum(comp[e] * PROP_ATOMICAS[e][idx] for e in comp) / total

    metales = [e for e in comp if e not in ("Li", "O", "P")]
    eneg_metal  = np.mean([PROP_ATOMICAS[e][1] for e in metales]) if metales else 0.0
    radio_metal = np.mean([PROP_ATOMICAS[e][2] for e in metales]) if metales else 0.0

    fila = {
        "masa_media":             media_ponderada(0),
        "electroneg_media":       media_ponderada(1),
        "radio_ionico_medio":     media_ponderada(2),
        "oxidacion_media":        media_ponderada(3),
        "ion_primaria_media":     media_ponderada(4),
        "electroneg_metal":       eneg_metal,
        "radio_metal":            radio_metal,
        "num_metales_transicion": float(len(metales)),
        "fraccion_litio":         comp.get("Li", 0) / total,
        "fraccion_oxigeno":       comp.get("O", 0) / total,
        "contiene_fosfato":       1.0 if "P" in comp else 0.0,
    }
    # Pequena perturbacion (variabilidad experimental controlada)
    for k in fila:
        fila[k] = fila[k] * (1 + 0.04 * rng.standard_normal())
    return fila


rng = np.random.default_rng(7)
N = 300
filas, voltajes = [], []
for _ in range(N):
    fam = rng.choice(list(FAMILIAS))
    comp = FAMILIAS[fam]
    f = caracteristicas_desde_composicion(comp, rng)
    # Voltaje medio (V vs Li/Li+) generado con una relacion fisicamente motivada:
    # crece con la electronegatividad del metal y con el efecto inductivo del fosfato
    V = (1.1 * f["electroneg_metal"]
         + 0.9 * f["contiene_fosfato"]
         + 0.18 * f["oxidacion_media"]
         + 0.004 * (70 - f["radio_metal"])
         + 1.5
         + 0.15 * rng.standard_normal())
    f["familia"] = fam
    filas.append(f)
    voltajes.append(V)

df = pd.DataFrame(filas)
df["voltaje"] = voltajes
# Etiqueta binaria: catodo de alto voltaje (relevante para alta densidad energetica)
UMBRAL_V = 3.5
df["alto_voltaje"] = (df["voltaje"] > UMBRAL_V).astype(int)

print("Dimensiones:", df.shape)
df.head()

# ======================================================================
# SECCION 2 - Exploracion de los datos
# ======================================================================

print("Balance de clases (alto_voltaje):")
print(df["alto_voltaje"].value_counts(), "\n")
print("Estadisticas del voltaje (V vs Li/Li+):")
print(df["voltaje"].describe(), "\n")
print("Valores ausentes por columna:")
print(df.isnull().sum().sum(), "valores ausentes en total")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].hist(df["voltaje"], bins=25, color="#1E5FA5", edgecolor="white")
ax[0].axvline(UMBRAL_V, color="#D76900", linestyle="--", linewidth=2,
              label=f"umbral = {UMBRAL_V} V")
ax[0].set_xlabel("Voltaje medio (V vs Li/Li+)")
ax[0].set_ylabel("Frecuencia")
ax[0].set_title("Distribucion del voltaje")
ax[0].legend()

conteo = df["alto_voltaje"].value_counts().sort_index()
ax[1].bar(["bajo (0)", "alto (1)"], conteo.values,
          color=["#7A828C", "#287832"], edgecolor="white")
ax[1].set_ylabel("Numero de materiales")
ax[1].set_title("Balance de clases")
plt.tight_layout()
plt.show()

# Matriz de correlacion de las caracteristicas numericas
caracteristicas = [c for c in df.columns
                   if c not in ("familia", "voltaje", "alto_voltaje")]
corr = df[caracteristicas].corr()

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(caracteristicas)))
ax.set_yticks(range(len(caracteristicas)))
ax.set_xticklabels(caracteristicas, rotation=90, fontsize=8)
ax.set_yticklabels(caracteristicas, fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlacion")
ax.set_title("Correlacion entre caracteristicas")
plt.tight_layout()
plt.show()

# ======================================================================
# SECCION 3 - Separacion train/test sin leakage
# ======================================================================

from sklearn.model_selection import train_test_split

X = df[caracteristicas].values
y = df["alto_voltaje"].values
nombres_variables = caracteristicas

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("Entrenamiento:", X_train.shape, " Test:", X_test.shape)
print("Balance train:", np.bincount(y_train), " Balance test:", np.bincount(y_test))

# ======================================================================
# SECCION 4 - Modelo de referencia (baseline)
# ======================================================================

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline trivial: clase mayoritaria
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
proba_dummy = np.full(len(y_test), y_train.mean())
print(f"Baseline (clase mayoritaria) AUC test: "
      f"{roc_auc_score(y_test, proba_dummy):.3f}")

# Referencia razonable: regresion logistica en Pipeline
pipe_lr = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=2000)),
])
auc_lr = cross_val_score(pipe_lr, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"Regresion logistica  AUC CV: {auc_lr.mean():.3f} +- {auc_lr.std():.3f}")

# ======================================================================
# SECCION 5 - Optimizacion de hiperparametros
# ======================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, loguniform
import time

# --- Busqueda en rejilla ---
param_grid = {
    "n_estimators":     [100, 200, 400],
    "max_depth":        [None, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
t0 = time.time()
gs_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=cv, scoring="roc_auc", n_jobs=-1,
)
gs_grid.fit(X_train, y_train)
t_grid = time.time() - t0
print(f"Rejilla    -> AUC CV: {gs_grid.best_score_:.3f}  "
      f"({len(gs_grid.cv_results_['params'])} combinaciones, {t_grid:.1f} s)")
print("  mejores params:", gs_grid.best_params_)

# --- Busqueda aleatoria (mismo presupuesto, espacio continuo) ---
param_dist = {
    "n_estimators":      randint(100, 400),
    "max_depth":         [None, 5, 10, 20],
    "min_samples_leaf":  randint(1, 8),
    "max_features":      loguniform(0.2, 1.0),
}
t0 = time.time()
gs_rand = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist, n_iter=27, cv=cv, scoring="roc_auc",
    random_state=42, n_jobs=-1,
)
gs_rand.fit(X_train, y_train)
t_rand = time.time() - t0
print(f"Aleatoria  -> AUC CV: {gs_rand.best_score_:.3f}  "
      f"(27 evaluaciones, {t_rand:.1f} s)")
print("  mejores params:", gs_rand.best_params_)

# ======================================================================
# Optimizacion bayesiana con scikit-optimize
# ======================================================================

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

espacio_bayes = {
    "n_estimators":     Integer(100, 400),
    "max_depth":        Integer(2, 20),
    "min_samples_leaf": Integer(1, 8),
    "max_features":     Real(0.2, 1.0, prior="uniform"),
}
t0 = time.time()
bayes_search = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    espacio_bayes,
    n_iter=27, cv=cv, scoring="roc_auc",
    random_state=42, n_jobs=-1,
)
bayes_search.fit(X_train, y_train)
t_bayes = time.time() - t0
print(f"Bayesiana  -> AUC CV: {bayes_search.best_score_:.3f}  "
      f"(27 evaluaciones, {t_bayes:.1f} s)")
print("  mejores params:", dict(bayes_search.best_params_))

# Resumen comparativo de las tres estrategias
resumen = pd.DataFrame({
    "estrategia": ["Rejilla", "Aleatoria", "Bayesiana"],
    "AUC_CV":     [gs_grid.best_score_, gs_rand.best_score_, bayes_search.best_score_],
    "evaluaciones": [len(gs_grid.cv_results_["params"]), 27, 27],
    "tiempo_s":   [t_grid, t_rand, t_bayes],
})
print(resumen.to_string(index=False))

# Modelo final: el mejor de la busqueda bayesiana, reentrenado en todo el train
rf_final = bayes_search.best_estimator_
rf_final.fit(X_train, y_train)

# ======================================================================
# SECCION 6 - Evaluacion final (test, una sola vez)
# ======================================================================

from sklearn.metrics import roc_auc_score, roc_curve

proba_test = rf_final.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, proba_test)

# Intervalo de confianza por bootstrap sobre el test
rng_bs = np.random.default_rng(42)
boot = []
for _ in range(2000):
    idx = rng_bs.integers(0, len(y_test), len(y_test))
    if len(np.unique(y_test[idx])) < 2:
        continue
    boot.append(roc_auc_score(y_test[idx], proba_test[idx]))
ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
print(f"AUC test: {auc_test:.3f}  IC95% bootstrap: [{ci_low:.3f}, {ci_high:.3f}]")

fpr, tpr, _ = roc_curve(y_test, proba_test)
plt.figure(figsize=(4.5, 4.5))
plt.plot(fpr, tpr, color="#1E5FA5", linewidth=2,
         label=f"RF (AUC = {auc_test:.3f})")
plt.plot([0, 1], [0, 1], color="#7A828C", linestyle="--", linewidth=1)
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curva ROC en el conjunto de test")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ======================================================================
# SECCION 7 - Interpretabilidad
# ======================================================================

# ======================================================================
# 7.1 - Importancia por permutacion
# ======================================================================

from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf_final, X_test, y_test,
    n_repeats=20, scoring="roc_auc",
    random_state=42, n_jobs=-1,
)
imp_perm = pd.DataFrame({
    "variable":    nombres_variables,
    "importancia": result.importances_mean,
    "std":         result.importances_std,
}).sort_values("importancia", ascending=False)
print(imp_perm.to_string(index=False))

orden = imp_perm.iloc[::-1]
plt.figure(figsize=(7, 4.5))
plt.barh(orden["variable"], orden["importancia"],
         xerr=orden["std"], color="#287832", edgecolor="white")
plt.xlabel("Caida de AUC al permutar (importancia)")
plt.title("Importancia por permutacion (test)")
plt.tight_layout()
plt.show()

# ======================================================================
# 7.2 - Analisis SHAP completo
# ======================================================================

import shap

explainer = shap.TreeExplainer(rf_final)
shap_raw = explainer.shap_values(X_test)

# Robusto frente a ambas convenciones de la API de shap:
#  - versiones antiguas: lista [clase_0, clase_1]
#  - versiones recientes: array (n_muestras, n_variables, n_clases)
if isinstance(shap_raw, list):
    sv = shap_raw[1]
    base_value = explainer.expected_value[1]
else:
    sv = shap_raw[:, :, 1]
    ev = explainer.expected_value
    base_value = ev[1] if np.ndim(ev) > 0 else ev

print("Forma de los valores SHAP (clase positiva):", sv.shape)
print("Valor base (prediccion media):", round(float(base_value), 3))

# --- Grafico de resumen SHAP (beeswarm): vision global ---
shap.summary_plot(sv, X_test, feature_names=nombres_variables, show=True)

# --- Grafico de cascada SHAP: explicacion de una prediccion individual ---
# Elegimos el material con la prediccion mas alta de la clase positiva
idx = int(np.argmax(proba_test))
print(f"Explicando el material idx={idx}  "
      f"(probabilidad predicha de alto voltaje = {proba_test[idx]:.3f})")

explicacion = shap.Explanation(
    values=sv[idx],
    base_values=float(base_value),
    data=X_test[idx],
    feature_names=nombres_variables,
)
shap.plots.waterfall(explicacion, show=True)

# --- Ranking de importancias globales por valor SHAP absoluto medio ---
imp_shap = pd.DataFrame({
    "variable":         nombres_variables,
    "importancia_shap": np.abs(sv).mean(axis=0),
}).sort_values("importancia_shap", ascending=False)
print(imp_shap.to_string(index=False))

# ======================================================================
# 7.3 - Coherencia entre metodos y lectura cientifica
# ======================================================================

comparacion = (imp_perm[["variable", "importancia"]]
               .rename(columns={"importancia": "perm"})
               .merge(imp_shap.rename(columns={"importancia_shap": "shap"}),
                      on="variable"))
comparacion["rank_perm"] = comparacion["perm"].rank(ascending=False).astype(int)
comparacion["rank_shap"] = comparacion["shap"].rank(ascending=False).astype(int)
print(comparacion.sort_values("rank_shap").to_string(index=False))

# ======================================================================
# Resumen del proyecto
# ======================================================================

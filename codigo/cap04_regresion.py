# -*- coding: utf-8 -*-
"""
Capítulo 4 · Regresión

Script extraído del notebook cap04_regresion.ipynb
Libro: "Inteligencia artificial para estudiantes de ciencias"
Jorge Bravo Abad — Ediciones Pirámide, 2026
"""


# ======================================================================
# Capitulo 4 — Regresion
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, ConstantKernel as C
)

rng = np.random.default_rng(42)
plt.rcParams["figure.figsize"] = (6, 4)

# ======================================================================
# Datos de propiedades moleculares
# ======================================================================

n = 600
nombres_variables = [
    "peso_molecular", "area_polar", "n_donantes", "n_aceptores",
    "logP", "n_enlaces_rotables", "n_anillos", "fraccion_sp3"
]

PM   = rng.uniform(100, 500, n)
APSA = rng.uniform(10, 150, n)
ND   = rng.integers(0, 6, n).astype(float)
NA   = rng.integers(0, 10, n).astype(float)
logP = rng.normal(2.0, 1.5, n)
NER  = rng.integers(0, 12, n).astype(float)
NAN  = rng.integers(0, 5, n).astype(float)
FSP3 = rng.uniform(0, 1, n)

X = np.column_stack([PM, APSA, ND, NA, logP, NER, NAN, FSP3])

# Propiedad objetivo con dependencia no lineal + ruido
y = (
    0.012 * PM
    - 0.03 * APSA
    + 0.4 * logP**2
    - 0.5 * np.sqrt(APSA)
    + 0.8 * ND
    - 0.2 * NA
    + 1.5 * np.sin(0.02 * PM)
    + rng.normal(0, 0.5, n)
)
# Centrar el objetivo para facilitar el ajuste de los modelos
y = (y - y.mean()) / y.std()

df = pd.DataFrame(X, columns=nombres_variables)
df["objetivo"] = y
df.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("Entrenamiento:", X_train.shape, " Test:", X_test.shape)

# ======================================================================
# 1. Regresion lineal y polinomial
# ======================================================================

for grado in [1, 2, 3, 5]:
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=grado, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ])
    rmse = -cross_val_score(pipe, X_train, y_train, cv=5,
                            scoring="neg_root_mean_squared_error")
    print(f"Grado {grado}: RMSE = {rmse.mean():.4f} +- {rmse.std():.4f}")

# ======================================================================
# 2. Maquinas de vectores soporte para regresion (SVR)
# ======================================================================

pipe_svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf"))
])

param_grid = {
    "svr__C":       [1, 10, 100, 1000],
    "svr__epsilon": [0.01, 0.05, 0.1],
    "svr__gamma":   ["scale", 0.01, 0.05, 0.1],
}
gs = GridSearchCV(pipe_svr, param_grid, cv=5,
                  scoring="neg_root_mean_squared_error",
                  n_jobs=-1)
gs.fit(X_train, y_train)

from sklearn.metrics import r2_score
y_pred_svr = gs.predict(X_test)

print("Mejores hiperparametros:", gs.best_params_)
print("RMSE validacion:", -gs.best_score_)
print("R2 test:", r2_score(y_test, y_pred_svr))

# ======================================================================
# 3. Bosques aleatorios (Random Forest)
# ======================================================================

rf = RandomForestRegressor(
    n_estimators=200,
    max_features="sqrt",      # seleccion aleatoria de variables
    min_samples_leaf=2,       # evita hojas con un solo punto
    oob_score=True,           # error out-of-bag (gratis!)
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

print(f"R2 OOB (estimacion gratuita): {rf.oob_score_:.4f}")
print(f"R2 test (evaluacion honesta): {rf.score(X_test, y_test):.4f}")

importancias = pd.Series(rf.feature_importances_,
                         index=nombres_variables).sort_values(ascending=False)
print("\nTop variables mas importantes:")
print(importancias.head(10))

fig, ax = plt.subplots()
importancias.sort_values().plot.barh(ax=ax)
ax.set_xlabel("Importancia")
ax.set_title("Importancia de variables (Random Forest)")
plt.tight_layout()
plt.show()

# ======================================================================
# 4. Procesos gaussianos: energias de adsorcion en catalizadores
# ======================================================================

m = 80  # pocas observaciones, regimen tipico de DFT
centro_banda_d = rng.uniform(-3.0, 0.0, m)     # eV
num_coordinacion = rng.uniform(6, 12, m)

Xc = np.column_stack([centro_banda_d, num_coordinacion])

# Energia de adsorcion (eV) con relacion no lineal tipo volcan + ruido pequeno
E_ads = (
    -0.8 * (centro_banda_d + 1.5)**2
    + 0.15 * num_coordinacion
    - 1.2
    + rng.normal(0, 0.08, m)
)

Xc_train, Xc_test, Ec_train, Ec_test = train_test_split(
    Xc, E_ads, test_size=0.30, random_state=42
)
print("Train:", Xc_train.shape, " Test:", Xc_test.shape)

# ======================================================================
# Seleccion de nucleo
# ======================================================================

kernel = C(1.0, (1e-2, 1e2)) \
       * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) \
       + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,  # varias inits para evitar minimos locales
    normalize_y=True,
    random_state=42
)

pipe_gp = Pipeline([
    ("scaler", StandardScaler()),
    ("gp",     gp)
])
pipe_gp.fit(Xc_train, Ec_train)

# Prediccion con incertidumbre
y_pred, y_std = pipe_gp.predict(Xc_test, return_std=True)

print(f"Hiperparametros optimizados: {pipe_gp.named_steps['gp'].kernel_}")
print(f"R2 test: {pipe_gp.score(Xc_test, Ec_test):.4f}")
print(f"Incertidumbre media: {y_std.mean():.4f}")

# ======================================================================
# Cuantificacion de incertidumbre y calibracion
# ======================================================================

cobertura = np.mean(np.abs(Ec_test - y_pred) < 1.96 * y_std)
print(f"Cobertura IC 95%: {cobertura:.3f}  (ideal: 0.950)")

orden = np.argsort(y_pred)
fig, ax = plt.subplots()
idx = np.arange(len(y_pred))
ax.errorbar(idx, y_pred[orden], yerr=1.96 * y_std[orden],
            fmt="o", capsize=3, label="Prediccion +- 1.96 sigma")
ax.plot(idx, Ec_test[orden], "x", color="black", label="Valor real (DFT)")
ax.set_xlabel("Catalizador de test (ordenado)")
ax.set_ylabel("Energia de adsorcion (eV)")
ax.set_title("Proceso gaussiano: prediccion con incertidumbre")
ax.legend()
plt.tight_layout()
plt.show()

# ======================================================================
# Resumen
# ======================================================================

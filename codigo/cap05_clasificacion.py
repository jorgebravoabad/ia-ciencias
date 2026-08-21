# -*- coding: utf-8 -*-
"""
Capítulo 5 · Clasificación

Script extraído del notebook cap05_clasificacion.ipynb
Libro: "Inteligencia artificial para estudiantes de ciencias"
Jorge Bravo Abad — Ediciones Pirámide, 2026
"""


# ======================================================================
# Capítulo 5 — Clasificación
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, RocCurveDisplay,
                             accuracy_score, classification_report)

rng_global = np.random.default_rng(0)
plt.rcParams["figure.dpi"] = 110

# ======================================================================
# 1. Conjunto de datos: transición de fase del modelo de Ising
# ======================================================================

def make_phase_transition_toy(n=2500, Tc=2.269, T_min=1.0, T_max=4.0,
                              noise=0.18, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.uniform(T_min, T_max, size=n)
    y = (T > Tc).astype(int)                       # 0 ordenado, 1 desordenado

    beta = 0.125
    m_clean = np.zeros(n)
    mask = T < Tc
    m_clean[mask] = (1.0 - (T[mask] / Tc)) ** beta
    e_clean = -np.tanh(1.2 / T)
    width = 0.18
    chi_clean = 0.2 + 1.0 / (1.0 + ((T - Tc) / width) ** 2)

    m = np.clip(m_clean + rng.normal(0, noise, n), 0, 1)
    e = e_clean + rng.normal(0, noise, n)
    chi = np.clip(chi_clean + rng.normal(0, 0.25 * noise, n), 0, None)

    X = np.column_stack([m, e, chi])
    return X, y, T

X_ising, y_ising, T_ising = make_phase_transition_toy(seed=42)
print("X:", X_ising.shape, " clases:", np.bincount(y_ising))

fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
names = ["m (magnetizacion)", "e (energia)", "chi (susceptibilidad)"]
for j in range(3):
    ax[j].scatter(T_ising, X_ising[:, j], s=4, alpha=0.4)
    ax[j].axvline(2.269, color="k", ls="--", lw=1)
    ax[j].set_xlabel("T"); ax[j].set_title(names[j])
fig.tight_layout(); plt.show()

# ======================================================================
# 2. Comparación de los cuatro clasificadores sobre Ising
# ======================================================================

Xtr, Xte, ytr, yte = train_test_split(
    X_ising, y_ising, test_size=0.3, random_state=0, stratify=y_ising)

modelos = {
    "Regresion logistica": Pipeline([("sc", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=1000))]),
    "SVM (RBF)":           Pipeline([("sc", StandardScaler()),
                                     ("clf", SVC(kernel="rbf", probability=True,
                                                 random_state=0))]),
    "k-NN (k=15)":         Pipeline([("sc", StandardScaler()),
                                     ("clf", KNeighborsClassifier(n_neighbors=15))]),
    "Bosque aleatorio":    RandomForestClassifier(n_estimators=300, random_state=0),
}

resultados = {}
for nombre, modelo in modelos.items():
    modelo.fit(Xtr, ytr)
    proba = modelo.predict_proba(Xte)[:, 1]
    pred = modelo.predict(Xte)
    resultados[nombre] = dict(acc=accuracy_score(yte, pred),
                              proba=proba, pred=pred)
    print(f"{nombre:22s}  exactitud = {resultados[nombre]['acc']:.3f}")

# ======================================================================
# 2.1 Matrices de confusión
# ======================================================================

fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for axi, (nombre, r) in zip(axes, resultados.items()):
    cm = confusion_matrix(yte, r["pred"])
    ConfusionMatrixDisplay(cm, display_labels=["ord.", "desord."]).plot(
        ax=axi, colorbar=False, cmap="Blues")
    axi.set_title(nombre, fontsize=9)
fig.tight_layout(); plt.show()

# ======================================================================
# 2.2 Curvas ROC y área bajo la curva (AUC)
# ======================================================================

plt.figure(figsize=(5.2, 4.4))
estilos = ["-", "--", "-.", ":"]
for (nombre, r), ls in zip(resultados.items(), estilos):
    fpr, tpr, _ = roc_curve(yte, r["proba"])
    a = auc(fpr, tpr)
    plt.plot(fpr, tpr, ls=ls, lw=2, label=f"{nombre} (AUC={a:.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
plt.xlabel("Tasa de falsos positivos"); plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curvas ROC - Ising"); plt.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()

# ======================================================================
# 2.3 Fronteras de decisión
# ======================================================================

X2 = X_ising[:, [0, 2]]                 # m y chi
X2tr, X2te, y2tr, y2te = train_test_split(
    X2, y_ising, test_size=0.3, random_state=0, stratify=y_ising)

modelos2 = {
    "Regresion logistica": Pipeline([("sc", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=1000))]),
    "SVM (RBF)":           Pipeline([("sc", StandardScaler()),
                                     ("clf", SVC(kernel="rbf", random_state=0))]),
    "k-NN (k=15)":         Pipeline([("sc", StandardScaler()),
                                     ("clf", KNeighborsClassifier(n_neighbors=15))]),
    "Bosque aleatorio":    RandomForestClassifier(n_estimators=300, random_state=0),
}

x_min, x_max = X2[:, 0].min() - .05, X2[:, 0].max() + .05
y_min, y_max = X2[:, 1].min() - .05, X2[:, 1].max() + .05
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
for axi, (nombre, modelo) in zip(axes, modelos2.items()):
    modelo.fit(X2tr, y2tr)
    Z = modelo.predict(grid).reshape(xx.shape)
    axi.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    axi.scatter(X2te[:, 0], X2te[:, 1], c=y2te, s=6, cmap="coolwarm",
                edgecolors="none", alpha=0.7)
    axi.set_title(f"{nombre}\nexactitud={modelo.score(X2te, y2te):.3f}",
                  fontsize=8)
    axi.set_xlabel("m"); axi.set_ylabel("chi")
fig.tight_layout(); plt.show()

# ======================================================================
# 3. Conjunto de datos: clasificación de espectros moleculares
# ======================================================================

def make_spectra(n_per_class=400, n_points=120, seed=0):
    rng = np.random.default_rng(seed)
    eje = np.linspace(0, 10, n_points)
    # posiciones de pico caracteristicas por familia
    familias = {
        0: [(2.0, 0.4), (5.0, 0.6)],
        1: [(3.5, 0.5), (7.5, 0.4)],
        2: [(1.5, 0.3), (4.0, 0.4), (8.5, 0.5)],
    }
    X, y = [], []
    for clase, picos in familias.items():
        for _ in range(n_per_class):
            espectro = np.zeros(n_points)
            for centro, ancho in picos:
                c = centro + rng.normal(0, 0.15)         # desplazamiento
                amp = 1.0 + rng.normal(0, 0.15)
                espectro += amp * np.exp(-0.5 * ((eje - c) / ancho) ** 2)
            espectro += rng.normal(0, 0.05, n_points)    # ruido
            X.append(espectro); y.append(clase)
    return np.array(X), np.array(y), eje

Xs, ys, eje = make_spectra(seed=1)
print("X:", Xs.shape, " clases:", np.bincount(ys))

plt.figure(figsize=(6, 3))
for clase in [0, 1, 2]:
    idx = np.where(ys == clase)[0][0]
    plt.plot(eje, Xs[idx], label=f"familia {clase}")
plt.xlabel("variable espectral"); plt.ylabel("intensidad")
plt.title("Ejemplo de espectro por familia"); plt.legend()
plt.tight_layout(); plt.show()

# ======================================================================
# 3.1 Comparación de clasificadores (multiclase) y matrices de confusión
# ======================================================================

Xstr, Xste, ystr, yste = train_test_split(
    Xs, ys, test_size=0.3, random_state=0, stratify=ys)

modelos_s = {
    "Regresion logistica": Pipeline([("sc", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=2000))]),
    "SVM (RBF)":           Pipeline([("sc", StandardScaler()),
                                     ("clf", SVC(kernel="rbf", probability=True,
                                                 random_state=0))]),
    "k-NN (k=7)":          Pipeline([("sc", StandardScaler()),
                                     ("clf", KNeighborsClassifier(n_neighbors=7))]),
    "Bosque aleatorio":    RandomForestClassifier(n_estimators=300, random_state=0),
}

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
for axi, (nombre, modelo) in zip(axes, modelos_s.items()):
    modelo.fit(Xstr, ystr)
    pred = modelo.predict(Xste)
    cm = confusion_matrix(yste, pred)
    ConfusionMatrixDisplay(cm).plot(ax=axi, colorbar=False, cmap="Greens")
    axi.set_title(f"{nombre}\nexactitud={accuracy_score(yste, pred):.3f}",
                  fontsize=8)
fig.tight_layout(); plt.show()

# ======================================================================
# 3.2 Curvas ROC multiclase (una frente al resto)
# ======================================================================

from sklearn.preprocessing import label_binarize
clf = modelos_s["Bosque aleatorio"]
proba = clf.predict_proba(Xste)
y_bin = label_binarize(yste, classes=[0, 1, 2])

plt.figure(figsize=(5.2, 4.4))
estilos = ["-", "--", "-."]
for k, ls in zip(range(3), estilos):
    fpr, tpr, _ = roc_curve(y_bin[:, k], proba[:, k])
    plt.plot(fpr, tpr, ls=ls, lw=2,
             label=f"familia {k} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
plt.xlabel("Tasa de falsos positivos"); plt.ylabel("Tasa de verdaderos positivos")
plt.title("ROC multiclase (una frente al resto) - espectros")
plt.legend(fontsize=8, loc="lower right"); plt.tight_layout(); plt.show()

# ======================================================================
# 4. Resumen
# ======================================================================

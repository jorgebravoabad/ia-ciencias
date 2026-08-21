# -*- coding: utf-8 -*-
"""
Capítulo 6 · Aprendizaje no supervisado

Script extraído del notebook cap06_no_supervisado.ipynb
Libro: "Inteligencia artificial para estudiantes de ciencias"
Jorge Bravo Abad — Ediciones Pirámide, 2026
"""


# ======================================================================
# Capítulo 6 — Aprendizaje no supervisado
# ======================================================================

# ======================================================================
# Configuración
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt

# Reproducibilidad global
SEMILLA = 42

# Ancho de impresion del libro: 12,5 cm = 4,92 pulgadas
ANCHO_IMPRESION = 4.92

# Paleta del libro (RGB aproximada de IA_ciencias.sty)
AZUL    = "#1E5FA5"
VERDE   = "#2E9E5B"
NARANJA = "#E8722A"
ROJO    = "#C0392B"
GRIS    = "#7F8C8D"
COLORES = [AZUL, NARANJA, VERDE, ROJO]
MARCADORES = ["o", "s", "^", "D"]

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Coma decimal al estilo europeo en los ejes
from matplotlib.ticker import FuncFormatter
coma = FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ","))

# ======================================================================
# Parte 1 — Agrupamiento de espectros de difracción de rayos X
# ======================================================================

def generar_espectros_xrd(n_por_fase=80, n_canales=180, semilla=SEMILLA):
    """Genera espectros XRD sinteticos de tres fases cristalinas."""
    rng = np.random.default_rng(semilla)
    dos_theta = np.linspace(10, 80, n_canales)
    # Posiciones de picos caracteristicas de cada fase
    fases = {
        0: [20.5, 35.0, 57.0],
        1: [28.0, 47.5, 56.0, 66.0],
        2: [22.0, 31.0, 45.0, 62.0, 75.0],
    }
    espectros, fase_real = [], []
    for fase, picos in fases.items():
        for _ in range(n_por_fase):
            espectro = np.zeros(n_canales)
            for posicion in picos:
                amplitud = rng.uniform(0.6, 1.0)
                ancho    = rng.uniform(0.8, 1.6)
                centro   = posicion + rng.normal(0, 0.4)
                espectro += amplitud * np.exp(-0.5 * ((dos_theta - centro) / ancho) ** 2)
            espectro += rng.normal(0, 0.02, n_canales)  # ruido de fondo
            espectro = np.clip(espectro, 0, None)
            espectros.append(espectro)
            fase_real.append(fase)
    return dos_theta, np.array(espectros), np.array(fase_real)

dos_theta, X_xrd, fase_real = generar_espectros_xrd()
print(f"Matriz de espectros: {X_xrd.shape[0]} muestras x {X_xrd.shape[1]} canales")
print(f"Fases reales: {np.bincount(fase_real)}")

fig, ax = plt.subplots(figsize=(ANCHO_IMPRESION, 3.2))
estilos = ["-", "--", ":"]
for fase in range(3):
    idx = np.where(fase_real == fase)[0][0]
    ax.plot(dos_theta, X_xrd[idx] + fase * 1.2,   # desplazamiento vertical
            color=COLORES[fase], linestyle=estilos[fase], linewidth=1.3,
            label=f"Fase {fase}")
ax.set_xlabel(r"$2\theta$ (grados)")
ax.set_ylabel("Intensidad (u.a., desplazada)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()

# ======================================================================
# $k$-means con selección de $K$ por el coeficiente de silueta
# ======================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X_sc = StandardScaler().fit_transform(X_xrd)

resultados = {}
for K in range(2, 7):
    km = KMeans(n_clusters=K, init="k-means++", n_init=20, random_state=SEMILLA)
    etiquetas = km.fit_predict(X_sc)
    inercia = km.inertia_
    silueta = silhouette_score(X_sc, etiquetas)
    resultados[K] = {"inercia": inercia, "silueta": silueta}
    print(f"K={K:2d}: inercia={inercia:8.1f}, silueta={silueta:.4f}")

K_optimo = max(resultados, key=lambda k: resultados[k]["silueta"])
print(f"\nK optimo segun silueta: {K_optimo}")

Ks = list(resultados.keys())
inercias = [resultados[K]["inercia"] for K in Ks]
siluetas = [resultados[K]["silueta"] for K in Ks]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ANCHO_IMPRESION, 2.6))
ax1.plot(Ks, inercias, color=AZUL, linestyle="-", marker="o", markersize=4)
ax1.set_xlabel("Numero de clusteres $K$")
ax1.set_ylabel("Inercia")
ax1.set_title("Metodo del codo", fontsize=9)

ax2.plot(Ks, siluetas, color=NARANJA, linestyle="--", marker="s", markersize=4)
ax2.axvline(K_optimo, color=GRIS, linestyle=":", linewidth=1)
ax2.set_xlabel("Numero de clusteres $K$")
ax2.set_ylabel("Silueta media")
ax2.set_title("Coeficiente de silueta", fontsize=9)
fig.tight_layout()
plt.show()

from sklearn.metrics import adjusted_rand_score

km = KMeans(n_clusters=K_optimo, n_init=20, random_state=SEMILLA)
etiquetas_km = km.fit_predict(X_sc)
print(f"ARI (k-means vs fases reales): {adjusted_rand_score(fase_real, etiquetas_km):.3f}")

# ======================================================================
# DBSCAN con estimación de $\varepsilon$ por el gráfico de distancias $k$-NN
# ======================================================================

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

min_samples = 5
nn = NearestNeighbors(n_neighbors=min_samples).fit(X_sc)
distancias, _ = nn.kneighbors(X_sc)
distancias_k = np.sort(distancias[:, -1])

fig, ax = plt.subplots(figsize=(ANCHO_IMPRESION, 2.6))
ax.plot(distancias_k, color=AZUL, linewidth=1.3)
# El epsilon razonable esta en el codo; usamos el percentil 90 como heuristica
eps = np.percentile(distancias_k, 90)
ax.axhline(eps, color=ROJO, linestyle="--", linewidth=1,
           label=f"epsilon = {eps:.2f}".replace(".", ","))
ax.set_xlabel("Muestras ordenadas")
ax.set_ylabel(f"Distancia al vecino {min_samples}")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()

db = DBSCAN(eps=eps, min_samples=min_samples)
etiquetas_db = db.fit_predict(X_sc)

n_clusteres = len(set(etiquetas_db)) - (1 if -1 in etiquetas_db else 0)
n_ruido = (etiquetas_db == -1).sum()
print(f"Clusteres encontrados: {n_clusteres}")
print(f"Puntos de ruido: {n_ruido} ({100*n_ruido/len(etiquetas_db):.1f}%)")

# ======================================================================
# Parte 2 — Reducción de dimensionalidad de datos de expresión génica
# ======================================================================

def generar_expresion_genica(n_por_tipo=60, n_genes=400, n_tipos=4, semilla=SEMILLA):
    """Genera una matriz de expresion genica con n_tipos firmas distintas."""
    rng = np.random.default_rng(semilla)
    expresion = rng.normal(0, 1, (n_por_tipo * n_tipos, n_genes))
    tipo_real = np.repeat(np.arange(n_tipos), n_por_tipo)
    bloque = n_genes // (n_tipos + 1)
    for tipo in range(n_tipos):
        inicio = tipo * bloque
        genes_firma = slice(inicio, inicio + bloque)
        expresion[tipo_real == tipo, genes_firma] += rng.uniform(2.5, 4.0)
    return expresion, tipo_real

X_gen, tipo_real = generar_expresion_genica()
print(f"Matriz de expresion: {X_gen.shape[0]} muestras x {X_gen.shape[1]} genes")
print(f"Tipos celulares reales: {np.bincount(tipo_real)}")

# ======================================================================
# PCA: proyección a 2D y curva de varianza acumulada
# ======================================================================

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# --- Uso 1: proyectar a 2D para visualizar ---
pipe_vis = Pipeline([
    ("sc",  StandardScaler()),
    ("pca", PCA(n_components=2, random_state=SEMILLA)),
])
Z2 = pipe_vis.fit_transform(X_gen)

# --- Uso 2: elegir el numero de componentes que retiene el 95% ---
pipe_full = Pipeline([("sc", StandardScaler()), ("pca", PCA())])
pipe_full.fit(X_gen)
pca = pipe_full.named_steps["pca"]
var_acum = np.cumsum(pca.explained_variance_ratio_)
k_95 = int(np.argmax(var_acum >= 0.95) + 1)
print(f"Varianza explicada por PC1+PC2: {var_acum[1]*100:.1f}%")
print(f"Componentes para el 95% de varianza: {k_95} de {X_gen.shape[1]}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ANCHO_IMPRESION, 2.7))
for tipo in range(4):
    masc = tipo_real == tipo
    ax1.scatter(Z2[masc, 0], Z2[masc, 1], s=14, alpha=0.7,
                color=COLORES[tipo], marker=MARCADORES[tipo],
                label=f"Tipo {tipo}", edgecolor="white", linewidth=0.3)
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
ax1.set_title("Proyeccion PCA 2D", fontsize=9)
ax1.legend(frameon=False, fontsize=7, ncol=2)

ax2.plot(np.arange(1, len(var_acum)+1), var_acum, color=AZUL, linewidth=1.3)
ax2.axhline(0.95, color=ROJO, linestyle="--", linewidth=1, label="95 %")
ax2.axvline(k_95, color=GRIS, linestyle=":", linewidth=1)
ax2.set_xlabel("Numero de componentes")
ax2.set_ylabel("Varianza acumulada")
ax2.set_title("Curva de varianza", fontsize=9)
ax2.legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()

# ======================================================================
# $t$-SNE para visualización
# ======================================================================

from sklearn.manifold import TSNE

# Reducir primero con PCA a 30 dimensiones
X_pca = Pipeline([
    ("sc",  StandardScaler()),
    ("pca", PCA(n_components=30, random_state=SEMILLA)),
]).fit_transform(X_gen)

tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
            init="pca", random_state=SEMILLA)
Z_tsne = tsne.fit_transform(X_pca)
print(f"Proyeccion t-SNE: {Z_tsne.shape}")

fig, ax = plt.subplots(figsize=(ANCHO_IMPRESION, 3.2))
for tipo in range(4):
    masc = tipo_real == tipo
    ax.scatter(Z_tsne[masc, 0], Z_tsne[masc, 1], s=16, alpha=0.75,
               color=COLORES[tipo], marker=MARCADORES[tipo],
               label=f"Tipo {tipo}", edgecolor="white", linewidth=0.3)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
plt.show()

# ======================================================================
# Parte 3 — Detección de anomalías en una serie de sensores
# ======================================================================

def generar_serie_sensores(n=1000, frac_anomalias=0.04, semilla=SEMILLA):
    """Serie temporal de un sensor con anomalias puntuales inyectadas."""
    rng = np.random.default_rng(semilla)
    t = np.arange(n)
    senal = (10 + 2.0 * np.sin(2 * np.pi * t / 120)   # estacionalidad
             + 0.002 * t                              # deriva lenta
             + rng.normal(0, 0.3, n))                 # ruido
    es_anomalia = np.zeros(n, dtype=bool)
    n_anom = int(frac_anomalias * n)
    idx_anom = rng.choice(n, size=n_anom, replace=False)
    senal[idx_anom] += rng.choice([-1, 1], n_anom) * rng.uniform(3, 6, n_anom)
    es_anomalia[idx_anom] = True
    return t, senal, es_anomalia

t_sensor, senal, es_anomalia = generar_serie_sensores()
print(f"Longitud de la serie: {len(senal)}")
print(f"Anomalias inyectadas: {es_anomalia.sum()}")

# ======================================================================
# Construcción de características
# ======================================================================

from scipy.ndimage import median_filter

tendencia = median_filter(senal, size=15)   # tendencia local robusta
residuo = senal - tendencia                  # desviacion respecto a la tendencia
diferencia = np.gradient(senal)              # cambio local

X_sensor = np.column_stack([senal, residuo, diferencia])
X_sensor_sc = StandardScaler().fit_transform(X_sensor)
print(f"Matriz de caracteristicas: {X_sensor.shape}")

# ======================================================================
# Isolation Forest
# ======================================================================

from sklearn.ensemble import IsolationForest

iforest = IsolationForest(n_estimators=200, contamination=0.05, random_state=SEMILLA)
prediccion = iforest.fit_predict(X_sensor_sc)   # +1 normal, -1 anomalia
puntuacion = iforest.decision_function(X_sensor_sc)  # menor = mas anomalo

es_anomalia_pred = (prediccion == -1)
n_detectadas = es_anomalia_pred.sum()
print(f"Anomalias detectadas: {n_detectadas} ({100*n_detectadas/len(prediccion):.1f}%)")

idx_top = np.argsort(puntuacion)[:10]
print("Indices de las 10 mayores anomalias:", idx_top)

# ======================================================================
# Visualización y evaluación
# ======================================================================

fig, ax = plt.subplots(figsize=(ANCHO_IMPRESION, 3.0))
ax.plot(t_sensor, senal, color=GRIS, linewidth=0.6, alpha=0.8, label="Senal")
ax.scatter(t_sensor[es_anomalia], senal[es_anomalia], s=40, marker="o",
           facecolor="none", edgecolor=AZUL, linewidth=1.2, label="Anomalia real")
ax.scatter(t_sensor[es_anomalia_pred], senal[es_anomalia_pred], s=12, marker="x",
           color=ROJO, label="Detectada")
ax.set_xlabel("Tiempo")
ax.set_ylabel("Lectura del sensor")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
fig.tight_layout()
plt.show()

from sklearn.metrics import precision_score, recall_score

precision = precision_score(es_anomalia, es_anomalia_pred)
recall = recall_score(es_anomalia, es_anomalia_pred)
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"De las 10 mayores anomalias, reales: {es_anomalia[idx_top].sum()} / 10")

# ======================================================================
# Resumen
# ======================================================================

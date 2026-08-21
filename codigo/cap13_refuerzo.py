# -*- coding: utf-8 -*-
"""
Capítulo 13 · Aprendizaje por refuerzo

Script extraído del notebook cap13_refuerzo.ipynb
Libro: "Inteligencia artificial para estudiantes de ciencias"
Jorge Bravo Abad — Ediciones Pirámide, 2026
"""


# ======================================================================
# Capítulo 13 — Aprendizaje por refuerzo
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt

# Semilla global para reproducibilidad
np.random.seed(42)

# Estilo de figuras coherente con el resto del libro
plt.rcParams.update({
    "figure.figsize": (6.0, 4.2),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# ======================================================================
# Parte 1 — Q-learning: control de temperatura de un reactor
# ======================================================================

# ======================================================================
# 1.1 El entorno
# ======================================================================

class EntornoReactor:
    """Entorno simplificado tipo Gym para el control de temperatura de un reactor.

    Interfaz clasica de Gym:
        reset()  -> observacion
        step(a)  -> (observacion, recompensa, terminado, info)

    Estado observado : indice de la celda de temperatura (entero).
    Acciones         : 0 = enfriar, 1 = mantener, 2 = calentar.

    Con Gymnasium la interfaz cambia ligeramente: reset() -> (obs, info)
    y step() -> (obs, recompensa, terminated, truncated, info). Aqui usamos
    la version clasica por simplicidad.
    """
    def __init__(self, t_min=20.0, t_max=200.0, n_celdas=37,
                 t_objetivo=150.0, tolerancia=5.0, max_pasos=100, semilla=None):
        self.t_min, self.t_max = t_min, t_max
        self.n_celdas   = n_celdas
        self.t_objetivo = t_objetivo
        self.tolerancia = tolerancia
        self.max_pasos  = max_pasos
        self.delta      = (t_max - t_min) / (n_celdas - 1)  # paso termico por accion
        self.n_acciones = 3
        self.rng = np.random.default_rng(semilla)

    def _discretizar(self, t):
        idx = int(round((t - self.t_min) / self.delta))
        return int(np.clip(idx, 0, self.n_celdas - 1))

    def reset(self):
        # temperatura inicial aleatoria en todo el rango
        self.temperatura = self.rng.uniform(self.t_min, self.t_max)
        self.paso = 0
        return self._discretizar(self.temperatura)

    def step(self, accion):
        if accion == 0:      self.temperatura -= self.delta   # enfriar
        elif accion == 2:    self.temperatura += self.delta   # calentar
        # accion == 1: mantener
        self.temperatura += self.rng.normal(0, 0.3)           # ruido del proceso
        self.temperatura  = float(np.clip(self.temperatura, self.t_min, self.t_max))
        self.paso += 1

        error      = abs(self.temperatura - self.t_objetivo)
        en_ventana = error <= self.tolerancia
        if en_ventana:
            recompensa = 1.0
        else:
            recompensa = -error / (self.t_max - self.t_min)   # penalizacion normalizada

        terminado = self.paso >= self.max_pasos
        info = {"temperatura": self.temperatura, "en_ventana": en_ventana}
        return self._discretizar(self.temperatura), recompensa, terminado, info


entorno = EntornoReactor(semilla=42)
print(f"Numero de estados (celdas): {entorno.n_celdas}")
print(f"Numero de acciones        : {entorno.n_acciones}")
print(f"Temperatura objetivo      : {entorno.t_objetivo} +/- {entorno.tolerancia} C")
print(f"Celda objetivo            : indice {entorno._discretizar(entorno.t_objetivo)}")

# ======================================================================
# 1.2 El algoritmo de Q-learning tabular
# ======================================================================

def entrenar_qlearning(entorno, n_episodios=2000, alpha=0.1, gamma=0.95,
                       eps_inicio=1.0, eps_fin=0.05, semilla=0):
    """Q-learning tabular con exploracion epsilon-greedy decreciente.

    Devuelve la tabla Q aprendida y el retorno obtenido en cada episodio.
    """
    rng = np.random.default_rng(semilla)
    Q   = np.zeros((entorno.n_celdas, entorno.n_acciones))
    retornos = np.zeros(n_episodios)

    for episodio in range(n_episodios):
        # epsilon decae linealmente de eps_inicio a eps_fin
        eps = eps_fin + (eps_inicio - eps_fin) * (1 - episodio / n_episodios)
        estado    = entorno.reset()
        retorno   = 0.0
        terminado = False

        while not terminado:
            # seleccion epsilon-greedy
            if rng.random() < eps:
                accion = int(rng.integers(entorno.n_acciones))
            else:
                accion = int(np.argmax(Q[estado]))

            estado_sig, recompensa, terminado, _ = entorno.step(accion)

            # actualizacion de Q-learning (error de diferencia temporal)
            objetivo = recompensa + gamma * np.max(Q[estado_sig]) * (not terminado)
            Q[estado, accion] += alpha * (objetivo - Q[estado, accion])

            estado   = estado_sig
            retorno += recompensa

        retornos[episodio] = retorno

    return Q, retornos


Q, retornos = entrenar_qlearning(entorno, n_episodios=2000,
                                 alpha=0.1, gamma=0.95, semilla=0)

print(f"Retorno medio (primeros 50 episodios): {retornos[:50].mean():7.2f}")
print(f"Retorno medio (ultimos  50 episodios): {retornos[-50:].mean():7.2f}")

# ======================================================================
# 1.3 Curva de aprendizaje
# ======================================================================

def media_movil(x, ventana=50):
    return np.convolve(x, np.ones(ventana) / ventana, mode="valid")

suavizado = media_movil(retornos, ventana=50)

fig, ax = plt.subplots()
ax.plot(retornos, alpha=0.25, color="gray", label="retorno por episodio")
ax.plot(range(len(suavizado)), suavizado, color="C0", lw=2,
        label="media movil (50 ep.)")
ax.set_xlabel("Episodio")
ax.set_ylabel("Retorno")
ax.set_title("Curva de aprendizaje del Q-learning")
ax.legend()
fig.tight_layout()
plt.show()

# ======================================================================
# 1.4 La política aprendida
# ======================================================================

politica = np.argmax(Q, axis=1)
temperaturas_celda = entorno.t_min + np.arange(entorno.n_celdas) * entorno.delta
nombres_accion = {0: "enfriar", 1: "mantener", 2: "calentar"}
colores_accion = {0: "C0", 1: "C2", 2: "C3"}

fig, ax = plt.subplots(figsize=(7.5, 2.6))
for a in (0, 1, 2):
    mascara = politica == a
    ax.scatter(temperaturas_celda[mascara], np.zeros(mascara.sum()),
               s=90, marker="s", color=colores_accion[a], label=nombres_accion[a])
ax.axvline(entorno.t_objetivo, color="black", ls="--", lw=1)
ax.axvspan(entorno.t_objetivo - entorno.tolerancia,
           entorno.t_objetivo + entorno.tolerancia, color="black", alpha=0.08)
ax.text(entorno.t_objetivo, 0.04, "ventana\nobjetivo", ha="center", fontsize=9)
ax.set_yticks([])
ax.set_xlabel("Temperatura (C)")
ax.set_title("Politica aprendida en funcion de la temperatura")
ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.55))
ax.set_ylim(-0.1, 0.12)
fig.tight_layout()
plt.show()

# ======================================================================
# 1.5 Los valores $Q$ y una trayectoria con la política aprendida
# ======================================================================

fig, ax = plt.subplots()
for a in (0, 1, 2):
    ax.plot(temperaturas_celda, Q[:, a], marker="o", ms=3,
            color=colores_accion[a], label=nombres_accion[a])
ax.axvline(entorno.t_objetivo, color="black", ls="--", lw=1, label="objetivo")
ax.set_xlabel("Temperatura (C)")
ax.set_ylabel("Valor Q(s, a)")
ax.set_title("Valores Q aprendidos para cada accion")
ax.legend()
fig.tight_layout()
plt.show()

# Simular un episodio siguiendo la politica greedy (sin exploracion)
entorno_test = EntornoReactor(semilla=7)
estado = entorno_test.reset()
historial_temp = [entorno_test.temperatura]
terminado = False
while not terminado:
    accion = int(np.argmax(Q[estado]))
    estado, _, terminado, info = entorno_test.step(accion)
    historial_temp.append(info["temperatura"])

fig, ax = plt.subplots()
ax.plot(historial_temp, color="C0", lw=1.8)
ax.axhline(entorno.t_objetivo, color="black", ls="--", lw=1)
ax.axhspan(entorno.t_objetivo - entorno.tolerancia,
           entorno.t_objetivo + entorno.tolerancia, color="black", alpha=0.08,
           label="ventana objetivo")
ax.set_xlabel("Paso de tiempo")
ax.set_ylabel("Temperatura (C)")
ax.set_title("Control del reactor con la politica aprendida")
ax.legend()
fig.tight_layout()
plt.show()

pasos_en_ventana = sum(abs(t - entorno.t_objetivo) <= entorno.tolerancia
                       for t in historial_temp)
print(f"Temperatura inicial: {historial_temp[0]:.1f} C")
print(f"Pasos dentro de la ventana objetivo: {pasos_en_ventana} de {len(historial_temp)}")

# ======================================================================
# Parte 2 — Optimización bayesiana: condiciones de una reacción de síntesis
# ======================================================================

# ======================================================================
# 2.1 El espacio de búsqueda y la función objetivo
# ======================================================================

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Espacio de busqueda
espacio = [
    Real( 50.0, 200.0, name="temperatura"),     # grados Celsius
    Real(  1.0,  10.0, name="presion"),          # bar
    Real(  0.5,   4.0, name="tiempo_reaccion"),  # horas
]

def rendimiento_reaccion(temperatura, presion, tiempo_reaccion):
    """Rendimiento (%) de una reaccion hipotetica. En la practica, esta
    funcion ejecutaria el experimento o una simulacion costosa.
    Optimo en T=150 C, P=5 bar, t=2 h."""
    rend = (100
            - 0.005 * (temperatura - 150)**2
            - 0.02  * (presion - 5)**2
            - 2.0   * (tiempo_reaccion - 2)**2
            + np.random.normal(0, 1.5))          # ruido experimental
    return float(np.clip(rend, 0, 100))

@use_named_args(espacio)
def objetivo(**params):
    # gp_minimize MINIMIZA: devolvemos el rendimiento negado
    return -rendimiento_reaccion(**params)

# ======================================================================
# 2.2 El bucle de optimización bayesiana
# ======================================================================

resultado = gp_minimize(
    func             = objetivo,
    dimensions       = espacio,
    n_calls          = 40,        # total de evaluaciones
    n_initial_points = 8,         # evaluaciones iniciales aleatorias
    acq_func         = "EI",      # Expected Improvement
    noise            = 1.5**2,    # varianza del ruido experimental
    random_state     = 42,
)

print(f"Mejor rendimiento encontrado: {-resultado.fun:.2f}%")
print("Condiciones optimas:")
for nombre, valor in zip(["temperatura", "presion", "tiempo"], resultado.x):
    print(f"  {nombre:12s}: {valor:6.2f}")
print(f"Evaluaciones realizadas: {len(resultado.func_vals)}")

# ======================================================================
# 2.3 Convergencia y los mejores experimentos
# ======================================================================

rendimientos    = -np.array(resultado.func_vals)
mejor_acumulado = np.maximum.accumulate(rendimientos)

fig, ax = plt.subplots()
ax.plot(range(1, len(rendimientos) + 1), rendimientos, "o", alpha=0.4,
        color="gray", label="evaluacion individual")
ax.plot(range(1, len(mejor_acumulado) + 1), mejor_acumulado, "-o", ms=4,
        color="C0", label="mejor acumulado")
ax.axvline(8.5, color="C3", ls=":", lw=1)
ax.text(8.7, ax.get_ylim()[0] + 2, "fin de la fase inicial", color="C3",
        fontsize=9, rotation=90, va="bottom")
ax.set_xlabel("Numero de evaluaciones")
ax.set_ylabel("Rendimiento (%)")
ax.set_title("Convergencia de la optimizacion bayesiana")
ax.legend()
fig.tight_layout()
plt.show()

indices_ordenados = np.argsort(resultado.func_vals)[:5]
print("Cinco mejores experimentos:")
for i, idx in enumerate(indices_ordenados, start=1):
    T, P, t = resultado.x_iters[idx]
    rend = -resultado.func_vals[idx]
    print(f"  {i}. T={T:5.0f} C, P={P:4.1f} bar, t={t:3.1f} h  ->  {rend:5.1f}%")

# ======================================================================
# 2.4 Dependencia parcial estimada por el proceso gaussiano
# ======================================================================

from skopt.plots import plot_objective

plot_objective(resultado, n_samples=40,
               dimensions=["temperatura", "presion", "tiempo_reaccion"])
plt.gcf().set_size_inches(8.5, 8.5)
plt.tight_layout()
plt.show()

# ======================================================================
# 2.5 ¿Es realmente mejor que la búsqueda aleatoria?
# ======================================================================

import warnings

def busqueda_aleatoria(n_evaluaciones, semilla):
    rng = np.random.default_rng(semilla)
    rendimientos = []
    for _ in range(n_evaluaciones):
        T = rng.uniform( 50, 200)
        P = rng.uniform(  1,  10)
        t = rng.uniform(0.5,   4)
        rendimientos.append(rendimiento_reaccion(T, P, t))
    return np.maximum.accumulate(rendimientos)

N = 40            # presupuesto de evaluaciones
n_semillas = 20

curvas_bo   = np.zeros((n_semillas, N))
curvas_rand = np.zeros((n_semillas, N))

for s in range(n_semillas):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = gp_minimize(objetivo, espacio, n_calls=N, n_initial_points=8,
                          acq_func="EI", noise=1.5**2, random_state=s)
    curvas_bo[s]   = np.maximum.accumulate(-np.array(res.func_vals))
    curvas_rand[s] = busqueda_aleatoria(N, semilla=100 + s)

media_bo,   err_bo   = curvas_bo.mean(0),   curvas_bo.std(0)   / np.sqrt(n_semillas)
media_rand, err_rand = curvas_rand.mean(0), curvas_rand.std(0) / np.sqrt(n_semillas)

x = np.arange(1, N + 1)
fig, ax = plt.subplots()
ax.plot(x, media_bo, "-o", ms=3, color="C0", label="optimizacion bayesiana")
ax.fill_between(x, media_bo - err_bo, media_bo + err_bo, color="C0", alpha=0.2)
ax.plot(x, media_rand, "-s", ms=3, color="C3", label="busqueda aleatoria")
ax.fill_between(x, media_rand - err_rand, media_rand + err_rand, color="C3", alpha=0.2)
ax.set_xlabel("Numero de evaluaciones")
ax.set_ylabel("Mejor rendimiento acumulado (%)")
ax.set_title(f"BO frente a busqueda aleatoria (media de {n_semillas} semillas)")
ax.legend()
fig.tight_layout()
plt.show()

for k in (5, 10, 20, 40):
    print(f"Tras {k:2d} evaluaciones:  BO = {media_bo[k-1]:5.2f}%   "
          f"aleatoria = {media_rand[k-1]:5.2f}%")

# ======================================================================
# Resumen
# ======================================================================

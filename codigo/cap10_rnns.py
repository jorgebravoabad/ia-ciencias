# -*- coding: utf-8 -*-
"""
Capítulo 10 · Redes neuronales recurrentes y series temporales

Script extraído del notebook cap10_rnns.ipynb
Libro: "Inteligencia artificial para estudiantes de ciencias"
Jorge Bravo Abad — Ediciones Pirámide, 2026
"""


# ======================================================================
# Capitulo 10 - Redes neuronales recurrentes y series temporales
# ======================================================================

# ======================================================================
# Notebook: `cap10_rnns.ipynb`
# ======================================================================

# ======================================================================
# 0. Entorno y reproducibilidad
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Reproducibilidad
SEMILLA = 0
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", dispositivo)
print("Version de PyTorch:", torch.__version__)

# ======================================================================
# 1. Generacion del atractor de Lorenz con RK4
# ======================================================================

def lorenz_derivadas(u, sigma=10., rho=28., beta=8./3.):
    """Calcula (dx/dt, dy/dt, dz/dt) para el sistema de Lorenz."""
    x, y, z = u
    return np.array([sigma*(y - x),
                     x*(rho - z) - y,
                     x*y - beta*z])

def rk4_paso(f, u, dt):
    """Un paso de Runge-Kutta de orden 4."""
    k1 = f(u)
    k2 = f(u + dt/2 * k1)
    k3 = f(u + dt/2 * k2)
    k4 = f(u + dt   * k3)
    return u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Generar trayectoria
dt   = 0.01
N    = 50_000
u0   = np.array([1.0, 1.0, 1.0])
traj = np.zeros((N, 3))
traj[0] = u0
for i in range(1, N):
    traj[i] = rk4_paso(lorenz_derivadas, traj[i-1], dt)

# Descartar transitorio inicial (primeros 5000 pasos)
traj = traj[5000:]

print(f"Forma de la trayectoria: {traj.shape}")  # (45000, 3)
print(f"Media: {traj.mean(axis=0).round(2)}")
print(f"Std:   {traj.std(axis=0).round(2)}")

# ======================================================================
# Visualizacion del atractor
# ======================================================================

fig = plt.figure(figsize=(11, 4))

# Atractor 3D
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.3, color="#1E5FA5")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Atractor de Lorenz")

# Componentes vs tiempo (primeros 3000 pasos)
ax2 = fig.add_subplot(1, 2, 2)
t = np.arange(3000) * dt
ax2.plot(t, traj[:3000, 0], lw=0.8, label="x")
ax2.plot(t, traj[:3000, 1], lw=0.8, label="y")
ax2.plot(t, traj[:3000, 2], lw=0.8, label="z")
ax2.set_xlabel("tiempo simulado")
ax2.set_ylabel("estado")
ax2.set_title("Componentes frente al tiempo")
ax2.legend()
plt.tight_layout()
plt.show()

# ======================================================================
# 2. Preprocesado: normalizacion y ventanas deslizantes
# ======================================================================

# ----- Normalizacion (solo con la estadistica del entrenamiento) -----
n_train_raw = int(0.7 * len(traj))
media = traj[:n_train_raw].mean(axis=0)
std   = traj[:n_train_raw].std(axis=0)
traj_norm = (traj - media) / std

# ----- Ventanas deslizantes -----
L = 40  # longitud del contexto (pasos de tiempo)

def crear_ventanas(datos, L):
    X = np.array([datos[i:i+L]   for i in range(len(datos)-L)])
    y = np.array([datos[i+L]     for i in range(len(datos)-L)])
    return X, y

X_all, y_all = crear_ventanas(traj_norm, L)

# Split temporal (NO aleatorio!)
n_total = len(X_all)
n_train = int(0.7 * n_total)
n_val   = int(0.15 * n_total)

X_train = torch.tensor(X_all[:n_train],         dtype=torch.float32)
y_train = torch.tensor(y_all[:n_train],         dtype=torch.float32)
X_val   = torch.tensor(X_all[n_train:n_train+n_val], dtype=torch.float32)
y_val   = torch.tensor(y_all[n_train:n_train+n_val], dtype=torch.float32)
X_test  = torch.tensor(X_all[n_train+n_val:],   dtype=torch.float32)
y_test  = torch.tensor(y_all[n_train+n_val:],   dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=256, shuffle=True)

print(f"Ventanas totales: {n_total}")
print(f"Entrenamiento: {len(X_train)}   Validacion: {len(X_val)}   Test: {len(X_test)}")
print(f"Forma de X_train: {tuple(X_train.shape)}  (batch, L, 3)")

# ======================================================================
# 3. Modelo LSTM
# ======================================================================

# ----- Modelo LSTM -----
class LorenzLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64,
                 num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # [batch, seq_len, features]
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, L, 3]
        out, _ = self.lstm(x)   # out: [batch, L, hidden_size]
        return self.fc(out[:, -1, :])  # solo el ultimo paso: [batch, 3]

# ======================================================================
# Bucle de entrenamiento
# ======================================================================

def entrenar(model, train_loader, X_val, y_val, n_epocas=100,
             lr=1e-3, ruta_guardado=None, verbose=True):
    """Entrena un modelo de secuencia y devuelve el historial de perdidas."""
    model = model.to(dispositivo)
    criterio   = nn.MSELoss()
    optimizador = torch.optim.Adam(model.parameters(), lr=lr)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, patience=5, factor=0.5, min_lr=1e-5)

    X_val_d, y_val_d = X_val.to(dispositivo), y_val.to(dispositivo)
    hist_train, hist_val = [], []
    mejor_val_loss = float("inf")

    for epoca in range(1, n_epocas + 1):
        model.train()
        perdida_train = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dispositivo), yb.to(dispositivo)
            optimizador.zero_grad()
            pred = model(xb)
            perdida = criterio(pred, yb)
            perdida.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizador.step()
            perdida_train += perdida.item() * len(xb)
        perdida_train /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_pred  = model(X_val_d)
            perdida_val = criterio(val_pred, y_val_d).item()
        planificador.step(perdida_val)

        hist_train.append(perdida_train)
        hist_val.append(perdida_val)

        if perdida_val < mejor_val_loss:
            mejor_val_loss = perdida_val
            if ruta_guardado is not None:
                torch.save(model.state_dict(), ruta_guardado)

        if verbose and epoca % 10 == 0:
            print(f"Epoca {epoca:3d}: train={perdida_train:.6f}  val={perdida_val:.6f}")

    return {"train": hist_train, "val": hist_val, "mejor_val": mejor_val_loss}

N_EPOCAS = 100  # reducir (p. ej. a 30) para una ejecucion rapida

torch.manual_seed(SEMILLA)
modelo_lstm = LorenzLSTM(hidden_size=128, num_layers=2)
hist_lstm = entrenar(modelo_lstm, train_loader, X_val, y_val,
                     n_epocas=N_EPOCAS, ruta_guardado="mejor_lstm_lorenz.pt")
print(f"\nMejor val_loss (LSTM): {hist_lstm['mejor_val']:.6f}")

# ======================================================================
# 4. Modelo GRU y comparacion con LSTM
# ======================================================================

class LorenzGRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64,
                 num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)            # out: [batch, L, hidden_size]
        return self.fc(out[:, -1, :])   # solo el ultimo paso

torch.manual_seed(SEMILLA)
modelo_gru = LorenzGRU(hidden_size=128, num_layers=2)
hist_gru = entrenar(modelo_gru, train_loader, X_val, y_val,
                    n_epocas=N_EPOCAS, ruta_guardado="mejor_gru_lorenz.pt")
print(f"\nMejor val_loss (GRU): {hist_gru['mejor_val']:.6f}")

def contar_parametros(m):
    return sum(p.numel() for p in m.parameters())

print(f"Parametros LSTM: {contar_parametros(modelo_lstm):,}")
print(f"Parametros GRU : {contar_parametros(modelo_gru):,}")
print(f"Reduccion GRU vs LSTM: "
      f"{100*(1 - contar_parametros(modelo_gru)/contar_parametros(modelo_lstm)):.1f} %")

# ======================================================================
# Curvas de aprendizaje
# ======================================================================

plt.figure(figsize=(8, 4))
plt.semilogy(hist_lstm["val"], label="LSTM (validacion)")
plt.semilogy(hist_gru["val"],  label="GRU (validacion)")
plt.xlabel("Epoca")
plt.ylabel("MSE de validacion (escala log)")
plt.title("LSTM frente a GRU: curvas de aprendizaje")
plt.legend()
plt.tight_layout()
plt.show()

# ======================================================================
# Error de prediccion a un paso en test
# ======================================================================

def mse_test_un_paso(model, ruta, X_test, y_test):
    model.load_state_dict(torch.load(ruta, map_location=dispositivo))
    model.to(dispositivo).eval()
    with torch.no_grad():
        pred = model(X_test.to(dispositivo)).cpu().numpy()
    return ((pred - y_test.numpy())**2).mean()

mse_lstm = mse_test_un_paso(modelo_lstm, "mejor_lstm_lorenz.pt", X_test, y_test)
mse_gru  = mse_test_un_paso(modelo_gru,  "mejor_gru_lorenz.pt",  X_test, y_test)

print(f"MSE test (1 paso) - LSTM: {mse_lstm:.6f}")
print(f"MSE test (1 paso) - GRU : {mse_gru:.6f}")

# ======================================================================
# 5. Prediccion autorregresiva y horizonte de predictibilidad
# ======================================================================

def predecir_autorregresivo(model, ventana_inicial, n_predicciones):
    """Genera una trayectoria autorregresiva realimentando cada prediccion."""
    model.eval()
    ventana   = ventana_inicial.clone().to(dispositivo)  # [1, L, 3]
    pred_traj = []
    with torch.no_grad():
        for _ in range(n_predicciones):
            siguiente = model(ventana)          # [1, 3]
            pred_traj.append(siguiente.cpu().numpy())
            # Desplazar ventana: quitar el primer paso, anadir la prediccion
            ventana = torch.cat([ventana[:, 1:, :],
                                 siguiente.unsqueeze(1)], dim=1)
    return np.array(pred_traj).squeeze()        # [n_predicciones, 3]

# Cargar el mejor modelo LSTM y predecir
modelo_lstm.load_state_dict(torch.load("mejor_lstm_lorenz.pt",
                                       map_location=dispositivo))
n_predicciones = 500
pred_traj = predecir_autorregresivo(modelo_lstm, X_test[0:1], n_predicciones)

# Trayectoria real correspondiente (normalizada, para comparar)
real_traj = y_all[n_train+n_val : n_train+n_val+n_predicciones]

# Error cuadratico por paso en unidades normalizadas
mse_por_paso = ((pred_traj - real_traj)**2).mean(axis=1)

# Desnormalizar solo para representar las trayectorias
pred_fis = pred_traj * std + media
real_fis = real_traj * std + media

# ======================================================================
# Visualizacion del horizonte
# ======================================================================

umbral = 1.0
if (mse_por_paso > umbral).any():
    horizonte = int(np.argmax(mse_por_paso > umbral))
else:
    horizonte = len(mse_por_paso)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Componente x: real vs predicha
ax1.plot(real_fis[:, 0], label="real",    lw=1.5)
ax1.plot(pred_fis[:, 0], label="predicha", lw=1.5, linestyle="--")
ax1.axvline(horizonte, color="gray", linestyle=":",
            label=f"horizonte ~ {horizonte} pasos")
ax1.set_xlabel("Pasos de prediccion autorregresiva")
ax1.set_ylabel("x (unidades fisicas)")
ax1.set_title("Trayectoria real frente a predicha")
ax1.legend()

# MSE por paso
ax2.semilogy(mse_por_paso)
ax2.axhline(y=umbral, color="red", linestyle="--",
            label="Umbral de predictibilidad")
ax2.set_xlabel("Pasos de prediccion autorregresiva")
ax2.set_ylabel("MSE (normalizado, escala log)")
ax2.set_title("Horizonte de predictibilidad del sistema de Lorenz")
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Horizonte de predictibilidad (MSE > {umbral}): {horizonte} pasos")
print(f"Limite teorico aproximado (Lyapunov): ~110 pasos")

# ======================================================================
# 6. Horizonte de predictibilidad en funcion de la longitud del contexto
# ======================================================================

def construir_loaders(L, traj_norm):
    """Construye loaders y tensores con split temporal para un contexto dado."""
    X_all_L, y_all_L = crear_ventanas(traj_norm, L)
    n_tot = len(X_all_L)
    n_tr  = int(0.7 * n_tot)
    n_v   = int(0.15 * n_tot)
    Xtr = torch.tensor(X_all_L[:n_tr],          dtype=torch.float32)
    ytr = torch.tensor(y_all_L[:n_tr],          dtype=torch.float32)
    Xv  = torch.tensor(X_all_L[n_tr:n_tr+n_v],  dtype=torch.float32)
    yv  = torch.tensor(y_all_L[n_tr:n_tr+n_v],  dtype=torch.float32)
    Xte = torch.tensor(X_all_L[n_tr+n_v:],      dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
    info = {"X_all": X_all_L, "y_all": y_all_L,
            "n_tr": n_tr, "n_v": n_v, "X_test": Xte, "X_val": Xv, "y_val": yv}
    return loader, info

def horizonte_para_L(L, n_epocas=40, n_pred=500, umbral=1.0):
    torch.manual_seed(SEMILLA)
    loader, info = construir_loaders(L, traj_norm)
    modelo = LorenzLSTM(hidden_size=64, num_layers=2)
    entrenar(modelo, loader, info["X_val"], info["y_val"],
             n_epocas=n_epocas, ruta_guardado=None, verbose=False)
    pred = predecir_autorregresivo(modelo, info["X_test"][0:1], n_pred)
    inicio = info["n_tr"] + info["n_v"]
    real = info["y_all"][inicio : inicio + n_pred]
    mse  = ((pred - real)**2).mean(axis=1)
    if (mse > umbral).any():
        return int(np.argmax(mse > umbral))
    return len(mse)

contextos = [10, 20, 40, 80, 160]
horizontes = []
for L_i in contextos:
    h = horizonte_para_L(L_i, n_epocas=40)
    horizontes.append(h)
    print(f"L = {L_i:3d}  ->  horizonte = {h} pasos")

plt.figure(figsize=(8, 4))
plt.plot(contextos, horizontes, marker="o")
plt.axhline(y=110, color="red", linestyle="--",
            label="Limite teorico (Lyapunov ~110 pasos)")
plt.xlabel("Longitud del contexto L (pasos)")
plt.ylabel("Horizonte de predictibilidad (pasos)")
plt.title("Horizonte de predictibilidad frente a longitud del contexto")
plt.legend()
plt.tight_layout()
plt.show()

# ======================================================================
# 7. Resumen
# ======================================================================

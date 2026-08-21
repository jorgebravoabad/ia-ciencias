# -*- coding: utf-8 -*-
"""
Capítulo 11 · Modelos generativos

Script extraído del notebook cap11_generativos.ipynb
Libro: "Inteligencia artificial para estudiantes de ciencias"
Jorge Bravo Abad — Ediciones Pirámide, 2026
"""


# ======================================================================
# Capítulo 11 — Modelos generativos
# ======================================================================

# ======================================================================
# 0. Preparación del entorno
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Semilla para reproducibilidad
SEMILLA = 0
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__)
print("Dispositivo:", dispositivo)

# ======================================================================
# 1. Conjunto de datos: espectros Raman de polímeros
# ======================================================================

N_PUNTOS = 1024          # dimension de cada espectro (input_dim del VAE)
N_TIPOS  = 4             # tipos de polimero
N_POR_TIPO = 300         # espectros por tipo

eje = np.linspace(0, 1, N_PUNTOS)   # desplazamiento Raman normalizado

# Posiciones (centro), anchura e intensidad de los picos caracteristicos de
# cada tipo de polimero. Cada fila de la lista es un tipo.
picos_por_tipo = [
    [(0.15, 0.012, 1.0), (0.45, 0.010, 0.7), (0.78, 0.015, 0.9)],
    [(0.22, 0.010, 0.9), (0.55, 0.014, 1.0), (0.70, 0.012, 0.6)],
    [(0.10, 0.013, 0.8), (0.38, 0.011, 1.0), (0.62, 0.010, 0.7), (0.88, 0.016, 0.5)],
    [(0.30, 0.012, 1.0), (0.50, 0.009, 0.5), (0.82, 0.013, 0.8)],
]

def lorentziana(x, centro, anchura):
    return (anchura**2) / ((x - centro)**2 + anchura**2)

def genera_espectro(picos, rng):
    y = np.zeros_like(eje)
    for centro, anchura, intensidad in picos:
        # Pequena variabilidad fisica entre muestras del mismo tipo
        c = centro + rng.normal(0, 0.004)
        a = anchura * (1 + rng.normal(0, 0.08))
        inten = intensidad * (1 + rng.normal(0, 0.06))
        y = y + inten * lorentziana(eje, c, a)
    # Linea de base suave
    base = 0.05 + 0.10 * np.sin(2 * np.pi * eje * rng.uniform(0.5, 1.5))
    y = y + base
    # Ruido de medida
    y = y + rng.normal(0, 0.01, size=y.shape)
    return y

rng = np.random.default_rng(SEMILLA)
X_list, y_list = [], []
for tipo, picos in enumerate(picos_por_tipo):
    for _ in range(N_POR_TIPO):
        X_list.append(genera_espectro(picos, rng))
        y_list.append(tipo)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
print("Forma de X:", X.shape, " etiquetas:", np.bincount(y))

# Normalizacion global al rango [-1, 1]
x_min, x_max = X.min(), X.max()
X_norm = 2 * (X - x_min) / (x_max - x_min) - 1.0

# Mezcla y particion train/test
idx = rng.permutation(len(X_norm))
X_norm, y = X_norm[idx], y[idx]
n_train = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:n_train], X_norm[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
print("Entrenamiento:", X_train.shape, " Test:", X_test.shape)

# Visualizamos un espectro representativo de cada tipo
fig, ax = plt.subplots(figsize=(7, 4))
for tipo in range(N_TIPOS):
    i = np.where(y_train == tipo)[0][0]
    ax.plot(eje, X_train[i], label=f"Tipo {tipo}", lw=1.2)
ax.set_xlabel("Desplazamiento Raman (normalizado)")
ax.set_ylabel("Intensidad (normalizada)")
ax.set_title("Un espectro representativo de cada tipo de polimero")
ax.legend()
plt.tight_layout(); plt.show()

# DataLoaders
tensor_train = torch.tensor(X_train)
tensor_y     = torch.tensor(y_train)
train_ds     = TensorDataset(tensor_train, tensor_y)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
print("Lotes por epoca:", len(train_loader))

# ======================================================================
# 2. Definición del VAE
# ======================================================================

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=2):
        super().__init__()
        # Codificador: x -> (mu, log_var)
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decodificador: z -> x_hat (salida en [-1, 1] con Tanh)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Tanh(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparametrizacion(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)          # N(0, I)
        return mu + eps * std                # truco de reparametrizacion

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparametrizacion(mu, log_var)
        x_hat       = self.decode(z)
        return x_hat, mu, log_var


def perdida_vae(x, x_hat, mu, log_var, beta=1.0):
    '''ELBO = reconstruccion + beta * KL (por muestra del lote).'''
    reconstruccion = F.mse_loss(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (reconstruccion + beta * kl) / x.size(0)

# ======================================================================
# 3. Entrenamiento del VAE
# ======================================================================

vae        = VAE(input_dim=N_PUNTOS, latent_dim=2).to(dispositivo)
optimizador = torch.optim.Adam(vae.parameters(), lr=1e-3)
N_EPOCAS   = 60
BETA       = 1.0

historial_vae = []
for epoca in range(1, N_EPOCAS + 1):
    vae.train()
    perdida_total = 0.0
    for lote in train_loader:
        # El DataLoader devuelve (espectros, etiquetas); usamos solo los espectros
        xb = lote[0].to(dispositivo)
        xb = xb.view(xb.size(0), -1)
        optimizador.zero_grad()
        x_hat, mu, log_var = vae(xb)
        perdida = perdida_vae(xb, x_hat, mu, log_var, beta=BETA)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item()
    historial_vae.append(perdida_total / len(train_loader))
    if epoca % 10 == 0:
        print(f"Epoca {epoca:3d}: perdida = {historial_vae[-1]:.4f}")

plt.figure(figsize=(6, 3.5))
plt.plot(range(1, N_EPOCAS + 1), historial_vae, lw=1.5)
plt.xlabel("Epoca"); plt.ylabel("Perdida ELBO")
plt.title("Curva de entrenamiento del VAE")
plt.tight_layout(); plt.show()

# ======================================================================
# Comprobación de la reconstrucción
# ======================================================================

vae.eval()
with torch.no_grad():
    xb = torch.tensor(X_test[:4]).to(dispositivo)
    x_hat, _, _ = vae(xb)
    xb = xb.cpu().numpy(); x_hat = x_hat.cpu().numpy()

fig, axes = plt.subplots(1, 4, figsize=(13, 3))
for k in range(4):
    axes[k].plot(eje, xb[k], lw=1.0, label="original")
    axes[k].plot(eje, x_hat[k], lw=1.0, ls="--", label="reconstruido")
    axes[k].set_title(f"Espectro {k}")
    if k == 0: axes[k].legend(fontsize=8)
plt.tight_layout(); plt.show()

# ======================================================================
# 4. Visualización del espacio latente 2D
# ======================================================================

vae.eval()
with torch.no_grad():
    todos = torch.tensor(X_train).to(dispositivo)
    mu, _ = vae.encode(todos)
    Z = mu.cpu().numpy()

plt.figure(figsize=(6, 5))
marcadores = ["o", "s", "^", "D"]
for tipo in range(N_TIPOS):
    sel = y_train == tipo
    plt.scatter(Z[sel, 0], Z[sel, 1], s=14, alpha=0.6,
                marker=marcadores[tipo], label=f"Tipo {tipo}")
plt.xlabel("$z_1$"); plt.ylabel("$z_2$")
plt.title("Espacio latente 2D del VAE (coloreado por tipo)")
plt.legend(); plt.tight_layout(); plt.show()

# ======================================================================
# 5. Generación de espectros nuevos
# ======================================================================

# ======================================================================
# 5.1 Muestreo del prior
# ======================================================================

vae.eval()
with torch.no_grad():
    z_nuevo   = torch.randn(6, 2).to(dispositivo)    # muestrear del prior
    espectros = vae.decode(z_nuevo).cpu().numpy()

fig, ax = plt.subplots(figsize=(7, 4))
for k in range(6):
    ax.plot(eje, espectros[k] + k * 0.4, lw=1.0)   # desplazados verticalmente
ax.set_xlabel("Desplazamiento Raman (normalizado)")
ax.set_ylabel("Intensidad (desplazada)")
ax.set_title("Seis espectros generados muestreando del prior")
ax.set_yticks([])
plt.tight_layout(); plt.show()

# ======================================================================
# 5.2 Interpolación en el espacio latente
# ======================================================================

# Dos muestras de tipos distintos
i1 = np.where(y_train == 0)[0][0]
i2 = np.where(y_train == 2)[0][0]

vae.eval()
with torch.no_grad():
    par = torch.tensor(X_train[[i1, i2]]).to(dispositivo)
    mu_par, _ = vae.encode(par)
    z1, z2 = mu_par[0], mu_par[1]

    ts = np.linspace(0, 1, 7)
    interp = []
    for t in ts:
        z = (1 - t) * z1 + t * z2
        interp.append(vae.decode(z.unsqueeze(0)).cpu().numpy()[0])

fig, ax = plt.subplots(figsize=(7, 5))
for k, (t, esp) in enumerate(zip(ts, interp)):
    ax.plot(eje, esp + k * 0.4, lw=1.0, label=f"t = {t:.2f}")
ax.set_xlabel("Desplazamiento Raman (normalizado)")
ax.set_ylabel("Intensidad (desplazada)")
ax.set_title("Interpolacion latente del Tipo 0 (t=0) al Tipo 2 (t=1)")
ax.set_yticks([]); ax.legend(fontsize=8, loc="upper right")
plt.tight_layout(); plt.show()

# ======================================================================
# 6. Comparación con una GAN
# ======================================================================

def bloque_mlp(dim_in, dim_out, activacion=nn.ReLU):
    return nn.Sequential(nn.Linear(dim_in, dim_out),
                         nn.BatchNorm1d(dim_out),
                         activacion())

latent_dim = 64
data_dim   = N_PUNTOS

G = nn.Sequential(
    bloque_mlp(latent_dim, 256),
    bloque_mlp(256, 512),
    nn.Linear(512, data_dim),
    nn.Tanh()              # salida en [-1, 1] (datos normalizados)
).to(dispositivo)

D = nn.Sequential(
    bloque_mlp(data_dim, 512, nn.LeakyReLU),
    bloque_mlp(512, 256, nn.LeakyReLU),
    nn.Linear(256, 1),
    nn.Sigmoid()
).to(dispositivo)

criterio = nn.BCELoss()
opt_G    = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D    = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

real_label, fake_label = 1.0, 0.0

N_EPOCAS_GAN = 80
hist_D, hist_G = [], []

for epoca in range(1, N_EPOCAS_GAN + 1):
    perdida_D_ep, perdida_G_ep, n_lotes = 0.0, 0.0, 0
    for lote in train_loader:
        xb = lote[0].to(dispositivo)
        xb = xb.view(xb.size(0), -1)
        bs = xb.size(0)

        # ----- Paso del discriminador -----
        opt_D.zero_grad()
        pred_real = D(xb)
        loss_real = criterio(pred_real,
                             torch.full((bs, 1), real_label, device=dispositivo))
        z         = torch.randn(bs, latent_dim, device=dispositivo)
        x_fake    = G(z).detach()   # detach: no actualizar G en este paso
        pred_fake = D(x_fake)
        loss_fake = criterio(pred_fake,
                             torch.full((bs, 1), fake_label, device=dispositivo))
        loss_D    = (loss_real + loss_fake) / 2
        loss_D.backward()
        opt_D.step()

        # ----- Paso del generador -----
        opt_G.zero_grad()
        z         = torch.randn(bs, latent_dim, device=dispositivo)
        x_fake    = G(z)
        pred_G    = D(x_fake)       # el G quiere que D diga "real"
        loss_G    = criterio(pred_G,
                             torch.full((bs, 1), real_label, device=dispositivo))
        loss_G.backward()
        opt_G.step()

        perdida_D_ep += loss_D.item(); perdida_G_ep += loss_G.item(); n_lotes += 1

    hist_D.append(perdida_D_ep / n_lotes)
    hist_G.append(perdida_G_ep / n_lotes)
    if epoca % 20 == 0:
        print(f"Epoca {epoca:3d}: loss_D={hist_D[-1]:.4f}  loss_G={hist_G[-1]:.4f}")

import math
plt.figure(figsize=(6, 3.5))
plt.plot(hist_D, lw=1.3, label="Discriminador $D$")
plt.plot(hist_G, lw=1.3, label="Generador $G$")
plt.axhline(math.log(2), color="gray", ls=":", lw=1.0, label=r"$\log 2$")
plt.xlabel("Epoca"); plt.ylabel("Perdida")
plt.title("Curvas de perdida de la GAN")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()

# ======================================================================
# Espectros generados por la GAN
# ======================================================================

G.eval()
with torch.no_grad():
    z = torch.randn(6, latent_dim, device=dispositivo)
    espectros_gan = G(z).cpu().numpy()
G.train()  # BatchNorm1d necesita modo train para lotes de tamano 1 en otros usos

fig, ax = plt.subplots(figsize=(7, 4))
for k in range(6):
    ax.plot(eje, espectros_gan[k] + k * 0.4, lw=1.0)
ax.set_xlabel("Desplazamiento Raman (normalizado)")
ax.set_ylabel("Intensidad (desplazada)")
ax.set_title("Seis espectros generados por la GAN")
ax.set_yticks([])
plt.tight_layout(); plt.show()

# ======================================================================
# Comparación VAE frente a GAN
# ======================================================================

vae.eval()
with torch.no_grad():
    z_vae = torch.randn(50, 2).to(dispositivo)
    muestras_vae = vae.decode(z_vae).cpu().numpy()
    z_gan = torch.randn(50, latent_dim, device=dispositivo)
    G.eval(); muestras_gan = G(z_gan).cpu().numpy(); G.train()

reales = X_train

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
axes[0].plot(eje, reales[:8].T, lw=0.7); axes[0].set_title("Reales")
axes[1].plot(eje, muestras_vae[:8].T, lw=0.7); axes[1].set_title("VAE")
axes[2].plot(eje, muestras_gan[:8].T, lw=0.7); axes[2].set_title("GAN")
for a in axes: a.set_xlabel("Desplazamiento Raman")
axes[0].set_ylabel("Intensidad")
plt.tight_layout(); plt.show()

# Distancia entre espectro medio generado y espectro medio real
d_vae = np.linalg.norm(muestras_vae.mean(0) - reales.mean(0))
d_gan = np.linalg.norm(muestras_gan.mean(0) - reales.mean(0))
print(f"Distancia |media generada - media real|  VAE: {d_vae:.4f}   GAN: {d_gan:.4f}")

# ======================================================================
# 7. Resumen
# ======================================================================

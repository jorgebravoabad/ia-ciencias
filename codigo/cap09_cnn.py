# -*- coding: utf-8 -*-
"""
Capítulo 9 · Redes neuronales convolucionales

Script extraído del notebook cap09_cnn.ipynb
Libro: "Inteligencia artificial para estudiantes de ciencias"
Jorge Bravo Abad — Ediciones Pirámide, 2026
"""


# ======================================================================
# Capítulo 9 — Redes neuronales convolucionales
# ======================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt
import time

torch.manual_seed(0)
np.random.seed(0)

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {dispositivo}")

# ======================================================================
# 1. Una CNN desde cero sobre MNIST
# ======================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_clases=10):
        super().__init__()
        # Bloque convolucional
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        # Cabeza clasificadora
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, num_clases)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # -> 32 x 14 x 14
        x = self.pool(self.relu(self.conv2(x)))  # -> 64 x 7 x 7
        x = x.view(x.size(0), -1)                # -> 3136
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

modelo = SimpleCNN().to(dispositivo)
print(modelo)

# ======================================================================
# Dónde viven los parámetros
# ======================================================================

def contar(modulo):
    return sum(p.numel() for p in modulo.parameters())

print(f"conv1: {contar(modelo.conv1):>8,}   (3*3*1 + 1) * 32")
print(f"conv2: {contar(modelo.conv2):>8,}   (3*3*32 + 1) * 64")
print(f"fc1  : {contar(modelo.fc1):>8,}   (64*7*7 + 1) * 128")
print(f"fc2  : {contar(modelo.fc2):>8,}   (128 + 1) * 10")
print(f"TOTAL: {contar(modelo):>8,}")

densos = contar(modelo.fc1) + contar(modelo.fc2)
print(f"\nFraccion en capas densas: {100 * densos / contar(modelo):.1f}%")

# ======================================================================
# Datos: MNIST
# ======================================================================

transformacion = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # media y std de MNIST
])

datos_train_full = datasets.MNIST("./data", train=True,  download=True,
                                  transform=transformacion)
datos_test       = datasets.MNIST("./data", train=False, download=True,
                                  transform=transformacion)

# Particion train / validacion
n_val = 5000
n_train = len(datos_train_full) - n_val
datos_train, datos_val = torch.utils.data.random_split(
    datos_train_full, [n_train, n_val],
    generator=torch.Generator().manual_seed(0))

loader_train = DataLoader(datos_train, batch_size=128, shuffle=True)
loader_val   = DataLoader(datos_val,   batch_size=256, shuffle=False)
loader_test  = DataLoader(datos_test,  batch_size=256, shuffle=False)

print(f"train: {len(datos_train)}  val: {len(datos_val)}  test: {len(datos_test)}")

# Visualizacion de unos cuantos digitos
imagenes, etiquetas = next(iter(loader_train))
fig, axes = plt.subplots(2, 5, figsize=(7, 3))
for ax, img, et in zip(axes.ravel(), imagenes, etiquetas):
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"{et.item()}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# ======================================================================
# Bucle de entrenamiento y evaluación
# ======================================================================

criterio    = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

def entrenar_epoca(modelo, loader, optimizador, criterio, dispositivo):
    modelo.train()
    perdida_acum, correctos, total = 0.0, 0, 0
    for imagenes, etiquetas in loader:
        imagenes  = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)
        optimizador.zero_grad()
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()
        perdida_acum += perdida.item() * imagenes.size(0)
        correctos    += (salidas.argmax(1) == etiquetas).sum().item()
        total        += etiquetas.size(0)
    return perdida_acum / total, correctos / total

def evaluar(modelo, loader, criterio, dispositivo):
    modelo.eval()
    perdida_acum, correctos, total = 0.0, 0, 0
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes  = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)
            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)
            perdida_acum += perdida.item() * imagenes.size(0)
            correctos    += (salidas.argmax(1) == etiquetas).sum().item()
            total        += etiquetas.size(0)
    return perdida_acum / total, correctos / total

num_epocas = 10
hist = {"tl": [], "ta": [], "vl": [], "va": []}

t0 = time.time()
for epoca in range(1, num_epocas + 1):
    tl, ta = entrenar_epoca(modelo, loader_train, optimizador, criterio, dispositivo)
    vl, va = evaluar(modelo, loader_val, criterio, dispositivo)
    hist["tl"].append(tl); hist["ta"].append(ta)
    hist["vl"].append(vl); hist["va"].append(va)
    print(f"Epoca {epoca:2d}: train loss={tl:.4f} acc={ta:.4f} | "
          f"val loss={vl:.4f} acc={va:.4f}")
print(f"\nTiempo total: {time.time() - t0:.1f} s")

test_loss, test_acc = evaluar(modelo, loader_test, criterio, dispositivo)
print(f"Exactitud en test: {test_acc:.4f}")

# Curvas de aprendizaje
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
ep = range(1, num_epocas + 1)
ax1.plot(ep, hist["tl"], "o-", label="train")
ax1.plot(ep, hist["vl"], "s--", label="val")
ax1.set_xlabel("epoca"); ax1.set_ylabel("perdida"); ax1.legend(); ax1.set_title("Perdida")
ax2.plot(ep, hist["ta"], "o-", label="train")
ax2.plot(ep, hist["va"], "s--", label="val")
ax2.set_xlabel("epoca"); ax2.set_ylabel("exactitud"); ax2.legend(); ax2.set_title("Exactitud")
plt.tight_layout()
plt.show()

# ======================================================================
# Mapa de saliencia (preludio del ejercicio 9.6)
# ======================================================================

modelo.eval()
imagenes, etiquetas = next(iter(loader_test))
fig, axes = plt.subplots(2, 5, figsize=(9, 4))
for k in range(5):
    img = imagenes[k:k+1].to(dispositivo).clone().requires_grad_(True)
    salida = modelo(img)
    clase = salida.argmax(1)
    modelo.zero_grad()
    salida[0, clase].backward()
    sal = img.grad.abs().squeeze().cpu().numpy()
    axes[0, k].imshow(imagenes[k].squeeze(), cmap="gray")
    axes[0, k].set_title(f"y={etiquetas[k].item()} pred={clase.item()}")
    axes[0, k].axis("off")
    axes[1, k].imshow(sal, cmap="hot")
    axes[1, k].axis("off")
axes[0, 0].set_ylabel("entrada")
axes[1, 0].set_ylabel("saliencia")
fig.suptitle("Mapas de saliencia: gradiente de la prediccion respecto a la entrada")
plt.tight_layout()
plt.show()

# ======================================================================
# 2. Una arquitectura más profunda sobre CIFAR-10
# ======================================================================

import torchvision.transforms as T

# Transformaciones de entrenamiento (con aumento)
train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225])   # estadisticas ImageNet
])
# Transformacion de evaluacion (sin aumento, solo normalizacion)
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225])
])

cifar_train = datasets.CIFAR10("./data", train=True,  download=True,
                               transform=train_transform)
cifar_test  = datasets.CIFAR10("./data", train=False, download=True,
                               transform=eval_transform)

loader_ctrain = DataLoader(cifar_train, batch_size=128, shuffle=True,  num_workers=2)
loader_ctest  = DataLoader(cifar_test,  batch_size=256, shuffle=False, num_workers=2)

clases_cifar = ["avion", "auto", "pajaro", "gato", "ciervo",
                "perro", "rana", "caballo", "barco", "camion"]
print(f"train: {len(cifar_train)}  test: {len(cifar_test)}")

class CNNConBN(nn.Module):
    def __init__(self, num_clases=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))   # average pooling adaptativo
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_clases)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

modelo_cifar = CNNConBN(num_clases=10).to(dispositivo)
print(f"Parametros: {sum(p.numel() for p in modelo_cifar.parameters()):,}")

criterio_c    = nn.CrossEntropyLoss()
optimizador_c = torch.optim.Adam(modelo_cifar.parameters(), lr=1e-3,
                                 weight_decay=1e-4)

# En CPU conviene reducir num_epocas; en GPU 15-20 epocas dan ~75-80% en test.
num_epocas_c = 15
hist_c = {"tl": [], "ta": [], "vl": [], "va": []}

t0 = time.time()
for epoca in range(1, num_epocas_c + 1):
    tl, ta = entrenar_epoca(modelo_cifar, loader_ctrain, optimizador_c,
                            criterio_c, dispositivo)
    vl, va = evaluar(modelo_cifar, loader_ctest, criterio_c, dispositivo)
    hist_c["tl"].append(tl); hist_c["ta"].append(ta)
    hist_c["vl"].append(vl); hist_c["va"].append(va)
    print(f"Epoca {epoca:2d}: train loss={tl:.4f} acc={ta:.4f} | "
          f"test loss={vl:.4f} acc={va:.4f}")
print(f"\nTiempo total: {time.time() - t0:.1f} s")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
ep = range(1, num_epocas_c + 1)
ax1.plot(ep, hist_c["tl"], "o-", label="train")
ax1.plot(ep, hist_c["vl"], "s--", label="test")
ax1.set_xlabel("epoca"); ax1.set_ylabel("perdida"); ax1.legend(); ax1.set_title("Perdida CIFAR-10")
ax2.plot(ep, hist_c["ta"], "o-", label="train")
ax2.plot(ep, hist_c["va"], "s--", label="test")
ax2.set_xlabel("epoca"); ax2.set_ylabel("exactitud"); ax2.legend(); ax2.set_title("Exactitud CIFAR-10")
plt.tight_layout()
plt.show()

# ======================================================================
# 3. Transfer learning con ResNet-18 sobre microscopía de materiales
# ======================================================================

# ----- Generador sintetico de microestructuras (sustituible por datos reales) -----
# Cuatro clases: bandas horizontales, bandas verticales, granos circulares, textura fina.
from torch.utils.data import TensorDataset

def generar_microestructura(clase, size=96, rng=None):
    rng = rng or np.random.default_rng()
    img = rng.normal(0.5, 0.05, (size, size))
    if clase == 0:        # bandas horizontales (laminado)
        for r in range(0, size, 8):
            img[r:r+4, :] += 0.35
    elif clase == 1:      # bandas verticales
        for c in range(0, size, 8):
            img[:, c:c+4] += 0.35
    elif clase == 2:      # granos equiaxiales (circulos)
        yy, xx = np.ogrid[:size, :size]
        for _ in range(12):
            cy, cx = rng.integers(8, size-8, 2)
            rad = rng.integers(5, 11)
            img[(yy-cy)**2 + (xx-cx)**2 < rad**2] += 0.4
    else:                 # textura fina (martensita)
        img += rng.normal(0, 0.22, (size, size))
    img = np.clip(img, 0, 1)
    return img.astype(np.float32)

def construir_dataset(n_por_clase, semilla):
    rng = np.random.default_rng(semilla)
    X, y = [], []
    for clase in range(4):
        for _ in range(n_por_clase):
            g = generar_microestructura(clase, rng=rng)         # (H, W) gris
            rgb = np.repeat(g[None, :, :], 3, axis=0)            # -> 3 canales
            X.append(rgb); y.append(clase)
    X = torch.tensor(np.stack(X))
    y = torch.tensor(y)
    return X, y

# 200 imagenes: 50 por clase. Reparto 70/15/15 train/val/test.
X_all, y_all = construir_dataset(n_por_clase=50, semilla=42)
print("conjunto:", X_all.shape, "etiquetas:", torch.bincount(y_all).tolist())

# Normalizacion ImageNet (ResNet-18 la espera)
media = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
desv  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ResNet-18 espera entradas de 224x224
X_all = torch.nn.functional.interpolate(X_all, size=224, mode="bilinear",
                                         align_corners=False)
X_all = (X_all - media) / desv

# Particion estratificada simple
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(X_all), generator=g)
n = len(X_all)
i_tr, i_va = int(0.70*n), int(0.85*n)
idx_tr, idx_va, idx_te = perm[:i_tr], perm[i_tr:i_va], perm[i_va:]

micro_train = TensorDataset(X_all[idx_tr], y_all[idx_tr])
micro_val   = TensorDataset(X_all[idx_va], y_all[idx_va])
micro_test  = TensorDataset(X_all[idx_te], y_all[idx_te])

loader_mtr = DataLoader(micro_train, batch_size=32, shuffle=True)
loader_mva = DataLoader(micro_val,   batch_size=32, shuffle=False)
loader_mte = DataLoader(micro_test,  batch_size=32, shuffle=False)
print(f"train: {len(micro_train)}  val: {len(micro_val)}  test: {len(micro_test)}")

# Visualizacion de una muestra de cada clase (deshaciendo la normalizacion)
nombres = ["bandas H", "bandas V", "granos", "textura fina"]
fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
for clase in range(4):
    g_img = generar_microestructura(clase, rng=np.random.default_rng(clase))
    axes[clase].imshow(g_img, cmap="gray")
    axes[clase].set_title(nombres[clase]); axes[clase].axis("off")
plt.tight_layout()
plt.show()

# ======================================================================
# Carga de ResNet-18 y reemplazo de la cabeza
# ======================================================================

from torchvision.models import ResNet18_Weights

# 1. Cargar ResNet-18 preentrenada en ImageNet
pesos  = ResNet18_Weights.DEFAULT
modelo_tl = models.resnet18(weights=pesos)

# 2. Congelar TODO el backbone
for param in modelo_tl.parameters():
    param.requires_grad = False

# 3. Reemplazar la cabeza final
num_clases = 4        # 4 tipos de microestructura
n_entradas = modelo_tl.fc.in_features   # 512 para ResNet-18
modelo_tl.fc = nn.Linear(n_entradas, num_clases)
modelo_tl = modelo_tl.to(dispositivo)

# Solo la cabeza es entrenable
entrenables = sum(p.numel() for p in modelo_tl.parameters() if p.requires_grad)
total       = sum(p.numel() for p in modelo_tl.parameters())
print(f"Entrenables: {entrenables:,} / {total:,}  ({100*entrenables/total:.2f}%)")

# ======================================================================
# Fase 1 — entrenar solo la cabeza
# ======================================================================

criterio_tl   = nn.CrossEntropyLoss()
optimizador1  = torch.optim.Adam(modelo_tl.fc.parameters(), lr=1e-3)

hist_f1 = {"tl": [], "ta": [], "vl": [], "va": []}
for epoca in range(1, 11):
    tl, ta = entrenar_epoca(modelo_tl, loader_mtr, optimizador1, criterio_tl, dispositivo)
    vl, va = evaluar(modelo_tl, loader_mva, criterio_tl, dispositivo)
    hist_f1["tl"].append(tl); hist_f1["ta"].append(ta)
    hist_f1["vl"].append(vl); hist_f1["va"].append(va)
    print(f"[Fase 1] Epoca {epoca:2d}: train acc={ta:.3f} | val acc={va:.3f}")

# ======================================================================
# Fase 2 — *fine-tuning* del último bloque
# ======================================================================

# Descongelar el ultimo bloque del backbone
for param in modelo_tl.layer4.parameters():
    param.requires_grad = True

optimizador2 = torch.optim.Adam([
    {"params": modelo_tl.layer4.parameters(), "lr": 1e-4},  # lr pequena
    {"params": modelo_tl.fc.parameters(),     "lr": 1e-3},
])

entrenables = sum(p.numel() for p in modelo_tl.parameters() if p.requires_grad)
print(f"Entrenables tras descongelar layer4: {entrenables:,} / {total:,}")

hist_f2 = {"tl": [], "ta": [], "vl": [], "va": []}
for epoca in range(1, 11):
    tl, ta = entrenar_epoca(modelo_tl, loader_mtr, optimizador2, criterio_tl, dispositivo)
    vl, va = evaluar(modelo_tl, loader_mva, criterio_tl, dispositivo)
    hist_f2["tl"].append(tl); hist_f2["ta"].append(ta)
    hist_f2["vl"].append(vl); hist_f2["va"].append(va)
    print(f"[Fase 2] Epoca {epoca:2d}: train acc={ta:.3f} | val acc={va:.3f}")

test_loss, test_acc = evaluar(modelo_tl, loader_mte, criterio_tl, dispositivo)
print(f"\nExactitud en test (transfer learning): {test_acc:.3f}")

# Curva de aprendizaje de las dos fases concatenadas
va_total = hist_f1["va"] + hist_f2["va"]
ta_total = hist_f1["ta"] + hist_f2["ta"]
ep = range(1, len(va_total) + 1)
plt.figure(figsize=(6, 3.5))
plt.plot(ep, ta_total, "o-", label="train")
plt.plot(ep, va_total, "s--", label="val")
plt.axvline(10.5, color="gray", ls=":", lw=1)
plt.text(5.5, min(va_total), "Fase 1", ha="center")
plt.text(15.5, min(va_total), "Fase 2 (fine-tuning)", ha="center")
plt.xlabel("epoca"); plt.ylabel("exactitud"); plt.legend()
plt.title("Transfer learning: cabeza + fine-tuning")
plt.tight_layout()
plt.show()

# ======================================================================
# Apéndice — CNN 1D para clasificación de espectros
# ======================================================================

class CNN1D(nn.Module):
    "CNN para clasificacion de espectros 1D de longitud L."
    def __init__(self, L, num_clases):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),    # L/2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),    # L/4
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)  # -> 128 x 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_clases)
        )
    def forward(self, x):
        # x: [batch, 1, L] (1 canal, longitud L)
        return self.classifier(self.features(x))

# Uso: espectros de longitud 1024, 5 clases
modelo_1d = CNN1D(L=1024, num_clases=5).to(dispositivo)
print(f"Parametros: {sum(p.numel() for p in modelo_1d.parameters()):,}")

# ----- Espectros sinteticos: 5 clases con picos gaussianos en posiciones distintas -----
def generar_espectros(n_por_clase, L=1024, semilla=0):
    rng = np.random.default_rng(semilla)
    centros_por_clase = [
        [120, 400], [250, 600], [80, 512, 900], [330, 720], [150, 480, 850]
    ]
    eje = np.arange(L)
    X, y = [], []
    for clase, centros in enumerate(centros_por_clase):
        for _ in range(n_por_clase):
            esp = rng.normal(0.0, 0.03, L)
            for c in centros:
                c_j = c + rng.integers(-8, 9)
                amp = rng.uniform(0.7, 1.0)
                anch = rng.uniform(6, 12)
                esp += amp * np.exp(-(eje - c_j)**2 / (2 * anch**2))
            X.append(esp.astype(np.float32)); y.append(clase)
    X = torch.tensor(np.stack(X)).unsqueeze(1)   # [N, 1, L]
    y = torch.tensor(y)
    return X, y

Xe, ye = generar_espectros(n_por_clase=200, L=1024, semilla=1)
print("espectros:", Xe.shape, "clases:", torch.bincount(ye).tolist())

g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(Xe), generator=g)
corte = int(0.8 * len(Xe))
tr, te = perm[:corte], perm[corte:]
esp_train = TensorDataset(Xe[tr], ye[tr])
esp_test  = TensorDataset(Xe[te], ye[te])
loader_etr = DataLoader(esp_train, batch_size=64, shuffle=True)
loader_ete = DataLoader(esp_test,  batch_size=128, shuffle=False)

# Un espectro de ejemplo
plt.figure(figsize=(7, 2.5))
plt.plot(Xe[0, 0].numpy())
plt.title(f"Espectro de ejemplo (clase {ye[0].item()})")
plt.xlabel("indice"); plt.ylabel("intensidad")
plt.tight_layout(); plt.show()

criterio_e   = nn.CrossEntropyLoss()
optimizador_e = torch.optim.Adam(modelo_1d.parameters(), lr=1e-3)

for epoca in range(1, 11):
    tl, ta = entrenar_epoca(modelo_1d, loader_etr, optimizador_e, criterio_e, dispositivo)
    print(f"Epoca {epoca:2d}: train loss={tl:.4f} acc={ta:.4f}")

test_loss, test_acc = evaluar(modelo_1d, loader_ete, criterio_e, dispositivo)
print(f"\nExactitud en test (CNN 1D): {test_acc:.4f}")

# ======================================================================
# Visualización de los filtros de la primera capa
# ======================================================================

filtros = modelo_1d.features[0].weight.detach().cpu().numpy()  # [32, 1, 7]
fig, axes = plt.subplots(4, 8, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.plot(filtros[i, 0])
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Filtros aprendidos de la primera capa Conv1d")
plt.tight_layout()
plt.show()

# ======================================================================
# Cierre y conexión con los ejercicios
# ======================================================================

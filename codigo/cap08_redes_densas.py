# -*- coding: utf-8 -*-
"""
Capítulo 8 · Redes neuronales densas

Script extraído del notebook cap08_redes_densas.ipynb
Libro: "Inteligencia artificial para estudiantes de ciencias"
Jorge Bravo Abad — Ediciones Pirámide, 2026
"""


# ======================================================================
# Capítulo 8 — Redes neuronales densas
# ======================================================================

# Instalación de dependencias (descomenta en Colab/Kaggle si hace falta)
# !pip install torch torchvision scikit-learn matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Reproducibilidad
SEMILLA = 0
torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)

# PyTorch usa la GPU si esta disponible; en caso contrario, la CPU
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'

print('PyTorch:', torch.__version__)
print('Dispositivo:', dispositivo)

# ======================================================================
# 1. Diferenciación automática con `autograd`
# ======================================================================

# Tensor hoja: sus gradientes seran acumulados por .backward()
x = torch.tensor([2.0, 3.0], requires_grad=True)

# PyTorch registra cada operacion en el grafo computacional
z = x[0]**3 + 2*x[1]**2   # z = 8 + 18 = 26

# Una unica llamada calcula todos los gradientes
z.backward()

# dz/dx0 = 3*x0^2 = 12,   dz/dx1 = 4*x1 = 12
print('z       =', z.item())
print('x.grad  =', x.grad)        # tensor([12., 12.])

# ======================================================================
# 2. Definición de la arquitectura con `nn.Module`
# ======================================================================

class RedDensa(nn.Module):
    """
    Red completamente conectada con arquitectura configurable.

    Parametros
    ----------
    dims : list[int]
        Tamanios de cada capa, incluyendo entrada y salida.
        Ejemplo: [784, 256, 128, 10] define una red con dos capas
        ocultas de 256 y 128 neuronas para clasificacion de MNIST.
    activacion : nn.Module
        Clase de funcion de activacion para las capas ocultas.
        Por defecto, ReLU.
    """
    def __init__(self, dims, activacion=nn.ReLU):
        super().__init__()
        capas = []
        for entrada, salida in zip(dims[:-2], dims[1:-1]):
            capas.append(nn.Linear(entrada, salida))
            capas.append(activacion())
        # La capa de salida no lleva activacion: devolvemos logits
        # para usar con nn.CrossEntropyLoss (mas estable numericamente)
        capas.append(nn.Linear(dims[-2], dims[-1]))
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)

# Red 784 -> 256 -> 128 -> 10 para clasificacion de MNIST
modelo = RedDensa([784, 256, 128, 10])
n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print(modelo)
print(f'\nParametros entrenables: {n_params:,}')   # 235 146

def num_parametros(dims):
    return sum(dims[l]*(dims[l-1] + 1) for l in range(1, len(dims)))

print('Formula del capitulo:', f'{num_parametros([784, 256, 128, 10]):,}')
print('Conteo de PyTorch  :', f'{n_params:,}')
assert num_parametros([784, 256, 128, 10]) == n_params

# ======================================================================
# 3. Reconocimiento de dígitos manuscritos (MNIST)
# ======================================================================

from torchvision import datasets, transforms

# Transformacion: imagen PIL -> tensor float en [0, 1]
transformacion = transforms.ToTensor()

# Descarga (la primera vez) y carga de MNIST.
# Si el entorno no tiene acceso a internet, mas abajo hay un respaldo sintetico.
USAR_MNIST_REAL = True

if USAR_MNIST_REAL:
    try:
        ds_train = datasets.MNIST(root='./datos', train=True,
                                  download=True, transform=transformacion)
        ds_test  = datasets.MNIST(root='./datos', train=False,
                                  download=True, transform=transformacion)
        print('MNIST cargado:', len(ds_train), 'train,', len(ds_test), 'test')
    except Exception as e:
        print('No se pudo descargar MNIST:', type(e).__name__)
        print('Activando respaldo sintetico (misma forma 28x28, 10 clases).')
        USAR_MNIST_REAL = False

if not USAR_MNIST_REAL:
    # Respaldo offline: digitos sinteticos con la misma forma que MNIST.
    # Permite ejecutar el cuaderno completo sin conexion.
    rng = np.random.default_rng(7)
    plantillas = rng.normal(size=(10, 28, 28)).astype('float32') * 2.0
    def _sint(n, seed):
        r = np.random.default_rng(seed)
        y = r.integers(0, 10, n)
        X = plantillas[y] + r.normal(size=(n, 28, 28)).astype('float32')
        X = (X - X.min()) / (X.max() - X.min())
        return torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y.astype('int64'))
    Xtr, ytr = _sint(6000, 1)
    Xte, yte = _sint(1000, 2)
    ds_train = TensorDataset(Xtr, ytr)
    ds_test  = TensorDataset(Xte, yte)
    print('Respaldo sintetico:', len(ds_train), 'train,', len(ds_test), 'test')

fig, ejes = plt.subplots(2, 5, figsize=(8, 3.4))
for i, eje in enumerate(ejes.ravel()):
    imagen, etiqueta = ds_train[i]
    eje.imshow(imagen.squeeze().numpy(), cmap='gray')
    eje.set_title(f'y = {etiqueta}', fontsize=10)
    eje.axis('off')
plt.tight_layout()
plt.show()

from torch.utils.data import random_split

n_val = 10000 if len(ds_train) >= 60000 else max(1, len(ds_train)//6)
n_train = len(ds_train) - n_val
ds_tr, ds_val = random_split(ds_train, [n_train, n_val],
                             generator=torch.Generator().manual_seed(SEMILLA))

loader_train = DataLoader(ds_tr,  batch_size=128, shuffle=True)
loader_val   = DataLoader(ds_val, batch_size=512, shuffle=False)
loader_test  = DataLoader(ds_test, batch_size=512, shuffle=False)

print('Train:', len(ds_tr), '| Val:', len(ds_val), '| Test:', len(ds_test))

# ======================================================================
# El bucle de entrenamiento
# ======================================================================

def entrenar(modelo, loader_train, loader_val,
             num_epocas=15, lr=1e-3, weight_decay=0.0,
             dispositivo='cpu', verbose=True):
    modelo = modelo.to(dispositivo)
    criterio    = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr,
                             weight_decay=weight_decay)

    hist = {'loss_train': [], 'loss_val': [],
            'acc_train': [], 'acc_val': []}

    for epoca in range(1, num_epocas + 1):

        # -------- Fase de entrenamiento --------
        modelo.train()
        perdida_acum, correctos, total = 0.0, 0, 0
        for X_batch, y_batch in loader_train:
            X_batch = X_batch.to(dispositivo)
            y_batch = y_batch.to(dispositivo)
            X_batch = X_batch.view(X_batch.size(0), -1)

            pred    = modelo(X_batch)              # 1. Paso adelante
            perdida = criterio(pred, y_batch)      # 2. Perdida

            optimizador.zero_grad()                # 3a. Limpiar gradientes
            perdida.backward()                     # 3b. Retropropagacion
            optimizador.step()                     # 4. Actualizar pesos

            perdida_acum += perdida.item() * len(y_batch)
            correctos    += (pred.argmax(1) == y_batch).sum().item()
            total        += len(y_batch)

        # -------- Fase de validacion --------
        modelo.eval()
        val_perdida, val_ok, val_tot = 0.0, 0, 0
        with torch.no_grad():
            for X, y in loader_val:
                X = X.to(dispositivo).view(X.size(0), -1)
                y = y.to(dispositivo)
                pred_val = modelo(X)
                val_perdida += criterio(pred_val, y).item() * len(y)
                val_ok      += (pred_val.argmax(1) == y).sum().item()
                val_tot     += len(y)

        hist['loss_train'].append(perdida_acum / total)
        hist['loss_val'].append(val_perdida / val_tot)
        hist['acc_train'].append(correctos / total)
        hist['acc_val'].append(val_ok / val_tot)

        if verbose:
            print(f"Epoca {epoca:>3} | "
                  f"Loss: {hist['loss_train'][-1]:.4f}  "
                  f"Train acc: {hist['acc_train'][-1]:.3f}  "
                  f"Val acc: {hist['acc_val'][-1]:.3f}")

    return hist

modelo = RedDensa([784, 256, 128, 10])
hist = entrenar(modelo, loader_train, loader_val,
                num_epocas=15, lr=1e-3, dispositivo=dispositivo)

# ======================================================================
# Curvas de aprendizaje
# ======================================================================

epocas = range(1, len(hist['loss_train']) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

ax1.plot(epocas, hist['loss_train'], '-o', ms=3, label='entrenamiento')
ax1.plot(epocas, hist['loss_val'],   '-s', ms=3, label='validacion')
ax1.set_xlabel('epoca'); ax1.set_ylabel('perdida'); ax1.legend()
ax1.set_title('Perdida')

ax2.plot(epocas, hist['acc_train'], '-o', ms=3, label='entrenamiento')
ax2.plot(epocas, hist['acc_val'],   '-s', ms=3, label='validacion')
ax2.set_xlabel('epoca'); ax2.set_ylabel('exactitud'); ax2.legend()
ax2.set_title('Exactitud')

plt.tight_layout()
plt.show()

# ======================================================================
# Evaluación sobre el conjunto de test
# ======================================================================

modelo.eval()
test_ok, test_tot = 0, 0
with torch.no_grad():
    for X, y in loader_test:
        X = X.to(dispositivo).view(X.size(0), -1)
        y = y.to(dispositivo)
        test_ok  += (modelo(X).argmax(1) == y).sum().item()
        test_tot += len(y)

print(f'Exactitud sobre test: {test_ok/test_tot:.4f}')

# Visualizacion de predicciones sobre ejemplos de test
modelo.eval()
fig, ejes = plt.subplots(2, 5, figsize=(9, 4))
with torch.no_grad():
    for i, eje in enumerate(ejes.ravel()):
        imagen, etiqueta = ds_test[i]
        logits = modelo(imagen.view(1, -1).to(dispositivo))
        pred = logits.argmax(1).item()
        eje.imshow(imagen.squeeze().numpy(), cmap='gray')
        color = 'green' if pred == etiqueta else 'red'
        eje.set_title(f'pred={pred} (y={etiqueta})', fontsize=9, color=color)
        eje.axis('off')
plt.tight_layout()
plt.show()

# ======================================================================
# 4. Regularización y estabilización
# ======================================================================

# ======================================================================
# 4.1 Parada temprana (*early stopping*)
# ======================================================================

import copy

def entrenar_con_parada_temprana(modelo, loader_train, loader_val,
                                 num_epocas=60, lr=1e-3, paciencia=5,
                                 dispositivo='cpu'):
    modelo = modelo.to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr)

    mejor_val = float('inf')
    mejor_estado = None
    sin_mejora = 0
    epoca_parada = num_epocas
    hist = {'loss_train': [], 'loss_val': []}

    for epoca in range(1, num_epocas + 1):
        modelo.train()
        acum, total = 0.0, 0
        for X, y in loader_train:
            X = X.to(dispositivo).view(X.size(0), -1)
            y = y.to(dispositivo)
            optimizador.zero_grad()
            perdida = criterio(modelo(X), y)
            perdida.backward()
            optimizador.step()
            acum += perdida.item() * len(y); total += len(y)

        modelo.eval()
        vperd, vtot = 0.0, 0
        with torch.no_grad():
            for X, y in loader_val:
                X = X.to(dispositivo).view(X.size(0), -1)
                y = y.to(dispositivo)
                vperd += criterio(modelo(X), y).item() * len(y); vtot += len(y)

        hist['loss_train'].append(acum / total)
        hist['loss_val'].append(vperd / vtot)

        if hist['loss_val'][-1] < mejor_val - 1e-4:
            mejor_val = hist['loss_val'][-1]
            mejor_estado = copy.deepcopy(modelo.state_dict())
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                epoca_parada = epoca
                print(f'Parada temprana en la epoca {epoca} '
                      f'(mejor val_loss = {mejor_val:.4f})')
                break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return hist, epoca_parada

modelo_es = RedDensa([784, 256, 128, 10])
hist_es, epoca_parada = entrenar_con_parada_temprana(
    modelo_es, loader_train, loader_val,
    num_epocas=60, lr=1e-3, paciencia=5, dispositivo=dispositivo)

ep = range(1, len(hist_es['loss_train']) + 1)
plt.figure(figsize=(6, 3.8))
plt.plot(ep, hist_es['loss_train'], '-o', ms=3, label='entrenamiento')
plt.plot(ep, hist_es['loss_val'],   '-s', ms=3, label='validacion')
plt.axvline(epoca_parada, ls='--', color='gray', label='parada temprana')
plt.xlabel('epoca'); plt.ylabel('perdida'); plt.legend()
plt.title('Parada temprana')
plt.tight_layout(); plt.show()

# ======================================================================
# 4.2 *Weight decay*, *dropout* y *batch normalization*
# ======================================================================

class RedDensaRegularizada(nn.Module):
    """Red completamente conectada con regularizacion integrada."""

    def __init__(self, dims, p_dropout=0.3):
        super().__init__()
        capas = []
        for entrada, salida in zip(dims[:-2], dims[1:-1]):
            capas.append(nn.Linear(entrada, salida))
            capas.append(nn.BatchNorm1d(salida))   # estandariza la capa
            capas.append(nn.ReLU())
            capas.append(nn.Dropout(p=p_dropout))  # desconexion aleatoria
        capas.append(nn.Linear(dims[-2], dims[-1]))
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)

modelo_reg = RedDensaRegularizada([784, 512, 256, 10], p_dropout=0.3)
hist_reg = entrenar(modelo_reg, loader_train, loader_val,
                    num_epocas=15, lr=1e-3, weight_decay=1e-4,
                    dispositivo=dispositivo)

# ======================================================================
# 5. El mismo pipeline para propiedades moleculares
# ======================================================================

def dataset_molecular(n, seed=0):
    """Genera n moleculas representadas por 5 descriptores fisicoquimicos
    y una propiedad escalar relacionada con ellos de forma no lineal."""
    rng = np.random.default_rng(seed)
    n_atomos  = rng.integers(3, 20, n).astype('float32')
    n_enlaces = (n_atomos * rng.uniform(0.8, 1.4, n)).astype('float32')
    aromatico = rng.integers(0, 2, n).astype('float32')        # 0 / 1
    polaridad = rng.uniform(0, 5, n).astype('float32')
    masa      = (n_atomos * rng.uniform(6, 16, n)).astype('float32')

    X = np.stack([n_atomos, n_enlaces, aromatico, polaridad, masa], axis=1)

    # Propiedad objetivo: combinacion no lineal de los descriptores + ruido
    y = (-1.2*n_atomos - 0.5*n_enlaces + 3.0*aromatico*polaridad
         + 0.02*masa**1.5 + rng.normal(0, 2, n)).astype('float32')
    return X.astype('float32'), y.reshape(-1, 1)

X_mol, y_mol = dataset_molecular(2000, seed=0)
nombres = ['n_atomos', 'n_enlaces', 'aromatico', 'polaridad', 'masa']
print('Forma de X:', X_mol.shape, '| Forma de y:', y_mol.shape)
print('Descriptores:', nombres)

# Normalizacion z-score (ajustada SOLO con el train para no filtrar informacion)
n_tr = 1600
X_tr_raw, X_te_raw = X_mol[:n_tr], X_mol[n_tr:]
y_tr_raw, y_te_raw = y_mol[:n_tr], y_mol[n_tr:]

mu_X, sd_X = X_tr_raw.mean(0), X_tr_raw.std(0)
mu_y, sd_y = y_tr_raw.mean(), y_tr_raw.std()

X_tr = (X_tr_raw - mu_X) / sd_X
X_te = (X_te_raw - mu_X) / sd_X
y_tr = (y_tr_raw - mu_y) / sd_y
y_te = (y_te_raw - mu_y) / sd_y

X_tr_t = torch.from_numpy(X_tr); y_tr_t = torch.from_numpy(y_tr)
X_te_t = torch.from_numpy(X_te); y_te_t = torch.from_numpy(y_te)

loader_mol = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=32, shuffle=True)
print('Train:', len(X_tr_t), '| Test:', len(X_te_t))

def entrenar_regresion(modelo, loader, num_epocas=200, lr=1e-2,
                       dispositivo='cpu'):
    modelo = modelo.to(dispositivo)
    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr)
    historia = []
    for epoca in range(1, num_epocas + 1):
        modelo.train()
        acum, total = 0.0, 0
        for Xb, yb in loader:
            Xb = Xb.to(dispositivo); yb = yb.to(dispositivo)
            optimizador.zero_grad()
            perdida = criterio(modelo(Xb), yb)
            perdida.backward()
            optimizador.step()
            acum += perdida.item() * len(yb); total += len(yb)
        historia.append(acum / total)
    return historia

red_mol = RedDensa([5, 64, 64, 1])
historia_mol = entrenar_regresion(red_mol, loader_mol,
                                  num_epocas=200, lr=1e-2,
                                  dispositivo=dispositivo)
print(f'MSE final de entrenamiento (normalizado): {historia_mol[-1]:.4f}')

# Evaluacion: MSE y coeficiente de determinacion R^2 sobre test
red_mol.eval()
with torch.no_grad():
    pred_te = red_mol(X_te_t.to(dispositivo)).cpu()

mse = nn.functional.mse_loss(pred_te, y_te_t).item()
ss_res = ((y_te_t - pred_te)**2).sum().item()
ss_tot = ((y_te_t - y_te_t.mean())**2).sum().item()
r2 = 1 - ss_res / ss_tot

print(f'MSE de test (normalizado): {mse:.4f}')
print(f'R^2  de test             : {r2:.4f}')

# Volvemos a la escala original del objetivo
pred_orig = pred_te.numpy().ravel() * sd_y + mu_y
real_orig = y_te_raw.ravel()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(historia_mol)
ax1.set_xlabel('epoca'); ax1.set_ylabel('MSE (entrenamiento)')
ax1.set_title('Curva de aprendizaje')

lims = [real_orig.min(), real_orig.max()]
ax2.scatter(real_orig, pred_orig, s=12, alpha=0.5)
ax2.plot(lims, lims, '--', color='gray')
ax2.set_xlabel('propiedad real'); ax2.set_ylabel('propiedad predicha')
ax2.set_title(f'Predicho vs. real  (R^2 = {r2:.3f})')

plt.tight_layout(); plt.show()

# ======================================================================
# Resumen
# ======================================================================

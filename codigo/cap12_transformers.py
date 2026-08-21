# -*- coding: utf-8 -*-
"""
Capítulo 12 · Transformers y mecanismo de atención

Script extraído del notebook cap12_transformers.ipynb
Libro: "Inteligencia artificial para estudiantes de ciencias"
Jorge Bravo Abad — Ediciones Pirámide, 2026
"""


# ======================================================================
# Capítulo 12 — Transformers y mecanismo de atención
# ======================================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Reproducibilidad
SEMILLA = 0
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | dispositivo: {dispositivo}")

# ======================================================================
# 1. Autoatención escalada desde cero
# ======================================================================

class AutoAtencionEscalada(nn.Module):
    """Atencion escalada de producto escalar, implementada desde cero."""

    def __init__(self, d_modelo, d_k):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Linear(d_modelo, d_k, bias=False)
        self.W_K = nn.Linear(d_modelo, d_k, bias=False)
        self.W_V = nn.Linear(d_modelo, d_k, bias=False)

    def forward(self, x, mascara=None):
        # x: [batch, n_tokens, d_modelo]
        Q = self.W_Q(x)   # [batch, n, d_k]
        K = self.W_K(x)
        V = self.W_V(x)

        # Producto escalar escalado
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Mascara causal opcional (para modelos autorregresivos)
        if mascara is not None:
            scores = scores.masked_fill(mascara == 0, float("-inf"))

        pesos = F.softmax(scores, dim=-1)   # [batch, n, n]
        return torch.matmul(pesos, V), pesos


# Ejemplo: secuencia de 10 tokens, dimension 64, claves de dim 32
x = torch.randn(2, 10, 64)   # batch=2
modulo = AutoAtencionEscalada(d_modelo=64, d_k=32)
salida, pesos = modulo(x)
print(f"Salida:             {tuple(salida.shape)}")   # [2, 10, 32]
print(f"Pesos de atencion:  {tuple(pesos.shape)}")    # [2, 10, 10]
print(f"Cada fila de pesos suma 1: {torch.allclose(pesos.sum(-1), torch.ones(2, 10))}")

# ======================================================================
# 1.1 La matriz de atención refleja la estructura de la secuencia
# ======================================================================

torch.manual_seed(1)

n_tokens, d_modelo, periodo = 20, 64, 5
patrones_base = torch.randn(periodo, d_modelo)
# La posicion t comparte patron con t +/- multiplos de 5 (+ ruido pequeno)
secuencia = torch.stack([patrones_base[t % periodo] for t in range(n_tokens)])
secuencia = (secuencia + 0.1 * torch.randn(n_tokens, d_modelo)).unsqueeze(0)

mod_vis = AutoAtencionEscalada(d_modelo=d_modelo, d_k=d_modelo)
# Inicializamos W_Q = W_K = identidad para que la atencion mida similitud directa
with torch.no_grad():
    mod_vis.W_Q.weight.copy_(torch.eye(d_modelo))
    mod_vis.W_K.weight.copy_(torch.eye(d_modelo))
_, pesos_vis = mod_vis(secuencia)

fig, ax = plt.subplots(figsize=(5.0, 4.2))
im = ax.imshow(pesos_vis[0].detach().numpy(), cmap="viridis")
ax.set_xlabel("clave (posicion atendida)")
ax.set_ylabel("consulta (posicion)")
ax.set_title("Matriz de atencion: secuencia con periodo 5")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="peso")
plt.tight_layout(); plt.show()

print("Las bandas diagonales separadas 5 posiciones reflejan la periodicidad.")

# ======================================================================
# 1.2 El factor de escala $1/\sqrt{d_k}$ y la entropía del softmax
# ======================================================================

def entropia_media(pesos):
    # Entropia de Shannon promediada sobre filas y batch (en nats)
    p = pesos.clamp_min(1e-12)
    return (-(p * p.log()).sum(-1)).mean().item()

torch.manual_seed(2)
ds_k = [4, 16, 64, 256]
ent_con, ent_sin = [], []

for d_k in ds_k:
    x_e = torch.randn(1, 30, d_k)
    Q = x_e; K = x_e  # usamos la entrada directamente como Q y K
    scores = torch.matmul(Q, K.transpose(-2, -1))
    ent_sin.append(entropia_media(F.softmax(scores, dim=-1)))
    ent_con.append(entropia_media(F.softmax(scores / math.sqrt(d_k), dim=-1)))

ent_max = math.log(30)  # entropia de la distribucion uniforme sobre 30 tokens
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(ds_k, ent_con, "o-", label="con escala $1/\\sqrt{d_k}$")
ax.plot(ds_k, ent_sin, "s--", label="sin escala")
ax.axhline(ent_max, color="gray", ls=":", lw=1, label="uniforme (max)")
ax.set_xscale("log", base=2)
ax.set_xlabel("$d_k$"); ax.set_ylabel("entropia media (nats)")
ax.set_title("Efecto del factor de escala sobre la atencion")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

for d_k, ec, es in zip(ds_k, ent_con, ent_sin):
    print(f"d_k={d_k:>3}:  con escala={ec:.3f}   sin escala={es:.3f}")

# ======================================================================
# 1.3 Máscara causal para atención autorregresiva
# ======================================================================

torch.manual_seed(3)
n = 8
mascara_causal = torch.tril(torch.ones(n, n)).unsqueeze(0)  # [1, n, n]
x_c = torch.randn(1, n, 64)
mod_causal = AutoAtencionEscalada(d_modelo=64, d_k=64)
_, pesos_c = mod_causal(x_c, mascara_causal)

# Verificacion: el token i no atiende a ningun token j > i
triangulo_superior = pesos_c[0].triu(diagonal=1)
print(f"Suma de pesos en el triangulo superior (debe ser ~0): "
      f"{triangulo_superior.sum().item():.2e}")

fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(pesos_c[0].detach().numpy(), cmap="magma")
ax.set_title("Atencion causal (triangular inferior)")
ax.set_xlabel("clave"); ax.set_ylabel("consulta")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="peso")
plt.tight_layout(); plt.show()

# ======================================================================
# 2. Un transformer encoder para clasificación de espectros
# ======================================================================

N_BINS = 200          # numero de bins de m/z
N_CLASES = 4          # clases moleculares sinteticas
N_POR_CLASE = 400     # espectros por clase


def genera_espectros(n_por_clase=N_POR_CLASE, n_bins=N_BINS, n_clases=N_CLASES,
                     semilla=SEMILLA):
    """Genera espectros de masas sinteticos con patrones por clase."""
    rng = np.random.default_rng(semilla)
    # Cada clase tiene un conjunto distinto de picos caracteristicos (m/z)
    centros = [rng.choice(np.arange(15, n_bins - 15), size=3 + c, replace=False)
               for c in range(n_clases)]
    X, y = [], []
    eje = np.arange(n_bins)
    for c in range(n_clases):
        for _ in range(n_por_clase):
            esp = np.zeros(n_bins)
            for mu in centros[c]:
                mu_j = mu + rng.integers(-2, 3)          # ligero corrimiento
                intens = rng.uniform(0.5, 1.0)
                esp += intens * np.exp(-0.5 * ((eje - mu_j) / 1.8) ** 2)
            # Pico espurio aleatorio (interferencia) que dificulta la tarea
            mu_ruido = rng.integers(0, n_bins)
            esp += rng.uniform(0.0, 0.5) * np.exp(-0.5 * ((eje - mu_ruido) / 1.8) ** 2)
            esp += rng.uniform(0, 0.05, n_bins)          # linea base
            esp /= esp.max()
            X.append(esp); y.append(c)
    return np.array(X, dtype=np.float32), np.array(y)


X, y = genera_espectros()
print(f"Espectros: {X.shape}  |  distribucion de clases: {np.bincount(y)}")

# Visualizamos un espectro de cada clase
fig, axes = plt.subplots(N_CLASES, 1, figsize=(6.0, 5.0), sharex=True)
for c, ax in enumerate(axes):
    idx = np.where(y == c)[0][0]
    ax.plot(X[idx], lw=0.9)
    ax.set_ylabel(f"clase {c}", fontsize=8)
axes[-1].set_xlabel("bin de m/z")
axes[0].set_title("Un espectro representativo por clase")
plt.tight_layout(); plt.show()

# ======================================================================
# 2.1 Tokenización del espectro
# ======================================================================

N_NIVELES = 16
IDX_CLS = N_NIVELES + 1          # token [CLS] dedicado
VOCAB = N_NIVELES + 2            # 0=pad, 1..N_NIVELES=niveles, N_NIVELES+1=CLS


def a_tokens(X):
    niveles = np.clip(np.round(X * (N_NIVELES - 1)).astype(int), 0, N_NIVELES - 1)
    niveles = niveles + 1                                  # desplazar: 0 reservado a pad
    cls = np.full((X.shape[0], 1), IDX_CLS, dtype=int)     # [CLS] al inicio
    return np.concatenate([cls, niveles], axis=1)


T = a_tokens(X)
print(f"Secuencias de tokens: {T.shape}  |  tamano de vocabulario: {VOCAB}")
print(f"Primeros tokens del espectro 0: {T[0, :8]}  (el primero es [CLS]={IDX_CLS})")

from sklearn.model_selection import train_test_split

Xtr, Xte, Ttr, Tte, ytr, yte = train_test_split(
    X, T, y, test_size=0.25, random_state=SEMILLA, stratify=y
)
print(f"train: {len(ytr)}   test: {len(yte)}")

# ======================================================================
# 2.2 El codificador transformer
# ======================================================================

class CodificacionPosicional(nn.Module):
    def __init__(self, d_modelo, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_modelo)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_modelo, 2).float()
                        * (-math.log(10000.0) / d_modelo))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d]

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class BloqueTransformerEncoder(nn.Module):
    def __init__(self, d_modelo, n_cabezales, d_ff, dropout=0.1):
        super().__init__()
        self.atencion = nn.MultiheadAttention(d_modelo, n_cabezales,
                                              dropout=dropout,
                                              batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_modelo, d_ff), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_modelo),
        )
        self.norm1 = nn.LayerNorm(d_modelo)
        self.norm2 = nn.LayerNorm(d_modelo)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mascara_clave=None, devolver_pesos=False):
        # Subcapa 1: autoatencion con conexion residual
        aten_out, pesos = self.atencion(x, x, x, key_padding_mask=mascara_clave,
                                        need_weights=devolver_pesos,
                                        average_attn_weights=True)
        x = self.norm1(x + self.drop(aten_out))
        # Subcapa 2: FFN con conexion residual
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x, pesos


class TransformerEncoder(nn.Module):
    """Pila de L bloques transformer encoder para clasificacion."""

    def __init__(self, vocab_size, d_modelo=64, n_cabezales=4,
                 n_capas=2, d_ff=128, n_clases=N_CLASES, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_modelo, padding_idx=0)
        self.pos_enc = CodificacionPosicional(d_modelo, dropout=dropout)
        self.capas = nn.ModuleList([
            BloqueTransformerEncoder(d_modelo, n_cabezales, d_ff, dropout)
            for _ in range(n_capas)
        ])
        self.cabeza = nn.Linear(d_modelo, n_clases)

    def forward(self, tokens, mascara_clave=None, devolver_pesos=False):
        # tokens: [batch, n] con indices de vocabulario
        x = self.pos_enc(self.embedding(tokens))
        pesos_capa1 = None
        for i, capa in enumerate(self.capas):
            x, pesos = capa(x, mascara_clave,
                            devolver_pesos=(devolver_pesos and i == 0))
            if i == 0:
                pesos_capa1 = pesos
        # Clasificacion sobre el token [CLS] (posicion 0)
        logits = self.cabeza(x[:, 0, :])
        return (logits, pesos_capa1) if devolver_pesos else logits


modelo = TransformerEncoder(vocab_size=VOCAB).to(dispositivo)
n_params = sum(p.numel() for p in modelo.parameters())
print(f"Parametros del transformer: {n_params:,}")

# ======================================================================
# 2.3 Entrenamiento
# ======================================================================

def entrena(modelo, T_tr, y_tr, T_te, y_te, epocas=12, lote=64, lr=1e-3):
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.CrossEntropyLoss()
    T_tr = torch.tensor(T_tr, device=dispositivo)
    y_tr = torch.tensor(y_tr, device=dispositivo)
    T_te = torch.tensor(T_te, device=dispositivo)
    y_te = torch.tensor(y_te, device=dispositivo)
    historico = []
    for epoca in range(epocas):
        modelo.train()
        perm = torch.randperm(len(T_tr))
        for i in range(0, len(perm), lote):
            idx = perm[i:i + lote]
            optimizador.zero_grad()
            perdida = criterio(modelo(T_tr[idx]), y_tr[idx])
            perdida.backward()
            optimizador.step()
        modelo.eval()
        with torch.no_grad():
            acc = (modelo(T_te).argmax(1) == y_te).float().mean().item()
        historico.append(acc)
        if (epoca + 1) % 3 == 0 or epoca == 0:
            print(f"epoca {epoca + 1:>2}  acc_test = {acc:.3f}")
    return historico


hist_transformer = entrena(modelo, Ttr, ytr, Tte, yte)
acc_transformer = hist_transformer[-1]

# ======================================================================
# 2.4 Comparación con una CNN 1D (ejercicio 12.2b)
# ======================================================================

class CNN1D(nn.Module):
    def __init__(self, n_bins=N_BINS, n_clases=N_CLASES):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cabeza = nn.Linear(32, n_clases)

    def forward(self, x):
        z = self.extractor(x).squeeze(-1)
        return self.cabeza(z)


torch.manual_seed(SEMILLA)
cnn = CNN1D().to(dispositivo)
optimizador = torch.optim.Adam(cnn.parameters(), lr=1e-3)
criterio = nn.CrossEntropyLoss()

Xtr_t = torch.tensor(Xtr, device=dispositivo).unsqueeze(1)
Xte_t = torch.tensor(Xte, device=dispositivo).unsqueeze(1)
ytr_t = torch.tensor(ytr, device=dispositivo)
yte_t = torch.tensor(yte, device=dispositivo)

for epoca in range(12):
    cnn.train()
    perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm), 64):
        idx = perm[i:i + 64]
        optimizador.zero_grad()
        perdida = criterio(cnn(Xtr_t[idx]), ytr_t[idx])
        perdida.backward()
        optimizador.step()

cnn.eval()
with torch.no_grad():
    acc_cnn = (cnn(Xte_t).argmax(1) == yte_t).float().mean().item()

print(f"Exactitud en test  ->  Transformer: {acc_transformer:.3f}   "
      f"CNN 1D: {acc_cnn:.3f}")

# ======================================================================
# 2.5 Visualización de los pesos de atención (ejercicio 12.2c)
# ======================================================================

idx_test = 0
espectro_token = torch.tensor(Tte[idx_test:idx_test + 1], device=dispositivo)
modelo.eval()
with torch.no_grad():
    _, pesos1 = modelo(espectro_token, devolver_pesos=True)

# Atencion del token [CLS] (fila 0) hacia cada token; descartamos la columna [CLS]
aten_cls = pesos1[0, 0, 1:].cpu().numpy()
espectro_continuo = Xte[idx_test]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True)
ax1.plot(espectro_continuo, color="tab:blue", lw=1.0)
ax1.set_ylabel("intensidad")
ax1.set_title(f"Espectro de test (clase real = {yte[idx_test]})")
ax2.plot(aten_cls, color="tab:red", lw=1.0)
ax2.set_ylabel("atencion [CLS]")
ax2.set_xlabel("bin de m/z")
plt.tight_layout(); plt.show()

# Correlacion entre intensidad y atencion recibida (este espectro)
corr = np.corrcoef(espectro_continuo, aten_cls)[0, 1]
print(f"Correlacion intensidad-atencion (un espectro): {corr:.3f}")

# Promediamos sobre el conjunto de test para una conclusion robusta:
# la correlacion de un solo espectro es ruidosa.
correlaciones = []
T_te_tensor = torch.tensor(Tte, device=dispositivo)
with torch.no_grad():
    for k in range(len(Tte)):
        _, pesos_k = modelo(T_te_tensor[k:k + 1], devolver_pesos=True)
        a_k = pesos_k[0, 0, 1:].cpu().numpy()
        correlaciones.append(np.corrcoef(Xte[k], a_k)[0, 1])
correlaciones = np.array(correlaciones)
print(f"Correlacion media sobre test: {correlaciones.mean():.3f} "
      f"(positiva en el {100 * (correlaciones > 0).mean():.0f}% de los espectros)")

# ======================================================================
# 3. ChemBERTa: propiedades moleculares a partir de SMILES
# ======================================================================

# pip install transformers
from transformers import AutoTokenizer, AutoModel

# Cargar ChemBERTa preentrenado (descarga desde HuggingFace la primera vez)
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
backbone = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
print("ChemBERTa cargado correctamente.")

class ChemBERTaRegressor(nn.Module):
    """ChemBERTa + cabeza de regresion para propiedad molecular."""

    def __init__(self, backbone, d_modelo=768, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.cabeza = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_modelo, 1)
        )

    def forward(self, input_ids, attention_mask):
        salidas = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask)
        # Representacion del token [CLS]
        cls = salidas.last_hidden_state[:, 0, :]
        return self.cabeza(cls).squeeze(-1)


modelo_chem = ChemBERTaRegressor(backbone)

# Tokenizar una lista de moleculas SMILES
smiles_lista = ["CC(=O)Nc1ccc(O)cc1",  # paracetamol
                "c1ccccc1",             # benceno
                "CCO"]                  # etanol

tokens = tokenizer(smiles_lista, padding=True, truncation=True,
                   max_length=128, return_tensors="pt")

with torch.no_grad():
    pred = modelo_chem(**tokens)
print(f"Predicciones (cabeza sin entrenar): {pred}")

# ======================================================================
# 3.1 Preparar el fine-tuning
# ======================================================================

# Fine-tuning: descongelar solo las ultimas capas + cabeza
for param in modelo_chem.backbone.parameters():
    param.requires_grad = False
for param in modelo_chem.backbone.encoder.layer[-2:].parameters():
    param.requires_grad = True
for param in modelo_chem.cabeza.parameters():
    param.requires_grad = True

optimizador = torch.optim.Adam(
    filter(lambda p: p.requires_grad, modelo_chem.parameters()),
    lr=1e-4
)

n_entrenables = sum(p.numel() for p in modelo_chem.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in modelo_chem.parameters())
print(f"Parametros entrenables: {n_entrenables:,} de {n_total:,} "
      f"({100 * n_entrenables / n_total:.1f}%)")

# ======================================================================
# Resumen
# ======================================================================

"""
planograma.py
=============

Modelo de optimizacion de planogramas OXXO (reto MA2007B) y dos metaheuristicas
para resolverlo: **Simulated Annealing** y **GRASP + busqueda local**.

El refrigerador se modela como una cuadricula de charolas (jx, jy) con espacios k
(ver `Modelo_planogramas.tex`). Este modulo:

  1. Carga el CSV historico y deriva todos los parametros del modelo.
  2. Implementa una representacion de solucion con evaluacion incremental
     (objetivo no lineal + penalizaciones por restricciones R1-R13).
  3. Implementa Simulated Annealing y GRASP+LS sobre el mismo evaluador.

Convenciones de indices (internos, base 0):
  - i  : producto,            i in 0..I-1
  - jx : columna de charola,  jx in 0..NX-1  (NX=5)   -> columna fisica jx+1
  - jy : nivel de charola,    jy in 0..NY-1  (NY=6)   -> 0 = ABAJO, NY-1 = ARRIBA
  - k  : espacio en charola,  k in 0..K-1
La codificacion `grid[jx, jy, k] = i` (o -1 si vacio) cumple R4 por construccion.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Geometria fija del refrigerador
# --------------------------------------------------------------------------- #
NX = 5   # columnas de charolas (coordenada horizontal jx)
NY = 6   # niveles de charolas  (coordenada vertical   jy), 0 = abajo
NCH = NX * NY  # 30 charolas

FAMILIAS = ["cola", "agua", "sabor"]   # conjunto L (3 familias)
L = len(FAMILIAS)
FAM_IDX = {f: l for l, f in enumerate(FAMILIAS)}


# --------------------------------------------------------------------------- #
#  Mapeo CHAROLA (1..30)  ->  (jx, jy)
# --------------------------------------------------------------------------- #
def charola_a_jxjy(charola: int) -> tuple[int, int]:
    """Convierte el numero de charola (1..30) a indices internos (jx, jy).

    La numeracion del CSV es VERTICAL por columnas (verificado con las coords X/Y):
    CH 1..6 -> columna 1 (de arriba hacia abajo), CH 7..12 -> columna 2, etc.

    Devuelve jx in 0..NX-1 y jy in 0..NY-1 con jy=0 = nivel inferior (abajo).
    """
    c = int(charola) - 1                 # base 0
    jx = c // NY                         # 0..4
    pos_desde_arriba = c % NY            # 0 = arriba ... 5 = abajo
    jy = (NY - 1) - pos_desde_arriba     # invertimos: 0 = abajo ... 5 = arriba
    return jx, jy


# --------------------------------------------------------------------------- #
#  Clasificacion de familia y parseo de volumen desde la descripcion
# --------------------------------------------------------------------------- #
# Palabras clave (en orden de prioridad). Editable.
_KW_COLA = ("coca", "cola", "pepsi")            # refrescos de cola
_KW_AGUA = ("agua", "mineral", "topo chico")    # aguas / mineralizadas


def clasificar_familia(desc: str) -> str:
    """Asigna una de las 3 familias {cola, agua, sabor} a partir de ITEM_DESC.

    'sabor' es la familia por defecto (jugos, energeticos, sabores, etc.).
    """
    d = (desc or "").lower()
    if any(k in d for k in _KW_COLA):
        # "Pepsi Black", "CocaCola", "Coca-Cola" -> cola. (light/zero siguen siendo cola)
        return "cola"
    if any(k in d for k in _KW_AGUA):
        return "agua"
    return "sabor"


_RE_LITROS = re.compile(r"([\d]+(?:\.\d+)?)\s*(?:l|lt|lts|litros?)\b", re.IGNORECASE)
_RE_ML = re.compile(r"([\d]+(?:\.\d+)?)\s*ml\b", re.IGNORECASE)


def parsear_volumen_ml(desc: str) -> float | None:
    """Extrae el volumen en mililitros desde la descripcion. None si no se encuentra.

    Corrige errores de captura donde se omitio el punto decimal en litros
    (p.ej. "125L"->1.25L, "25 lt"->2.5L, "15L"->1.5L): mientras el valor en
    litros sea irrealmente grande (>=5 L para una botella) se divide entre 10.
    """
    d = desc or ""
    m = _RE_ML.search(d)
    if m:
        return float(m.group(1))
    m = _RE_LITROS.search(d)
    if m:
        litros = float(m.group(1))
        while litros >= 5.0:        # ningun envase de este surtido supera ~3-4 L
            litros /= 10.0
        return litros * 1000.0
    return None


# --------------------------------------------------------------------------- #
#  Configuracion del problema / parametros ajustables
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    K: int = 9                     # espacios por charola (|K|)
    delta: float = 0.01            # peso de la sinergia en el objetivo
    epsilon: int = 10                    # maximo de charolas hibridas (R9)
    ponderar_F_por_frentes: bool = True   # F cuenta frentes en vez de filas
    max_productos: int | None = None      # limitar I a los N mas frecuentes (None = todos)

    # Valores por defecto / pisos para parametros geometricos
    alpha_default: float = 55.0    # ancho disponible de charola (cm)
    gamma_default: float = 38.0    # alto disponible de charola (cm)
    alpha_buffer: float = 1.0      # factor sobre el ancho historico observado
    gamma_buffer: float = 1.0
    peso_slack: float = 1.0        # factor sobre el peso maximo (tope del nivel inferior)
    peso_top_quantile: float = 0.5 # cuantil de peso que define el tope del nivel superior

    # Pesos de penalizacion (restricciones tratadas de forma blanda)
    pen_ancho: float = 100000.0
    pen_alto: float = 100000.0
    pen_peso: float = 100000.0
    pen_cobertura: float = 50000.0   # por producto no colocado (R3)
    pen_hibrido: float = 10000.0     # por charola hibrida que excede epsilon (R9)
    pen_columna: float = 10000.0     # por familia pura extra en una columna (R12)
    pen_disponible: float = 100000.0 # por usar charola no disponible (R6)


# --------------------------------------------------------------------------- #
#  Estructura con todos los parametros del modelo ya construidos
# --------------------------------------------------------------------------- #
@dataclass
class Problema:
    cfg: Config
    items: list                       # id de producto por indice i
    desc: list                        # descripcion por indice i
    beta: np.ndarray                  # ancho beta_i           (I,)
    theta: np.ndarray                 # alto  Theta_i          (I,)
    rho: np.ndarray                   # peso proxy rho_i (kg)  (I,)
    fam: np.ndarray                   # indice de familia      (I,)  in 0..L-1
    C: np.ndarray                     # C[i,l] pertenencia     (I, L)
    F: np.ndarray                     # F[i, jx, jy]           (I, NX, NY)
    G: np.ndarray                     # G[i1, i2] co-aparicion (I, I)
    alpha: np.ndarray                 # alpha[jx, jy]          (NX, NY)
    gamma: np.ndarray                 # gamma[jx, jy]          (NX, NY)
    rho_max: np.ndarray               # rho_max[jy]            (NY,)
    A: np.ndarray                     # A[jx, jy] disponible   (NX, NY) bool
    col_fam: np.ndarray               # familia asignada a cada columna (NX,)

    @property
    def I(self) -> int:
        return len(self.items)


# --------------------------------------------------------------------------- #
#  Construccion del problema a partir del CSV
# --------------------------------------------------------------------------- #
def construir_problema(csv_path: str, cfg: Config | None = None) -> Problema:
    """Lee el CSV historico y deriva todos los parametros del modelo."""
    cfg = cfg or Config()
    df = pd.read_csv(csv_path, encoding="latin-1")

    # Normalizar nombres de columna: quitar BOM/no-ASCII y arreglar TAMA(N)O_POST
    df.columns = [re.sub(r"[^\x20-\x7e]", "", str(c)).strip() for c in df.columns]
    df.rename(columns={c: "TAMANO_POST" for c in df.columns if c.startswith("TAMA")},
              inplace=True)

    # Normalizar tipos numericos relevantes
    df["CHAROLA"] = df["CHAROLA"].astype(int)
    df["NUM_FRENTES"] = pd.to_numeric(df["NUM_FRENTES"], errors="coerce").fillna(1).astype(int)
    df["ANCHO"] = pd.to_numeric(df["ANCHO"], errors="coerce")
    df["ALTO"] = pd.to_numeric(df["ALTO"], errors="coerce")

    # (jx, jy) por fila
    jxjy = df["CHAROLA"].map(charola_a_jxjy)
    df["jx"] = jxjy.map(lambda t: t[0])
    df["jy"] = jxjy.map(lambda t: t[1])

    # ----- Conjunto de productos I -----
    # Frecuencia global de cada producto (para limitar I si se pide)
    peso_fila = df["NUM_FRENTES"] if cfg.ponderar_F_por_frentes else 1
    df["_peso"] = peso_fila
    freq_global = df.groupby("ITEM")["_peso"].sum().sort_values(ascending=False)
    items = list(freq_global.index)
    if cfg.max_productos is not None:
        items = items[: cfg.max_productos]
    idx = {it: i for i, it in enumerate(items)}
    Inum = len(items)

    # Descripcion representativa por producto
    desc_map = (df.drop_duplicates("ITEM").set_index("ITEM")["ITEM_DESC"].to_dict())
    desc = [str(desc_map.get(it, "")) for it in items]

    # ----- Atributos por producto -----
    beta = np.zeros(Inum)
    theta = np.zeros(Inum)
    vol = np.zeros(Inum)
    fam = np.zeros(Inum, dtype=int)
    # ancho/alto: mediana observada del producto (robusta a ruido)
    ancho_med = df.groupby("ITEM")["ANCHO"].median()
    alto_med = df.groupby("ITEM")["ALTO"].median()
    for it, i in idx.items():
        beta[i] = float(ancho_med.get(it, np.nan))
        theta[i] = float(alto_med.get(it, np.nan))
        v = parsear_volumen_ml(desc[i])
        vol[i] = v if v is not None else np.nan
        fam[i] = FAM_IDX[clasificar_familia(desc[i])]
    # Imputar faltantes con la mediana del conjunto
    beta = np.where(np.isnan(beta), np.nanmedian(beta), beta)
    theta = np.where(np.isnan(theta), np.nanmedian(theta), theta)
    vol = np.where(np.isnan(vol), np.nanmedian(vol), vol)

    # Peso proxy: densidad ~1 g/ml -> kg = ml/1000
    rho = vol / 1000.0

    # C[i, l]
    C = np.zeros((Inum, L))
    C[np.arange(Inum), fam] = 1.0

    # ----- F[i, jx, jy] -----
    F = np.zeros((Inum, NX, NY))
    sub = df[df["ITEM"].isin(idx)]
    for it, jx, jy, w in zip(sub["ITEM"], sub["jx"], sub["jy"], sub["_peso"]):
        F[idx[it], jx, jy] += w

    # ----- G[i1, i2]: co-aparicion en la misma charola entre instancias -----
    # instancia = (CONJUNTO_ID, SEGMENTO_ID, TAMANO_POST, DIRECCION_LEGO_ID)
    G = np.zeros((Inum, Inum))
    claves = ["CONJUNTO_ID", "SEGMENTO_ID", "TAMANO_POST", "DIRECCION_LEGO_ID", "CHAROLA"]
    for _, grp in sub.groupby(claves):
        prods = sorted({idx[it] for it in grp["ITEM"]})
        for a in range(len(prods)):
            for b in range(a + 1, len(prods)):
                G[prods[a], prods[b]] += 1
                G[prods[b], prods[a]] += 1

    # ----- alpha[jx, jy]: ancho disponible (max historico de Sum(ancho*frentes)) -----
    df["_ancho_uso"] = df["ANCHO"] * df["NUM_FRENTES"]
    ancho_inst = (df.groupby(["CONJUNTO_ID", "SEGMENTO_ID", "TAMANO_POST",
                              "DIRECCION_LEGO_ID", "jx", "jy"])["_ancho_uso"].sum())
    ancho_max = ancho_inst.groupby(["jx", "jy"]).max()
    alpha = np.full((NX, NY), cfg.alpha_default)
    for (jx, jy), v in ancho_max.items():
        alpha[jx, jy] = max(cfg.alpha_default, float(v) * cfg.alpha_buffer)

    # ----- gamma[jx, jy]: alto disponible (max alto observado en esa charola) -----
    alto_max = df.groupby(["jx", "jy"])["ALTO"].max()
    gamma = np.full((NX, NY), cfg.gamma_default)
    for (jx, jy), v in alto_max.items():
        gamma[jx, jy] = max(cfg.gamma_default, float(v) * cfg.gamma_buffer)

    # ----- rho_max[jy]: tope de peso ESTRICTAMENTE decreciente con la altura -----
    # El .tex argumenta que el peso no debe inferirse del historico de frecuencias,
    # por lo que se usa un esquema decreciente disenado: el nivel inferior (jy=0)
    # admite al producto mas pesado y el tope baja linealmente hacia arriba. Asi se
    # garantiza factibilidad (todo cabe abajo) y se fuerza "pesados abajo".
    w_max = float(np.max(rho)) * cfg.peso_slack
    w_top = float(np.quantile(rho, cfg.peso_top_quantile))   # tope del nivel superior
    rho_max = np.linspace(w_max, w_top, NY)                  # jy=0 abajo ... NY-1 arriba

    # ----- A[jx, jy]: disponibilidad (todas disponibles por defecto) -----
    A = np.ones((NX, NY), dtype=bool)

    # ----- col_fam[jx]: familia dominante asignada a cada columna -----
    # Asignacion proporcional al numero de productos de cada familia.
    # Con cola=59, agua=16, sabor=114 y 5 columnas: sabor≈3, cola≈1, agua≈1.
    # Pero agua (16 prod) no puede llenar ni media columna (27 slots),
    # asi que fusionamos agua con la columna con mayor afinidad.
    fam_F_per_col = np.zeros((NX, L))
    for jx in range(NX):
        for l_idx in range(L):
            mask = (fam == l_idx)
            fam_F_per_col[jx, l_idx] = F[mask, jx, :].sum()

    # Asignacion proporcional por numero de productos
    fam_counts = np.bincount(fam, minlength=L).astype(float)
    # Columnas proporcionales (redondeando, minimo 0)
    col_shares = np.round(fam_counts / fam_counts.sum() * NX).astype(int)
    # Ajustar para que sumen NX
    while col_shares.sum() > NX:
        col_shares[np.argmax(col_shares)] -= 1
    while col_shares.sum() < NX:
        col_shares[np.argmin(col_shares)] += 1
    # Familias con 0 columnas comparten con otra (seran hibridas)
    # Asignar: ordenar columnas por F dominante de cada familia
    col_fam = np.full(NX, -1, dtype=int)
    available_cols = list(range(NX))
    for l_idx in np.argsort(-fam_counts):  # familia mas grande primero
        n_cols = col_shares[l_idx]
        if n_cols == 0:
            continue
        # Elegir las n_cols columnas con mayor F para esta familia
        scores = [(fam_F_per_col[jx, l_idx], jx) for jx in available_cols]
        scores.sort(reverse=True)
        for _, jx in scores[:n_cols]:
            col_fam[jx] = l_idx
            available_cols.remove(jx)
    # Columnas sin asignar -> familia con mayor F en esa columna
    for jx in range(NX):
        if col_fam[jx] < 0:
            col_fam[jx] = int(np.argmax(fam_F_per_col[jx]))

    return Problema(cfg=cfg, items=items, desc=desc, beta=beta, theta=theta,
                    rho=rho, fam=fam, C=C, F=F, G=G, alpha=alpha, gamma=gamma,
                    rho_max=rho_max, A=A, col_fam=col_fam)


# --------------------------------------------------------------------------- #
#  Solucion con evaluacion incremental
# --------------------------------------------------------------------------- #
class Solucion:
    """Asignacion grid[jx,jy,k] -> producto (o -1 vacio) con fitness incremental.

    fitness = objetivo - penalizaciones.  Mantiene caches por charola y totales
    globales para que aplicar/revertir un movimiento sea O(K) y fitness sea O(1).
    """

    def __init__(self, prob: Problema):
        self.prob = prob
        self.cfg = prob.cfg
        K = self.cfg.K
        self.grid = np.full((NX, NY, K), -1, dtype=int)
        self.counts = np.zeros(prob.I, dtype=int)   # veces que aparece cada producto

        # caches por charola
        self._obj = np.zeros((NX, NY))      # contribucion al objetivo
        self._pen = np.zeros((NX, NY))      # penalizacion local (ancho/alto/peso/disp)
        self._hibrida = np.zeros((NX, NY), dtype=bool)
        self._fam_pura = np.full((NX, NY), -1, dtype=int)  # familia pura (-1 si ninguna)

        # totales globales (cache)
        self._tot_obj = 0.0
        self._tot_pen_local = 0.0
        # las penalizaciones globales se recomputan a partir de los caches
        self._recalcular_todo()

    # ---- contribucion de una charola (objetivo + penalizacion local) ----
    def _contrib_charola(self, jx, jy):
        prob, cfg = self.prob, self.cfg
        cells = self.grid[jx, jy]
        ocup = cells[cells >= 0]
        if ocup.size == 0:
            return 0.0, 0.0, False, -1

        F = prob.F[:, jx, jy]
        # objetivo termino 1: suma de frecuencias por espacio ocupado
        obj = float(F[ocup].sum())

        # objetivo termino 2 (sinergia): sum_{i1,i2} G F F n1 n2
        cnt = Counter(ocup.tolist())
        prods = list(cnt.keys())
        syn = 0.0
        for a_i in range(len(prods)):
            ia = prods[a_i]
            na = cnt[ia]
            fa = F[ia]
            for b_i in range(len(prods)):
                ib = prods[b_i]
                syn += prob.G[ia, ib] * fa * F[ib] * na * cnt[ib]
        obj += cfg.delta * syn

        # penalizaciones locales
        pen = 0.0
        # R1 ancho
        ancho = float(prob.beta[ocup].sum())
        if ancho > prob.alpha[jx, jy]:
            pen += cfg.pen_ancho * (ancho - prob.alpha[jx, jy])
        # R2 alto (por espacio)
        exceso_alto = prob.theta[ocup] - prob.gamma[jx, jy]
        exceso_alto = exceso_alto[exceso_alto > 0]
        if exceso_alto.size:
            pen += cfg.pen_alto * float(exceso_alto.sum())
        # R10 peso por nivel
        exceso_peso = prob.rho[ocup] - prob.rho_max[jy]
        exceso_peso = exceso_peso[exceso_peso > 0]
        if exceso_peso.size:
            pen += cfg.pen_peso * float(exceso_peso.sum())
        # R6 disponibilidad
        if not prob.A[jx, jy]:
            pen += cfg.pen_disponible

        # familias presentes -> hibrida / familia pura
        fams = set(prob.fam[p] for p in prods)
        hibrida = len(fams) >= 2
        fam_pura = (next(iter(fams)) if len(fams) == 1 else -1)
        return obj, pen, hibrida, fam_pura

    def _recalcular_todo(self):
        for jx in range(NX):
            for jy in range(NY):
                o, p, h, fp = self._contrib_charola(jx, jy)
                self._obj[jx, jy] = o
                self._pen[jx, jy] = p
                self._hibrida[jx, jy] = h
                self._fam_pura[jx, jy] = fp
        self._tot_obj = float(self._obj.sum())
        self._tot_pen_local = float(self._pen.sum())

    # ---- penalizaciones globales (cobertura, hibridas, columnas) ----
    def _pen_global(self):
        cfg = self.cfg
        # R3 cobertura: cada producto al menos una vez
        no_colocados = int(np.count_nonzero(self.counts == 0))
        pen = cfg.pen_cobertura * no_colocados
        # R9 numero de charolas hibridas <= epsilon
        nh = int(self._hibrida.sum())
        if nh > cfg.epsilon:
            pen += cfg.pen_hibrido * (nh - cfg.epsilon)
        # R12 a lo mas una familia pura por columna
        for jx in range(NX):
            fams = set()
            for jy in range(NY):
                if not self._hibrida[jx, jy] and self._fam_pura[jx, jy] >= 0:
                    fams.add(self._fam_pura[jx, jy])
            if len(fams) > 1:
                pen += cfg.pen_columna * (len(fams) - 1)
        return pen, no_colocados, nh

    @property
    def fitness(self) -> float:
        pen_g, _, _ = self._pen_global()
        return self._tot_obj - self._tot_pen_local - pen_g

    # ---- aplicar un conjunto de cambios y devolver su inverso ----
    def aplicar(self, cambios):
        """cambios: lista de (jx, jy, k, nuevo_producto). Devuelve la lista inversa."""
        inverso = []
        charolas = set()
        for (jx, jy, k, nuevo) in cambios:
            viejo = int(self.grid[jx, jy, k])
            inverso.append((jx, jy, k, viejo))
            if viejo == nuevo:
                continue
            if viejo >= 0:
                self.counts[viejo] -= 1
            if nuevo >= 0:
                self.counts[nuevo] += 1
            self.grid[jx, jy, k] = nuevo
            charolas.add((jx, jy))
        for (jx, jy) in charolas:
            o, p, h, fp = self._contrib_charola(jx, jy)
            self._tot_obj += o - self._obj[jx, jy]
            self._tot_pen_local += p - self._pen[jx, jy]
            self._obj[jx, jy] = o
            self._pen[jx, jy] = p
            self._hibrida[jx, jy] = h
            self._fam_pura[jx, jy] = fp
        return inverso[::-1]

    # ---- utilidades ----
    def copia(self) -> "Solucion":
        s = Solucion.__new__(Solucion)
        s.prob = self.prob
        s.cfg = self.cfg
        s.grid = self.grid.copy()
        s.counts = self.counts.copy()
        s._obj = self._obj.copy()
        s._pen = self._pen.copy()
        s._hibrida = self._hibrida.copy()
        s._fam_pura = self._fam_pura.copy()
        s._tot_obj = self._tot_obj
        s._tot_pen_local = self._tot_pen_local
        return s

    def resumen(self) -> dict:
        pen_g, no_col, nh = self._pen_global()
        ncols_mal = 0
        for jx in range(NX):
            fams = {self._fam_pura[jx, jy] for jy in range(NY)
                    if not self._hibrida[jx, jy] and self._fam_pura[jx, jy] >= 0}
            if len(fams) > 1:
                ncols_mal += 1
        factible = (no_col == 0 and nh <= self.cfg.epsilon
                    and ncols_mal == 0 and self._tot_pen_local < 1e-5)
        return {
            "fitness": self.fitness,
            "objetivo": self._tot_obj,
            "penalizacion_local": self._tot_pen_local,
            "penalizacion_global": pen_g,
            "productos_no_colocados": no_col,
            "charolas_hibridas": nh,
            "epsilon": self.cfg.epsilon,
            "columnas_con_conflicto": ncols_mal,
            "espacios_usados": int(np.count_nonzero(self.grid >= 0)),
            "factible": bool(factible),
        }


# --------------------------------------------------------------------------- #
#  Movimientos / vecindarios (compartidos por SA y LS)
# --------------------------------------------------------------------------- #
def _todas_las_celdas(K):
    for jx in range(NX):
        for jy in range(NY):
            for k in range(K):
                yield jx, jy, k


def proponer_movimiento(sol: Solucion, rng: np.random.Generator):
    """Genera un movimiento aleatorio: cambiar, insertar, quitar, intercambiar,
    o intercambio familia-consciente.

    Devuelve la lista de cambios (jx, jy, k, nuevo_producto).
    """
    prob = sol.prob
    K = sol.cfg.K
    tipo = rng.integers(0, 6)
    jx = int(rng.integers(0, NX)); jy = int(rng.integers(0, NY)); k = int(rng.integers(0, K))

    if tipo == 0:        # cambiar/insertar producto de la familia correcta
        fam_col = prob.col_fam[jx]
        cands = np.where(prob.fam == fam_col)[0]
        if cands.size > 0 and rng.random() < 0.7:  # 70% misma familia
            p = int(rng.choice(cands))
        else:
            p = int(rng.integers(0, prob.I))
        return [(jx, jy, k, p)]
    if tipo == 1:        # quitar (vaciar celda)
        return [(jx, jy, k, -1)]
    if tipo == 2:        # colocar producto no cubierto
        no_col = np.where(sol.counts == 0)[0]
        if no_col.size > 0:
            p = int(rng.choice(no_col))
            # Intentar colocarlo en columna de su familia
            fam_p = prob.fam[p]
            cols_fam = np.where(prob.col_fam == fam_p)[0]
            if cols_fam.size > 0:
                jx = int(rng.choice(cols_fam))
        else:
            p = int(rng.integers(0, prob.I))
        return [(jx, jy, k, p)]
    if tipo == 3:        # intercambiar dos celdas
        jx2 = int(rng.integers(0, NX)); jy2 = int(rng.integers(0, NY))
        k2 = int(rng.integers(0, K))
        pa = int(sol.grid[jx, jy, k]); pb = int(sol.grid[jx2, jy2, k2])
        return [(jx, jy, k, pb), (jx2, jy2, k2, pa)]
    if tipo == 4:        # intercambio familia-consciente: mover producto
        # "equivocado" a su columna correcta
        actual = int(sol.grid[jx, jy, k])
        if actual >= 0 and prob.fam[actual] != prob.col_fam[jx]:
            # Buscar celda en columna correcta para intercambiar
            fam_p = prob.fam[actual]
            cols_ok = np.where(prob.col_fam == fam_p)[0]
            if cols_ok.size > 0:
                jx2 = int(rng.choice(cols_ok))
                jy2 = int(rng.integers(0, NY))
                k2 = int(rng.integers(0, K))
                pb = int(sol.grid[jx2, jy2, k2])
                return [(jx, jy, k, pb), (jx2, jy2, k2, actual)]
        return [(jx, jy, k, -1)]  # fallback: vaciar
    # tipo == 5: mover producto a charola con mejor F
    actual = int(sol.grid[jx, jy, k])
    if actual >= 0:
        best_jx = int(np.argmax(prob.F[actual, :, jy].sum(axis=-1) if jy < NY else 0))
        jy2 = int(rng.integers(0, NY))
        k2 = int(rng.integers(0, K))
        pb = int(sol.grid[best_jx, jy2, k2])
        return [(jx, jy, k, pb), (best_jx, jy2, k2, actual)]
    return [(jx, jy, k, -1)]


# --------------------------------------------------------------------------- #
#  Solucion inicial (greedy aleatorizada consciente de familias)
# --------------------------------------------------------------------------- #
def _charola_factible(prob: Problema, p: int, jx: int, jy: int,
                      rem_ancho: np.ndarray, grid: np.ndarray) -> bool:
    """True si el producto p cabe fisicamente en la charola (jx, jy)."""
    if not prob.A[jx, jy]:
        return False
    if prob.rho[p] > prob.rho_max[jy]:          # R10 peso
        return False
    if prob.theta[p] > prob.gamma[jx, jy]:      # R2 alto
        return False
    if rem_ancho[jx, jy] < prob.beta[p]:        # R1 ancho restante
        return False
    if not np.any(grid[jx, jy] < 0):            # espacio libre
        return False
    return True


def construir_inicial(prob: Problema, rng: np.random.Generator,
                      rcl_frac: float = 0.3) -> Solucion:
    """Construccion greedy-aleatorizada consciente de familias.

    Estrategia:
      1. Asignar cada producto a una charola en una columna de su familia.
         Esto minimiza charolas hibridas y respeta R12 (bloqueo vertical).
      2. Si no cabe en su columna, intentar cualquier charola (genera hibrida).
      3. Rellenar espacios libres con frentes adicionales de la misma familia.
    """
    sol = Solucion(prob)
    rem_ancho = prob.alpha.copy()

    def _colocar(p, jx, jy):
        libres = np.where(sol.grid[jx, jy] < 0)[0]
        if libres.size == 0:
            return False
        sol.aplicar([(jx, jy, int(libres[0]), p)])
        rem_ancho[jx, jy] -= prob.beta[p]
        return True

    # Columnas asignadas a cada familia
    fam_cols = {l: np.where(prob.col_fam == l)[0] for l in range(L)}

    # 1) Cobertura: cada producto al menos una vez
    #    Priorizar productos mas anchos (mas dificil colocarlos despues)
    orden = sorted(range(prob.I), key=lambda i: -prob.beta[i])
    for p in orden:
        fam_p = prob.fam[p]
        # Primero intentar columnas de su familia
        cand = []
        for jx in fam_cols[fam_p]:
            for jy in range(NY):
                if _charola_factible(prob, p, jx, jy, rem_ancho, sol.grid):
                    score = prob.F[p, jx, jy]
                    # Bonus: misma familia ya presente (mantiene pureza)
                    if sol._fam_pura[jx, jy] == fam_p or sol._fam_pura[jx, jy] == -1:
                        score += 5.0
                    cand.append((score, jx, jy))
        # Si no hay candidatos en su familia, probar cualquier columna
        if not cand:
            for jx in range(NX):
                if jx in fam_cols[fam_p]:
                    continue  # ya probadas
                for jy in range(NY):
                    if _charola_factible(prob, p, jx, jy, rem_ancho, sol.grid):
                        score = prob.F[p, jx, jy] - 2.0  # penalidad por hibridez
                        cand.append((score, jx, jy))
        if not cand:
            continue  # no colocado (penalizacion R3)
        cand.sort(reverse=True)
        n = max(1, int(len(cand) * rcl_frac))
        _, jx, jy = cand[int(rng.integers(0, n))]
        _colocar(p, jx, jy)

    # 2) Rellenar espacios libres con frentes adicionales
    for jx in range(NX):
        fam_col = prob.col_fam[jx]
        for jy in range(NY):
            while np.any(sol.grid[jx, jy] < 0):
                Fcol = prob.F[:, jx, jy].astype(float).copy()
                # Filtrar por factibilidad
                Fcol[prob.rho > prob.rho_max[jy]] = -1
                Fcol[prob.theta > prob.gamma[jx, jy]] = -1
                Fcol[prob.beta > rem_ancho[jx, jy]] = -1
                # Preferir misma familia que la charola (si ya es pura)
                target_fam = sol._fam_pura[jx, jy] if sol._fam_pura[jx, jy] >= 0 else fam_col
                misma_fam = (prob.fam == target_fam)
                # Intentar primero con misma familia
                Fcol_fam = Fcol.copy()
                Fcol_fam[~misma_fam] = -1
                if np.any(Fcol_fam > 0):
                    use = Fcol_fam
                elif np.any(Fcol > 0):
                    use = Fcol  # fallback: cualquier familia
                else:
                    break
                top_n = max(1, int(np.count_nonzero(use > 0) * 0.15))
                orden_p = np.argsort(-use)[:top_n]
                orden_p = [q for q in orden_p if use[q] > 0]
                if not orden_p:
                    break
                p = int(rng.choice(orden_p))
                k = int(np.where(sol.grid[jx, jy] < 0)[0][0])
                sol.aplicar([(jx, jy, k, p)])
                rem_ancho[jx, jy] -= prob.beta[p]

    return sol


# --------------------------------------------------------------------------- #
#  Simulated Annealing
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoMeta:
    solucion: Solucion
    historia: list           # mejor fitness por iteracion/registro
    tiempo: float
    iteraciones: int
    nombre: str = ""


def simulated_annealing(prob: Problema, rng: np.random.Generator | None = None,
                        T0: float = 50.0, Tmin: float = 1e-2, alpha_cool: float = 0.95,
                        iter_por_T: int = 400, sol_inicial: Solucion | None = None,
                        registrar_cada: int = 200, verbose: bool = False) -> ResultadoMeta:
    """Recocido simulado con enfriamiento geometrico T <- alpha_cool * T."""
    rng = rng or np.random.default_rng()
    t0 = time.perf_counter()
    sol = sol_inicial.copia() if sol_inicial else construir_inicial(prob, rng)
    mejor = sol.copia()
    mejor_fit = mejor.fitness
    historia = [mejor_fit]
    it = 0
    T = T0
    while T > Tmin:
        for _ in range(iter_por_T):
            it += 1
            cambios = proponer_movimiento(sol, rng)
            f_old = sol.fitness
            inv = sol.aplicar(cambios)
            f_new = sol.fitness
            d = f_new - f_old
            if d >= 0 or rng.random() < math.exp(d / T):
                if f_new > mejor_fit:
                    mejor_fit = f_new
                    mejor = sol.copia()
            else:
                sol.aplicar(inv)            # revertir
            if it % registrar_cada == 0:
                historia.append(mejor_fit)
        T *= alpha_cool
        if verbose:
            print(f"  T={T:8.3f}  mejor_fit={mejor_fit:12.2f}")
    # Reparar cobertura en la mejor solucion (R3)
    _reparar_cobertura(mejor, rng)
    historia.append(mejor.fitness)
    return ResultadoMeta(mejor, historia, time.perf_counter() - t0, it, "Simulated Annealing")


def _reparar_cobertura(sol: Solucion, rng: np.random.Generator):
    """Reparacion post-hoc: forzar que cada producto aparezca al menos una vez.

    Para cada producto no colocado, buscar una celda donde reemplazarlo
    (prefiriendo celdas vacias, duplicados de productos con muchas copias,
    o en charola compatible con su familia/peso/alto/ancho).
    """
    prob = sol.prob
    no_col = np.where(sol.counts == 0)[0]
    if no_col.size == 0:
        return

    def _ancho_charola(jx, jy):
        ocup = sol.grid[jx, jy][sol.grid[jx, jy] >= 0]
        return float(prob.beta[ocup].sum())

    for p in no_col:
        fam_p = prob.fam[p]
        colocado = False

        # 1) Buscar celda vacia en columna de su familia (que quepa en ancho)
        cols_fam = np.where(prob.col_fam == fam_p)[0]
        for jx in cols_fam:
            for jy in range(NY):
                if not prob.A[jx, jy]:
                    continue
                if prob.rho[p] > prob.rho_max[jy]:
                    continue
                if prob.theta[p] > prob.gamma[jx, jy]:
                    continue
                if _ancho_charola(jx, jy) + prob.beta[p] > prob.alpha[jx, jy]:
                    continue
                libres = np.where(sol.grid[jx, jy] < 0)[0]
                if libres.size > 0:
                    sol.aplicar([(jx, jy, int(libres[0]), p)])
                    colocado = True
                    break
            if colocado:
                break
        if colocado:
            continue

        # 2) Buscar celda vacia en cualquier columna disponible (que quepa en ancho)
        for jx in range(NX):
            for jy in range(NY):
                if not prob.A[jx, jy]:
                    continue
                if prob.rho[p] > prob.rho_max[jy]:
                    continue
                if prob.theta[p] > prob.gamma[jx, jy]:
                    continue
                if _ancho_charola(jx, jy) + prob.beta[p] > prob.alpha[jx, jy]:
                    continue
                libres = np.where(sol.grid[jx, jy] < 0)[0]
                if libres.size > 0:
                    sol.aplicar([(jx, jy, int(libres[0]), p)])
                    colocado = True
                    break
            if colocado:
                break
        if colocado:
            continue

        # 3) Reemplazar duplicado (producto q con >1 copia) prefiriendo misma columna/familia
        #    de modo que el ancho nuevo no exceda el limite.
        for jx in list(cols_fam) + [x for x in range(NX) if x not in cols_fam]:
            for jy in range(NY):
                if not prob.A[jx, jy]:
                    continue
                if prob.rho[p] > prob.rho_max[jy]:
                    continue
                if prob.theta[p] > prob.gamma[jx, jy]:
                    continue
                current_w = _ancho_charola(jx, jy)
                for k in range(sol.cfg.K):
                    q = int(sol.grid[jx, jy, k])
                    if q >= 0 and sol.counts[q] > 1:
                        if current_w - prob.beta[q] + prob.beta[p] <= prob.alpha[jx, jy]:
                            sol.aplicar([(jx, jy, k, p)])
                            colocado = True
                            break
                if colocado:
                    break
            if colocado:
                break
        if colocado:
            continue

        # 4) Fallback extremo: Reemplazar cualquier celda ocupada que sea factible fisicamente
        for jx in range(NX):
            for jy in range(NY):
                if not prob.A[jx, jy]:
                    continue
                if prob.rho[p] > prob.rho_max[jy]:
                    continue
                if prob.theta[p] > prob.gamma[jx, jy]:
                    continue
                current_w = _ancho_charola(jx, jy)
                for k in range(sol.cfg.K):
                    q = int(sol.grid[jx, jy, k])
                    if q >= 0:
                        if current_w - prob.beta[q] + prob.beta[p] <= prob.alpha[jx, jy]:
                            sol.aplicar([(jx, jy, k, p)])
                            colocado = True
                            break
                if colocado:
                    break
            if colocado:
                break


# --------------------------------------------------------------------------- #
#  GRASP + busqueda local
# --------------------------------------------------------------------------- #
def busqueda_local(sol: Solucion, rng: np.random.Generator,
                   max_iter: int = 4000, muestras: int = 40) -> Solucion:
    """Busqueda local de primer-mejora con vecindario muestreado."""
    it = 0
    mejora = True
    while mejora and it < max_iter:
        mejora = False
        for _ in range(muestras):
            it += 1
            cambios = proponer_movimiento(sol, rng)
            f_old = sol.fitness
            inv = sol.aplicar(cambios)
            if sol.fitness > f_old + 1e-9:        # primer-mejora
                mejora = True
                break
            sol.aplicar(inv)                       # revertir
            if it >= max_iter:
                break
    return sol


def grasp(prob: Problema, rng: np.random.Generator | None = None,
          n_iter: int = 20, rcl_frac: float = 0.3,
          ls_max_iter: int = 4000, verbose: bool = False) -> ResultadoMeta:
    """GRASP: multi-arranque de construccion greedy-aleatorizada + busqueda local."""
    rng = rng or np.random.default_rng()
    t0 = time.perf_counter()
    mejor = None
    mejor_fit = -math.inf
    historia = []
    for it in range(n_iter):
        sol = construir_inicial(prob, rng, rcl_frac=rcl_frac)
        sol = busqueda_local(sol, rng, max_iter=ls_max_iter)
        f = sol.fitness
        if f > mejor_fit:
            mejor_fit = f
            mejor = sol.copia()
        historia.append(mejor_fit)
        if verbose:
            print(f"  GRASP iter {it+1:2d}/{n_iter}  fit={f:12.2f}  mejor={mejor_fit:12.2f}")
    return ResultadoMeta(mejor, historia, time.perf_counter() - t0, n_iter, "GRASP + LS")


# --------------------------------------------------------------------------- #
#  Prueba rapida como script
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os

    aqui = os.path.dirname(os.path.abspath(__file__))
    csv = os.path.join(aqui, "ejemplo_planograma (1).csv")

    print("=== Validacion del mapeo CHAROLA -> (jx, jy) ===")
    for ch in (1, 6, 7, 13, 23, 30):
        print(f"  CHAROLA {ch:2d} -> {charola_a_jxjy(ch)}")
    print("  (CH1 -> col0 arriba (jy=5); CH6 -> col0 abajo (jy=0))")

    print("\n=== Construyendo problema ===")
    prob = construir_problema(csv)
    print(f"  productos (I)        : {prob.I}")
    print(f"  familias (cuenta)    : {np.bincount(prob.fam, minlength=L)} -> {FAMILIAS}")
    print(f"  beta  (ancho)  min/max: {prob.beta.min():.1f} / {prob.beta.max():.1f}")
    print(f"  theta (alto)   min/max: {prob.theta.min():.1f} / {prob.theta.max():.1f}")
    print(f"  rho   (peso kg)min/max: {prob.rho.min():.2f} / {prob.rho.max():.2f}")
    print(f"  rho_max por nivel(0=abajo): {np.round(prob.rho_max,2)}")
    print(f"  alpha rango          : {prob.alpha.min():.1f} .. {prob.alpha.max():.1f}")
    print(f"  gamma rango          : {prob.gamma.min():.1f} .. {prob.gamma.max():.1f}")
    print(f"  F total              : {prob.F.sum():.0f}")
    print(f"  G nnz                : {int(np.count_nonzero(prob.G))}")

    print("\n=== Prueba rapida de los algoritmos (presupuesto reducido) ===")
    rng = np.random.default_rng(42)
    r_sa = simulated_annealing(prob, np.random.default_rng(1), iter_por_T=150, alpha_cool=0.9)
    print(f"  SA   : fit={r_sa.solucion.fitness:12.2f}  t={r_sa.tiempo:5.1f}s  -> {r_sa.solucion.resumen()['factible']}")
    r_gr = grasp(prob, np.random.default_rng(2), n_iter=3, ls_max_iter=1500)
    print(f"  GRASP: fit={r_gr.solucion.fitness:12.2f}  t={r_gr.tiempo:5.1f}s  -> {r_gr.solucion.resumen()['factible']}")
    print("\nOK")

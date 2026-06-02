"""
Planograma OXXO — App Streamlit
Reto MA2007B
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import os

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modelos import (correr_pulp, correr_sa, correr_grasp,
                     cargar_optimo, extraer_referencias)
from visualizacion import render_fridge_html, sol_to_grid

# ── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Planograma OXXO",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS GLOBAL CON BRANDING OXXO ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
:root {
    --oxxo-red: #E60012;
    --oxxo-red-dark: #B8000E;
    --oxxo-yellow: #FFCC00;
    --oxxo-yellow-dark: #E5B800;
    --bg: #FFFBF0;
    --surface: #FFFFFF;
    --surface2: #FFF7DC;
    --border: #F0E5BC;
    --text: #1A1A1A;
    --text-soft: #4A4A4A;
    --muted: #7A7A7A;
}

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(at 0% 0%, rgba(255,204,0,0.08) 0%, transparent 50%),
        radial-gradient(at 100% 100%, rgba(230,0,18,0.05) 0%, transparent 50%);
    color: var(--text);
}

/* FORZAR colores de texto en todos los elementos de Streamlit */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {
    color: var(--text);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: var(--text) !important;
    font-weight: 800;
}
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong {
    color: var(--text);
}
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3 {
    color: var(--text) !important;
}
/* Labels de file uploaders */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploader"] label span,
[data-testid="stFileUploader"] small {
    color: var(--text) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--text-soft) !important;
}
/* Radio button labels */
.stRadio label, .stRadio label p, .stRadio label span {
    color: var(--text) !important;
}
/* Markdown headers fuera y dentro */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p {
    color: var(--text);
}
/* Info/banner azul de "Sube los dos archivos" */
.stAlert, .stAlert p, .stAlert div {
    color: var(--text) !important;
}
/* Expander content */
[data-testid="stExpander"] p,
[data-testid="stExpander"] li,
[data-testid="stExpander"] strong,
[data-testid="stExpander"] h1,
[data-testid="stExpander"] h2,
[data-testid="stExpander"] h3,
[data-testid="stExpander"] div {
    color: var(--text) !important;
}

/* Ocultar header default de Streamlit */
header[data-testid="stHeader"] { background: transparent; height: 0; }
.stDeployButton { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Header OXXO custom */
.oxxo-header {
    background: var(--oxxo-red);
    border-bottom: 4px solid var(--oxxo-yellow);
    padding: 14px 32px;
    margin: -1rem -1rem 0 -1rem;
    box-shadow: 0 4px 20px rgba(230,0,18,0.25);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.oxxo-brand { display: flex; align-items: center; gap: 16px; }
.oxxo-logo {
    background: white; color: var(--oxxo-red);
    font-weight: 900; font-size: 24px; letter-spacing: -0.04em;
    padding: 6px 14px; border-radius: 4px;
    box-shadow: 0 2px 0 var(--oxxo-yellow);
    transform: skewX(-4deg); display: inline-block;
}
.oxxo-logo span { transform: skewX(4deg); display: inline-block; }
.oxxo-brand-title {
    font-size: 16px; font-weight: 800; color: white; line-height: 1.1;
}
.oxxo-brand-sub {
    font-size: 11px; font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.9); margin-top: 2px;
}
.oxxo-pills { display: flex; gap: 8px; }
.oxxo-pill {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 6px 12px; border-radius: 20px;
    font-size: 11px; font-family: 'JetBrains Mono', monospace; color: white;
}
.oxxo-pill strong { color: var(--oxxo-yellow); font-weight: 700; }

/* Pestañas */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-bottom: 2px solid var(--border);
    gap: 0;
    padding: 0 16px;
}
.stTabs [data-baseweb="tab"] {
    padding: 14px 24px;
    color: var(--muted);
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    border-bottom: 3px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--oxxo-red) !important;
    border-bottom-color: var(--oxxo-red) !important;
    background: transparent !important;
}

/* Botones */
.stButton > button {
    background: var(--oxxo-red);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: 700;
    font-size: 13px;
    box-shadow: 0 2px 0 var(--oxxo-yellow-dark);
    transition: all .15s;
}
.stButton > button:hover {
    background: var(--oxxo-red-dark);
    transform: translateY(-1px);
    box-shadow: 0 4px 0 var(--oxxo-yellow-dark);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 2px dashed var(--oxxo-yellow);
    border-radius: 12px;
    padding: 24px;
}
[data-testid="stFileUploader"] section {
    background: var(--surface2);
}

/* Métricas (cards) */
.metric-card {
    background: var(--surface);
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    position: relative;
}
.metric-card.highlight { border-color: var(--oxxo-red); }
.metric-card.highlight::before {
    content: 'MEJOR';
    position: absolute; top: 12px; right: 12px;
    background: var(--oxxo-yellow); color: var(--text);
    font-size: 9px; font-weight: 800;
    padding: 3px 8px; border-radius: 12px;
    letter-spacing: 0.08em;
}
.metric-card.optimo {
    border-color: #2E7D32;
    background: #F1F8E9;
}
.metric-card.optimo::before {
    content: 'REFERENCIA OXXO';
    background: #2E7D32; color: white;
    position: absolute; top: 12px; right: 12px;
    font-size: 9px; font-weight: 800;
    padding: 3px 8px; border-radius: 12px;
    letter-spacing: 0.08em;
}
.metric-model-name {
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--oxxo-red);
    margin-bottom: 4px;
}
.metric-card.optimo .metric-model-name { color: #2E7D32; }
.metric-model-type { font-size: 13px; font-weight: 600; color: var(--text-soft); margin-bottom: 20px; }
.metric-big-num {
    font-size: 56px; font-weight: 900; line-height: 1;
    letter-spacing: -0.04em; color: var(--text);
}
.metric-big-num .unit { font-size: 16px; font-weight: 600; color: var(--muted); margin-left: 4px; }
.metric-big-label {
    font-size: 11px; color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 6px; margin-bottom: 18px;
}
.metric-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 12px;
}
.metric-row:last-child { border-bottom: none; }
.metric-row-label { color: var(--text-soft); font-weight: 500; }
.metric-row-value {
    font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--text);
}

/* Comparison bars */
.cmp-row {
    display: grid; grid-template-columns: 160px 1fr 100px;
    align-items: center; gap: 16px;
    padding: 12px 0; border-bottom: 1px solid var(--border);
}
.cmp-row:last-child { border-bottom: none; }
.cmp-row-name { font-size: 12px; font-weight: 700; }
.cmp-row-tag { font-size: 10px; color: var(--muted); font-family: 'JetBrains Mono', monospace; }
.cmp-bar-track {
    height: 22px; background: var(--surface2);
    border-radius: 11px; overflow: hidden;
}
.cmp-bar-fill {
    height: 100%; border-radius: 11px;
    display: flex; align-items: center; justify-content: flex-end;
    padding-right: 10px; color: white;
    font-size: 11px; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.cmp-bar-fill.pulp  { background: linear-gradient(90deg, #B8000E, #E60012); }
.cmp-bar-fill.sa    { background: linear-gradient(90deg, #E5B800, #FFCC00); color: #1A1A1A; }
.cmp-bar-fill.grasp { background: linear-gradient(90deg, #4A4A4A, #1A1A1A); }
.cmp-row-value {
    text-align: right; font-family: 'JetBrains Mono', monospace;
    font-weight: 700; font-size: 14px;
}

/* Sección comparativa */
.comparison-section {
    background: var(--surface); border: 2px solid var(--border);
    border-radius: 12px; padding: 24px; margin-bottom: 20px;
}
.section-title {
    font-size: 12px; font-weight: 800; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--oxxo-red); margin-bottom: 4px;
}
.section-sub { font-size: 13px; color: var(--muted); margin-bottom: 20px; }

/* Tabla de efectividad */
.efe-table {
    width: 100%; border-collapse: collapse; margin-top: 12px;
}
.efe-table thead tr { border-bottom: 2px solid var(--oxxo-red); }
.efe-table th {
    text-align: center; padding: 12px 8px;
    font-size: 11px; font-weight: 800;
    color: var(--oxxo-red); letter-spacing: 0.1em;
    text-transform: uppercase;
}
.efe-table th:first-child { text-align: left; color: var(--muted); }
.efe-table tbody tr { border-bottom: 1px solid var(--border); }
.efe-table td {
    padding: 12px 8px;
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    text-align: center;
}
.efe-table td:first-child {
    text-align: left;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: var(--text-soft);
}
.efe-table td.best {
    background: rgba(255,204,0,0.2);
    color: var(--oxxo-red);
    font-weight: 800;
}

/* Modal/Expander estilo */
[data-testid="stExpander"] {
    background: var(--surface);
    border: 2px solid var(--border);
    border-radius: 12px;
}
[data-testid="stExpander"] summary {
    background: var(--oxxo-yellow);
    font-weight: 800;
    color: var(--text);
    padding: 14px 20px;
}

.restriction-card {
    background: var(--surface2); border-radius: 8px;
    padding: 14px 16px; margin-bottom: 10px;
    border-left: 4px solid var(--oxxo-red);
}
.restriction-card .r-name {
    font-weight: 800; color: var(--oxxo-red);
    font-size: 12px; letter-spacing: 0.05em; margin-bottom: 4px;
}
.restriction-card .r-desc { font-size: 12px; color: var(--text-soft); line-height: 1.5; }

/* Status banners */
.status-banner {
    background: var(--surface2);
    border: 2px solid var(--oxxo-yellow);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
}
.status-banner.success {
    background: #F1F8E9;
    border-color: #2E7D32;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="oxxo-header">
    <div class="oxxo-brand">
        <div class="oxxo-logo"><span>OXXO</span></div>
        <div>
            <div class="oxxo-brand-title">Planograma Inteligente</div>
            <div class="oxxo-brand-sub">REFRESCOS &middot; CF &middot; DI &middot; TAMAÑO 3</div>
        </div>
    </div>
    <div class="oxxo-pills">
        <div class="oxxo-pill">Reto &middot; <strong>MA2007B</strong></div>
        <div class="oxxo-pill">Equipo &middot; <strong>5 personas</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── ESTADO ──────────────────────────────────────────────────────────────────
if 'resultados' not in st.session_state:
    st.session_state.resultados = None
    st.session_state.optimo_df = None
    st.session_state.refs = None


# ── SIDEBAR PARA UPLOAD ─────────────────────────────────────────────────────
st.markdown("### Carga de Datos")

col_up1, col_up2 = st.columns(2)

with col_up1:
    st.markdown("**1. Histórico de planogramas** (oxxo_1.csv)")
    hist_file = st.file_uploader(
        "Sube oxxo_1.csv",
        type=['csv'],
        key='hist',
        label_visibility='collapsed',
    )

with col_up2:
    st.markdown("**2. Óptimo OXXO** (ejemplo_planograma.csv)")
    opt_file = st.file_uploader(
        "Sube ejemplo_planograma.csv",
        type=['csv'],
        key='opt',
        label_visibility='collapsed',
    )

# Botón para correr modelos
correr = st.button("CORRER LOS 3 MODELOS", type="primary",
                    use_container_width=True,
                    disabled=(hist_file is None or opt_file is None))


# ── EJECUTAR MODELOS ─────────────────────────────────────────────────────────
if correr and hist_file and opt_file:
    with st.spinner("Cargando óptimo OXXO..."):
        # Guardar archivos temporales
        hist_path = '/tmp/_hist_input.csv'
        opt_path = '/tmp/_opt_input.csv'
        with open(hist_path, 'wb') as f:
            f.write(hist_file.getbuffer())
        with open(opt_path, 'wb') as f:
            f.write(opt_file.getbuffer())

        # Cargar óptimo
        try:
            optimo = cargar_optimo(opt_path)
            refs = extraer_referencias(optimo)
            st.session_state.optimo_df = optimo
            st.session_state.refs = refs
        except Exception as e:
            st.error(f"Error al cargar óptimo: {e}")
            st.stop()

        # Cargar histórico
        try:
            hist = pd.read_csv(hist_path)
            if 'DIRECCION_LEGO_ID' in hist.columns:
                hist['DIRECCION_LEGO_ID'] = hist['DIRECCION_LEGO_ID'].str.strip()
            hist_filt = hist[
                (hist['PLANOGRUPO'] == 'Refrescos') &
                (hist['MUEBLE_ID'] == 'CF') &
                (hist['DIRECCION_LEGO_ID'] == 'DI') &
                (hist['TAMAÑO'] == 3.0)
            ].copy()
        except Exception as e:
            st.error(f"Error al cargar histórico: {e}")
            st.stop()

    resultados = {}

    # PuLP
    with st.spinner("Corriendo PuLP MIQP (modelo exacto)..."):
        sol_p, m_p = correr_pulp(optimo, refs)
        resultados['pulp'] = {'sol': sol_p, 'metrics': m_p}

    # SA
    with st.spinner("Corriendo Simulated Annealing..."):
        sol_s, m_s = correr_sa(hist_filt, optimo, refs)
        resultados['sa'] = {'sol': sol_s, 'metrics': m_s}

    # GRASP
    with st.spinner("Corriendo GRASP + Local Search..."):
        sol_g, m_g = correr_grasp(hist_filt, optimo, refs)
        resultados['grasp'] = {'sol': sol_g, 'metrics': m_g}

    st.session_state.resultados = resultados

    st.markdown("""
    <div class="status-banner success">
        <div style="font-weight:800;color:#2E7D32;font-size:14px;letter-spacing:0.05em;">
            MODELOS EJECUTADOS CORRECTAMENTE
        </div>
        <div style="font-size:13px;color:var(--text-soft);margin-top:4px;">
            Los 3 modelos terminaron. Selecciona una pestaña para ver los resultados.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── MOSTRAR RESULTADOS ──────────────────────────────────────────────────────
if st.session_state.resultados is None:
    st.info("Sube los dos archivos y presiona el botón para correr los modelos.")
    with st.expander("Cómo funciona el modelo (click para abrir)", expanded=False):
        st.markdown("""
        ### Objetivo
        Asignar productos a las 18 charolas (3 puertas × 6 niveles) del refrigerador OXXO de forma que la solución
        **se parezca al óptimo aprobado por OXXO** (archivo `ejemplo_planograma.csv`, plan "opt_x_lego" con segmentos BCO, CLA, HRN).

        ### Geometría real
        - **3 puertas** de 55 cm de ancho.
        - **6 niveles** con alturas irregulares: N.1 y N.2 = 42cm (caben 1.5L/2L), N.3 y N.4 = 31.5cm, N.5 = 28cm, N.6 = 25cm.

        ### Función objetivo
        Para cada producto i y posición (jx, jy), `F[i,jx,jy]` = veces que ese producto aparece en esa posición en el óptimo OXXO.

        ```
        max Σ F[i,jx,jy] · X[i,jx,jy,k]  +  δ · Σ G[i1,i2] · X[i1] · X[i2]
        ```

        ### Tres métodos
        - **PuLP MIQP:** Optimización exacta con solver CBC.
        - **Simulated Annealing:** Metaheurística que parte de un planograma óptimo OXXO y hace mejoras locales.
        - **GRASP + LS:** Construcción greedy aleatorizada con búsqueda local.

        ### Restricciones
        """)
        restrictions = [
            ('R1 · Ancho', 'Suma de anchos por charola ≤ 55 cm.'),
            ('R2 · Alto', 'Alto del producto ≤ altura disponible del nivel. Por eso los grandes van abajo.'),
            ('R3 · Cobertura', 'Cada producto se coloca al menos 1 vez.'),
            ('R4 · Unicidad', 'Cada slot contiene a lo más 1 producto.'),
            ('R5-R8 · Familia y hibridismo', 'Detección de charolas que mezclan familias.'),
            ('R9 · Máximo híbridas', 'ε = 11 (calibrado al óptimo OXXO real).'),
            ('R10-R12 · Agrupación por columna (suave)', 'P1=cola, P2=cola, P3=sabor (del óptimo real).'),
            ('R13 · Binariedad', 'Variables binarias 0/1.'),
        ]
        for name, desc in restrictions:
            st.markdown(f'<div class="restriction-card"><div class="r-name">{name}</div><div class="r-desc">{desc}</div></div>',
                        unsafe_allow_html=True)
    st.stop()

# Si hay resultados, mostrar pestañas
res = st.session_state.resultados
optimo = st.session_state.optimo_df
refs = st.session_state.refs

MODEL_NAMES = {
    'pulp':   ('PuLP MIQP', 'Optimización exacta'),
    'sa':     ('Simulated Annealing', 'Metaheurística'),
    'grasp':  ('GRASP + LS', 'Metaheurística'),
    'optimo': ('Óptimo OXXO', 'Referencia oficial'),
}

# ── PESTAÑAS ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["PLANOGRAMA", "COINCIDENCIA", "EFECTIVIDAD"])


# ─── TAB 1: PLANOGRAMA ─────────────────────────────────────────────────────
with tab1:
    st.markdown("### Modelo Activo")
    modelo_sel = st.radio(
        "Seleccionar modelo:",
        ['optimo', 'pulp', 'sa', 'grasp'],
        format_func=lambda k: MODEL_NAMES[k][0],
        horizontal=True,
        label_visibility='collapsed',
    )

    # Obtener la solución según el modelo seleccionado
    if modelo_sel == 'optimo':
        # Usar el segmento BCO del óptimo
        bco = optimo[optimo['SEGMENTO_ID'] == 'BCO'].copy()
        # Expandir por NUM_FRENTES
        rows = []
        for (jx, jy), grp in bco.groupby(['jx', 'jy']):
            k = 0
            for _, r in grp.iterrows():
                for _ in range(int(r['NUM_FRENTES'])):
                    rows.append({
                        'item': r['ITEM'],
                        'desc': r['ITEM_DESC'].strip(),
                        'jx': jx, 'jy': jy, 'k': k,
                        'ancho': r['ANCHO'],
                        'familia': r['familia'],
                        'F': r['NUM_FRENTES'],
                    })
                    k += 1
        sol_df_sel = pd.DataFrame(rows)
        metrics_sel = {
            'pct_casa': 100.0, 'distancia_promedio': 0,
            'pct_fam_correcta': 100.0, 'jaccard_promedio': 100,
            'tiempo_s': 0, 'fitness': None, 'hibridas': None,
            'factible': True,
        }
    else:
        sol_df_sel = res[modelo_sel]['sol']
        metrics_sel = res[modelo_sel]['metrics']

    # Header info
    charolas_usadas = int(sol_df_sel.groupby(['jx', 'jy']).ngroups) if len(sol_df_sel) else 0
    coinc_str = f"{metrics_sel['pct_casa']}%" if modelo_sel != 'optimo' else '100% (referencia)'

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div style="margin: 12px 0;">
            <div style="font-size:22px;font-weight:800;color:var(--text);">{MODEL_NAMES[modelo_sel][0]}</div>
            <div style="font-size:11px;color:var(--muted);font-family:'JetBrains Mono',monospace;margin-top:4px;">
                3 puertas × 6 niveles &middot; {len(sol_df_sel)} frentes &middot; {charolas_usadas}/18 charolas &middot; coincidencia: {coinc_str}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="display:flex;gap:8px;justify-content:flex-end;align-items:center;height:100%;">
            <div style="background:#FFCC00;border:2px solid #E5B800;padding:6px 12px;border-radius:20px;
                font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;">
                Coincidencia: <strong style="color:#E60012;">{coinc_str}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Refrigerador
    st.markdown(render_fridge_html(sol_df_sel), unsafe_allow_html=True)


# ─── TAB 2: COINCIDENCIA ───────────────────────────────────────────────────
with tab2:
    # Cards con métricas de coincidencia (4 cards: optimo + 3 modelos)
    cols = st.columns(4)

    # Óptimo (referencia)
    n_prod_opt = optimo['ITEM'].nunique()
    n_frentes_opt = int(optimo[optimo['SEGMENTO_ID'] == 'BCO']['NUM_FRENTES'].sum())
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card optimo">
            <div class="metric-model-name">Óptimo OXXO</div>
            <div class="metric-model-type">Plan aprobado (BCO)</div>
            <div class="metric-big-num">100<span class="unit">%</span></div>
            <div class="metric-big-label">referencia (por definición)</div>
            <div class="metric-row"><span class="metric-row-label">Frentes</span>
                <span class="metric-row-value">{n_frentes_opt}</span></div>
            <div class="metric-row"><span class="metric-row-label">Charolas</span>
                <span class="metric-row-value">18/18</span></div>
            <div class="metric-row"><span class="metric-row-label">Productos</span>
                <span class="metric-row-value">{n_prod_opt}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # 3 modelos (mejor con highlight)
    pct_casa = {k: res[k]['metrics']['pct_casa'] for k in ['pulp', 'sa', 'grasp']}
    best_key = max(pct_casa, key=pct_casa.get)
    for idx, k in enumerate(['pulp', 'sa', 'grasp']):
        m = res[k]['metrics']
        sol_df = res[k]['sol']
        is_best = (k == best_key)
        with cols[idx + 1]:
            st.markdown(f"""
            <div class="metric-card{' highlight' if is_best else ''}">
                <div class="metric-model-name">{MODEL_NAMES[k][0]}</div>
                <div class="metric-model-type">{MODEL_NAMES[k][1]}</div>
                <div class="metric-big-num">{m['pct_casa']}<span class="unit">%</span></div>
                <div class="metric-big-label">productos en su casa óptima</div>
                <div class="metric-row"><span class="metric-row-label">Distancia promedio</span>
                    <span class="metric-row-value">{m['distancia_promedio']}</span></div>
                <div class="metric-row"><span class="metric-row-label">Familia correcta</span>
                    <span class="metric-row-value">{m['pct_fam_correcta']}%</span></div>
                <div class="metric-row"><span class="metric-row-label">Jaccard</span>
                    <span class="metric-row-value">{m['jaccard_promedio']}%</span></div>
                <div class="metric-row"><span class="metric-row-label">Tiempo</span>
                    <span class="metric-row-value">{m['tiempo_s']:.2f}s</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Barras comparativas
    def render_bars(title, sub, metric_key, formatter):
        values = {k: res[k]['metrics'][metric_key] for k in ['pulp', 'sa', 'grasp']}
        max_v = max(abs(v) for v in values.values()) or 1
        html = f"""<div class="comparison-section">
            <div class="section-title">{title}</div>
            <div class="section-sub">{sub}</div>"""
        for k in ['pulp', 'sa', 'grasp']:
            v = values[k]
            pct = abs(v) / max_v * 100
            html += f"""
            <div class="cmp-row">
                <div>
                    <div class="cmp-row-name">{MODEL_NAMES[k][0]}</div>
                    <div class="cmp-row-tag">{MODEL_NAMES[k][1]}</div>
                </div>
                <div class="cmp-bar-track">
                    <div class="cmp-bar-fill {k}" style="width:{pct}%">{formatter(v)}</div>
                </div>
                <div class="cmp-row-value">{formatter(v)}</div>
            </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    render_bars(
        "% en Casa Óptima (productos colocados donde OXXO los pone)",
        "Porcentaje de productos colocados exactamente en la (puerta, nivel) donde aparecen en los planogramas óptimos OXXO.",
        'pct_casa', lambda v: f"{v}%"
    )
    render_bars(
        "% Familia Correcta por Charola",
        "Charolas cuya familia dominante coincide con la familia dominante en el óptimo OXXO.",
        'pct_fam_correcta', lambda v: f"{v}%"
    )
    render_bars(
        "Distancia Promedio al Óptimo (menor = mejor)",
        "Distancia Manhattan promedio entre la posición del producto en el modelo y su casa en el óptimo OXXO. 0 = exacto.",
        'distancia_promedio', lambda v: f"{v:.2f}"
    )


# ─── TAB 3: EFECTIVIDAD ────────────────────────────────────────────────────
with tab3:
    # Tabla de efectividad con highlight automático
    rows_def = [
        ('% en casa óptima', 'pct_casa', lambda v: f"{v}%", True),
        ('Distancia promedio', 'distancia_promedio', lambda v: f"{v}", False),
        ('% familia correcta', 'pct_fam_correcta', lambda v: f"{v}%", True),
        ('Jaccard', 'jaccard_promedio', lambda v: f"{v}%", True),
        ('Frentes', None, lambda s, k: len(s[k]['sol']), True),
        ('Productos únicos', None, lambda s, k: s[k]['sol']['item'].nunique(), True),
        ('Híbridas', 'hibridas', lambda v: f"{v}", False),
        ('Tiempo (s)', 'tiempo_s', lambda v: f"{v:.2f}", False),
        ('Factible', 'factible', lambda v: "SI" if v else "NO", None),
    ]

    table_html = '<div class="comparison-section">'
    table_html += '<div class="section-title">Indicadores de Efectividad</div>'
    table_html += '<div class="section-sub">Métricas detalladas. Celda resaltada en amarillo = mejor en esa fila.</div>'
    table_html += '<table class="efe-table"><thead><tr><th>Indicador</th>'
    for k in ['pulp', 'sa', 'grasp']:
        table_html += f'<th>{MODEL_NAMES[k][0]}</th>'
    table_html += '</tr></thead><tbody>'

    for row in rows_def:
        label = row[0]
        higher_is_better = row[3]
        table_html += f'<tr><td>{label}</td>'

        # Calcular valores
        values = []
        for k in ['pulp', 'sa', 'grasp']:
            if row[1] is None:
                # Función especial que toma (res, modelo)
                v = row[2](res, k)
            else:
                v = res[k]['metrics'][row[1]]
            values.append(v)

        # Determinar el mejor índice
        best_idx = -1
        if higher_is_better is not None:
            numerics = []
            for v in values:
                try:
                    numerics.append(float(v))
                except (ValueError, TypeError):
                    numerics.append(1 if v is True else 0)
            best_idx = (numerics.index(max(numerics)) if higher_is_better
                        else numerics.index(min(numerics)))

        # Renderizar celdas
        for i, v in enumerate(values):
            cls = 'best' if i == best_idx else ''
            if row[1] is None:
                cell_value = str(v)
            else:
                cell_value = row[2](v)
            table_html += f'<td class="{cls}">{cell_value}</td>'
        table_html += '</tr>'

    table_html += '</tbody></table></div>'
    st.markdown(table_html, unsafe_allow_html=True)

    # Distribución por familia (modelo SA por default)
    st.markdown("""
    <div class="comparison-section">
        <div class="section-title">Distribución por Familia (Modelo SA)</div>
        <div class="section-sub">Cómo se reparten los frentes entre familias de bebidas.</div>
    """, unsafe_allow_html=True)

    sol_sa = res['sa']['sol']
    fam_counts = sol_sa.groupby('familia').size().to_dict()
    total = sum(fam_counts.values())
    fam_colors = {'cola': '#E60012', 'agua': '#1565C0', 'sabor': '#FF6F00'}

    fam_cards = '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px;">'
    for fam, cnt in sorted(fam_counts.items(), key=lambda x: -x[1]):
        pct = round(cnt / total * 100)
        col = fam_colors.get(fam, '#999')
        fam_cards += f"""
        <div style="padding:14px;border-radius:8px;border:2px solid {col};background:{col}10;text-align:center;">
            <div style="font-size:10px;font-weight:800;letter-spacing:0.15em;text-transform:uppercase;color:{col};margin-bottom:6px;">{fam}</div>
            <div style="font-size:28px;font-weight:800;line-height:1;color:{col};">{cnt}</div>
            <div style="font-size:11px;font-family:'JetBrains Mono',monospace;margin-top:4px;color:var(--muted);">{pct}% del total</div>
        </div>"""
    fam_cards += '</div></div>'
    st.markdown(fam_cards, unsafe_allow_html=True)

    # Top 10 productos
    st.markdown("""
    <div class="comparison-section">
        <div class="section-title">Top 10 Productos del Modelo SA</div>
        <div class="section-sub">Productos con mayor número de frentes en la solución.</div>
    """, unsafe_allow_html=True)

    top10 = sol_sa.groupby('desc').size().nlargest(10).reset_index(name='count')
    max_count = top10['count'].max() if len(top10) > 0 else 1

    top_html = '<ul style="list-style:none;padding:0;margin:0;">'
    for i, row in top10.iterrows():
        pct = round(row['count'] / max_count * 100)
        top_html += f"""
        <li style="display:grid;grid-template-columns:30px 1fr 100px 60px;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border);">
            <span style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:800;color:var(--oxxo-red);text-align:center;">{i+1}</span>
            <span style="font-size:12px;font-weight:600;">{row['desc']}</span>
            <div style="height:6px;background:var(--surface2);border-radius:3px;">
                <div style="height:100%;background:var(--oxxo-yellow);border-radius:3px;width:{pct}%;"></div>
            </div>
            <span style="text-align:right;font-family:'JetBrains Mono',monospace;font-weight:700;color:var(--oxxo-red);font-size:14px;">{row['count']}</span>
        </li>"""
    top_html += '</ul></div>'
    st.markdown(top_html, unsafe_allow_html=True)

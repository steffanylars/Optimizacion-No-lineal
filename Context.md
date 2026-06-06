# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Interactive Streamlit app that solves the OXXO refrigerator planogram (challenge MA2007B): assign beverage
products to an 18-charola fridge (3 doors × 6 levels) so the layout resembles an OXXO-approved reference.
It runs three solvers side by side — PuLP MIQP (exact), Simulated Annealing, and GRASP + Local Search — and
scores each by how closely it matches the reference. Code, comments, and UI are all in Spanish.

## Commands

```powershell
pip install -r requirements.txt      # streamlit, pandas, numpy, pulp (CBC ships with pulp)
streamlit run app.py                 # opens http://localhost:8501
```

- There are no tests, linter, or build step.
- `python lalo_metaheuristicas.py` runs a self-test of the metaheuristics engine, but its `__main__` block
  expects a local `ejemplo_planograma (1).csv` that is not in the repo, so it fails without that file.

### Temp files (cross-platform)
`app.py` and `modelos.py` write intermediate CSVs under `tempfile.gettempdir()` (`_hist_input.csv`,
`_opt_input.csv`, `_lalo.csv`), so the "CORRER LOS 3 MODELOS" flow runs on Windows and Streamlit Cloud alike.
In `app.py` the optimum/historical loads go through `@st.cache_data` helpers (`cargar_optimo_cached`,
`cargar_hist_filtrado_cached`) keyed by uploaded-file bytes. The two files differ in encoding (see below):
the optimum is read latin-1; the historical is read `utf-8-sig` with a latin-1 fallback.

## Runtime inputs (not committed)

The two CSVs are uploaded through the UI at runtime, never stored in the repo:
- **oxxo_1.csv** — historical planograms; drives SA/GRASP product attributes through the generic engine.
- **ejemplo_planograma.csv** — the OXXO-approved optimum; defines the objective and every reference metric.

Both are filtered everywhere to one segment: **PLANOGRUPO=Refrescos, MUEBLE_ID=CF, DIRECCION_LEGO_ID=DI,
TAMAÑO=3**. The two files have **different encodings**:
- **ejemplo_planograma.csv** is messy (UTF-8 BOM, but mixed bytes per column — `TAMAÑO_POST` is UTF-8 while
  `DISEÑO` is latin-1). `cargar_optimo`/`construir_problema` read it `encoding='latin-1'` (never fails on any
  byte) and normalize column names by ASCII substring (TAMA+POST→TAMANO_POST, SEGMENTO→SEGMENTO_ID, DISE→DISENO_REF).
- **oxxo_1.csv** is clean UTF-8 — its size column is literally `TAMAÑO`, so the historical loader reads it
  `utf-8-sig` (BOM-tolerant), falling back to latin-1. Reading it as latin-1 mangles `TAMAÑO` → KeyError.

## Architecture

Layered, with one important adapter seam:

- **app.py** — Streamlit UI + orchestration; heavy inline CSS for OXXO branding. All state lives in
  `st.session_state.resultados`. On button press it writes uploads to temp, calls `cargar_optimo` →
  `extraer_referencias`, then `correr_pulp` / `correr_sa` / `correr_grasp`, persists the run via
  `registro.registrar_corrida`, then renders four tabs (PLANOGRAMA / COINCIDENCIA / EFECTIVIDAD / REGISTROS).
- **registro.py** — persistence layer for the output table. Appends one record per solver (sharing one
  `id_solucion` per run) to `soluciones.json` next to the source files, so results survive across sessions.
  See "Output table (`registro.py`)" below.
- **geometria.py** — single source of truth for the fridge geometry: `NX=3, NY=6`,
  `ALTURAS=[42,42,31.5,31.5,28,25]`, `ANCHO_CHAROLA=55.0`, `K_DEFAULT=8`. No heavy deps, so both `modelos.py`
  and `visualizacion.py` import from it instead of each redefining the constants.
- **modelos.py** — the three solvers plus the adapter that bends the generic engine to OXXO geometry. Imports the
  geometry from `geometria.py` (`ALTURAS` aliased as `ALTURAS_NIVEL`). Also defines **`ParamsOXXO`**, the dataclass
  of UI-tunable params (`K`, `ancho_charola`, `demanda_global`, PuLP `top_n`/`time_limit`, SA/GRASP iters, `seed`,
  `epsilon`, `delta`); defaults reproduce the previous behavior. `K` is no longer a module global — it flows
  through `ParamsOXXO` → `Config(K=...)` → `prob.cfg.K` (and `params.K` in PuLP).
- **lalo_metaheuristicas.py** — standalone, reusable metaheuristics engine (`Config`, `Problema`, `Solucion`,
  `simulated_annealing`, `grasp`). Defaults to a *generic 5×6 fridge* (`NX=5, NY=6, NCH=30`) and a penalty-based
  `fitness`. By itself it knows nothing OXXO-specific.
- **visualizacion.py** — pure HTML/CSS render of the fridge (`render_fridge_html`); bottle color by sub-brand.
  Independent of the solvers.

### The adapter seam — read this before touching the solvers

`modelos.py::_setup_lalo()` **monkey-patches module globals** on `lalo_metaheuristicas` at runtime to turn the
generic 5×6 engine into the OXXO 3×6 layout: it overwrites `PL.NX/NY/NCH`, replaces `PL.charola_a_jxjy`, wraps
`PL.construir_problema` (injecting `rho_max=ALTURAS_NIVEL`, `col_fam=[0,0,2]`, a **flat `prob.alpha` = the chosen
`ancho_charola`** so SA's width cap matches PuLP/GRASP, and re-deriving `prob.fam`/`prob.C` with `familia_de_desc`
so solution families match the reference/`coincidencia`), and swaps `Solucion._pen_global`. `_setup_lalo(ancho_charola)`
takes the configured width. Consequences:
- The live geometry comes from **geometria.py** (imported by `modelos.py` and `visualizacion.py`), so there are no
  longer two copies to keep in sync. The `NX=5` in lalo is overridden and is *not* the live value at runtime.
- Editing lalo's module-level constants alone won't change app behavior — the patch wins.

### Two scoring functions — don't conflate them

- **Acceptance score** (`modelos.py::_ScorerIncremental`): a *similarity-to-OXXO* score,
  `en_casa*5 + fam_ok*2 + avg_jaccard*100`, evaluated incrementally (only recomputes the charolas a move
  touches via `update(grid, celdas)`). SA and GRASP optimize THIS.
- **Feasibility fitness** (`lalo Solucion.fitness`): objective minus R1–R13 constraint penalties. It is only read
  back via `sol.resumen()` to report `factible` / `hibridas`; the metaheuristics in modelos.py do not optimize it.

So SA/GRASP chase resemblance to the reference, while reported feasibility comes from the engine's constraint model.

**`metrics['fitness']` is homogeneous across all three solvers** = the engine penalized fitness. SA/GRASP get it
from `best.resumen()['fitness']`; **PuLP** post-evaluates its CBC solution through the same engine via
`modelos.py::_fitness_motor(sol_df, prob, item_to_idx)` (rebuilds a `Solucion` from the `sol_df`, mapping each
item to its engine index and skipping items outside `prob.items`). This is what feeds `desviacion_no_lineal` in
`registro.py`. Because the engine charges `pen_cobertura` (10000) per uncovered product of its historical-derived
universe, all three fitness values are dominated by coverage and are large/negative — that's expected and now
consistent. PuLP also keeps the raw CBC objective under `metrics['objetivo_lineal']` (not displayed anywhere).

### Reference objects (`modelos.py::extraer_referencias`)

From the optimum it precomputes `F_pos[(item,jx,jy)]` (frente counts = objective weights), `casa[item]` (best
cell per product), `fam_dict[(jx,jy)]` (dominant family per charola), and `prods_charola` (item set per cell).
`coincidencia()` scores any solution against these → `pct_casa`, `distancia_promedio` (Manhattan),
`pct_fam_correcta`, `jaccard_promedio`.

### Configurable params (`ParamsOXXO`) and the global demand

`app.py` builds a `ParamsOXXO` from an **"Parámetros avanzados"** expander above the run button and passes it to
all three solvers (`correr_pulp/sa/grasp(..., params)`). Defaults reproduce the prior behavior, so an untouched run
matches the old output. The headline lever is **`demanda_global` `D`** = *target total facings in the whole fridge*
(default = capacity `NX*NY*K`). All three solvers honor it the same way:
- **PuLP**: hard constraint `Σ n ≤ D` (clamped to `≥ |I|` so R3 coverage stays feasible).
- **GRASP**: construction stops once `D` facings are placed (its LS is swap-only, so it can't grow past `D`).
- **SA**: its insertion moves *would* fill to capacity (the acceptance score rewards matching facings), so SA tracks
  occupied cells and **rejects any move that pushes the count above `D`** (see `cur_occupied`/`delta_occ` in `correr_sa`).

### Solver specifics

- **PuLP** (`correr_pulp(hist_df, optimo, refs, params)`): exact MILP over the top-`pulp_top_n` products using
  integer count vars `n[i,jx,jy] ∈ {0..K}` (~810 vars; the **optimization** is **linear** — there is no δ·G synergy
  term). CBC with `timeLimit=params.pulp_time_limit`, `gapRel=0.05`. Constraints: width ≤`ancho_charola` per charola,
  ≤`K` frentes per charola, global `Σ n ≤ D`, product height vs `ALTURAS_NIVEL[jy]`, coverage 1–7 each. Extraction
  expands each `n>0` into that many rows (k correlativo). It takes `hist_df` only to build the engine `Problema`
  (`_setup_problema`) and report the homogeneous engine `fitness` for its CBC solution (see "Two scoring functions");
  the CBC optimization itself does not use `hist_df`.
- **SA** (`correr_sa`): warm-starts from a "cocktail" of the optimum's BCO+CLA+HRN segments
  (`_construir_cocktail`, itself capped at `D`); very low `T0=1.0`; move set biased toward placing products in their
  `casa`; respects the `D` cap as above.
- **GRASP** (`correr_grasp`): greedy-randomized fill per charola from the optimum's products (capped at `D`), then
  swap-only local search on the acceptance score.

In the PLANOGRAMA tab the "Óptimo OXXO" view is the optimum's BCO segment expanded by `NUM_FRENTES` and treated
as the 100% reference — it is not a solver output.

### Visualization (`visualizacion.py`)

`render_fridge_html(sol_df, ancho_charola, alturas)` draws the fridge to scale. Horizontal scale = `SHELF_W/ancho_charola`;
**bottle height is proportional to each product's real `ALTO`** (`alto*PX_PER_CM`, clamped to the shelf), so a 2L looks
taller than a can. Every `sol_df` (PuLP, SA/GRASP via `_extraer_sol`, and the óptimo view in `app.py`) now carries an
`'alto'` column for this; `sol_to_grid` reads it with a level-height fallback. `app.py` passes the run's `ancho_charola`
(stored in `st.session_state.params`) and `ALTURAS` into the render.

### Output table (`registro.py`)

A persistence layer that accumulates results **across sessions** in `soluciones.json`, written next to the
source files (`RUTA_JSON = dirname(__file__)/soluciones.json`). It is independent of the solvers (only depends on
`pandas`). Each "CORRER LOS 3 MODELOS" press calls `registrar_corrida(resultados)`, which loads the existing list,
computes the next `id_solucion` (`SOL-0001`, `SOL-0002`, … from the max existing suffix), **appends** one record
per model sharing that id, and saves. Tolerant of a missing/corrupt file (returns `[]`).

Columns (one record per solver): **`id_solucion · modelo · charolas · frentes · niveles · desviacion_no_lineal · timestamp`**.
Per model, from its `sol_df`/`metrics`: `charolas` = occupied `(jx,jy)` cells, `frentes` = rows of `sol_df`,
`niveles` = distinct `jy` occupied, `desviacion_no_lineal` = `metrics['fitness']` (the homogeneous engine fitness,
see "Two scoring functions"). `construir_registros` is the pure builder; `tabla_como_df` returns the table with
columns in canonical order.

`app.py` surfaces it in the **REGISTROS** tab (`render_registros()`: `st.dataframe` + CSV/JSON download buttons) and
also in an expander on the pre-run screen, so the persisted history is visible without re-running. `soluciones.json`
is a runtime data file — `.gitignore` only excludes `*.csv`/`.env`/`__pycache__/`, so it is committable; add it to
`.gitignore` if you'd rather not version it.

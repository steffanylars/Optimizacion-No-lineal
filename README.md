# Planograma OXXO - App Streamlit

App interactiva para resolver el planograma OXXO con tres modelos:
- **PuLP MIQP** (optimización exacta)
- **Simulated Annealing** (metaheurística)
- **GRASP + Local Search** (metaheurística)

## Cómo correrlo localmente

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Correr la app:
   ```bash
   streamlit run app.py
   ```

3. Se abrirá automáticamente en el navegador en `http://localhost:8501`

## Cómo usar la app

1. **Sube los dos CSV** desde la pantalla principal:
   - `oxxo_1.csv` (histórico de planogramas)
   - `ejemplo_planograma.csv` (óptimo aprobado por OXXO)

2. **Presiona "CORRER LOS 3 MODELOS"** — toma entre 5 y 60 segundos.

3. **Navega entre las tres pestañas:**
   - **PLANOGRAMA**: ve el refri renderizado para cada modelo, con switch entre Óptimo, PuLP, SA y GRASP. Hover en cada botella para ver detalles.
   - **COINCIDENCIA**: métricas de qué tanto se parece cada modelo al óptimo OXXO real.
   - **EFECTIVIDAD**: tabla comparativa con todas las métricas, distribución por familia y top 10 productos.

## Cómo desplegar en Streamlit Cloud (gratis)

1. Sube los archivos a un repo de GitHub.
2. Ve a https://share.streamlit.io y conecta tu cuenta.
3. "New app" → selecciona tu repo → `app.py` → Deploy.

## Estructura de archivos

```
planograma_streamlit/
├── app.py                       # App principal (Streamlit)
├── modelos.py                   # Lógica de los 3 modelos (PuLP, SA, GRASP)
├── visualizacion.py             # Renderizado HTML del refrigerador
├── lalo_metaheuristicas.py      # Código original de SA y GRASP (Lalo)
├── requirements.txt
└── README.md
```

## Notas técnicas

- El óptimo OXXO se filtra automáticamente a: **Refrescos · CF · DI · Tamaño 3**.
- La geometría del refri (18 charolas, 3 puertas × 6 niveles, alturas irregulares 42/42/31.5/31.5/28/25 cm) se infiere del archivo `ejemplo_planograma.csv`.
- F<sub>i,jx,jy</sub> = score basado en cuántas veces aparece el producto i en la posición (jx, jy) en el óptimo OXXO.
- El SA arranca con warm start desde el segmento BCO del óptimo (T₀=10, muy bajo).
- El PuLP usa los top 45 productos para mantener el modelo tratable (~6,500 variables).

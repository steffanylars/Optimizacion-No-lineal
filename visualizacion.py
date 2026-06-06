"""
Renderizado visual del refrigerador OXXO en HTML/CSS.
El refri se construye como string HTML y se inserta con st.markdown(unsafe_allow_html=True).
"""

from geometria import NX, NY, ALTURAS, ANCHO_CHAROLA


def get_color(desc, familia):
    """Color por sub-marca de bebida (no por familia general)"""
    n = (desc or '').lower()
    if 'cocacola light' in n or 'coca cola light' in n or 'cocacola sin' in n: return '#B0B0B0'
    if 'cocacola zero' in n or 'coca cola zero' in n: return '#1a1a1a'
    if 'cocacola' in n or 'coca cola' in n or 'coca-cola' in n: return '#E60012'
    if 'sprite zero' in n: return '#558B2F'
    if 'sprite' in n: return '#00A859'
    if 'fanta naranja' in n: return '#FF6F00'
    if 'fanta uva' in n: return '#6A1B9A'
    if 'fanta fresa' in n: return '#E91E63'
    if 'fanta toronja' in n: return '#FF5722'
    if 'fanta mandarina' in n or 'fanta limon' in n: return '#FFC107'
    if 'fanta' in n: return '#FF6F00'
    if 'pepsi black' in n: return '#000000'
    if 'pepsi light' in n or 'pepsi zero' in n: return '#0277BD'
    if 'pepsi' in n: return '#004B93'
    if '7up' in n or '7-up' in n or 'seven up' in n: return '#43A047'
    if 'mirinda' in n: return '#E64A19'
    if 'mundet' in n: return '#FF8F00'
    if 'manzanita' in n or 'manzana' in n: return '#33691E'
    if 'fresca' in n: return '#00897B'
    if 'aquarius' in n: return '#0288D1'
    if 'powerade' in n: return '#0D47A1'
    if 'gatorade' in n: return '#FF6F00'
    if 'jarritos' in n:
        if 'mandarina' in n: return '#FF6F00'
        if 'limon' in n or 'limón' in n: return '#CDDC39'
        if 'toronja' in n: return '#FF7043'
        if 'fresa' in n: return '#E53935'
        if 'tamarindo' in n: return '#5D4037'
        if 'mango' in n: return '#FFB300'
        if 'jamaica' in n: return '#AD1457'
        if 'uva' in n: return '#7B1FA2'
        if 'piña' in n or 'pina' in n: return '#F9A825'
        if 'sangria' in n or 'sangría' in n: return '#B71C1C'
        return '#BF360C'
    if 'boing' in n:
        if 'guayaba' in n: return '#689F38'
        if 'fresa' in n: return '#D32F2F'
        if 'mango' in n: return '#F57C00'
        if 'naranja' in n: return '#E64A19'
        if 'jamaica' in n: return '#AD1457'
        return '#2E7D32'
    if 'sangria' in n or 'sangría' in n or 'señorial' in n or 'senorial' in n: return '#B71C1C'
    if 'jumex' in n:
        if 'naranja' in n: return '#E65100'
        if 'mango' in n: return '#F9A825'
        if 'guayaba' in n: return '#558B2F'
        return '#F57F17'
    if 'red bull' in n: return '#1A237E'
    if 'monster' in n: return '#1B5E20'
    if 'electrolit' in n: return '#0097A7'
    if 'peñafiel' in n or 'penafiel' in n: return '#00838F'
    if 'ciel' in n: return '#4FC3F7'
    if 'mineral' in n: return '#0288D1'
    if 'agua' in n: return '#42A5F5'
    return {'cola': '#E60012', 'agua': '#1565C0', 'sabor': '#FF6F00'}.get(familia, '#666')


def sol_to_grid(sol_df):
    """Convierte el DataFrame de la solución a un grid {(jx,jy): [productos]}"""
    grid = {}
    for _, r in sol_df.iterrows():
        key = (int(r['jx']), int(r['jy']))
        av = r.get('alto')
        try:
            av = float(av)
            if av != av:        # NaN
                av = None
        except (TypeError, ValueError):
            av = None
        grid.setdefault(key, []).append({
            'nombre': str(r.get('desc', r.get('nombre', ''))).strip(),
            'ancho': float(r['ancho']),
            'alto': av,
            'familia': r['familia'],
            'k': int(r.get('k', 0)),
        })
    for k in grid:
        grid[k].sort(key=lambda x: x['k'])
    return grid


def render_fridge_html(sol_df, ancho_charola=ANCHO_CHAROLA, alturas=ALTURAS):
    """Genera el HTML del refrigerador a escala con todos los detalles.

    `ancho_charola` (cm) y `alturas` (cm por nivel) definen la escala física, de modo
    que el dibujo refleja la geometría configurada y el ALTO real de cada producto.
    """
    grid = sol_to_grid(sol_df)
    SHELF_W = 280
    ANCHO_SCALE = SHELF_W / float(ancho_charola)
    PX_PER_CM = 2.5

    css = """
    <style>
    .fridge-container {
        display: flex; justify-content: center; padding: 20px 0;
        background: var(--bg, #FFFBF0);
    }
    .fridge {
        background: linear-gradient(180deg, #E8E8E8 0%, #D0D0D0 100%);
        border-radius: 14px; padding: 14px 10px 10px;
        display: inline-flex; gap: 6px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15),
            inset 0 2px 0 rgba(255,255,255,0.5),
            inset 0 -2px 0 rgba(0,0,0,0.1);
    }
    .door {
        background: linear-gradient(180deg, #C8C8C8 0%, #B8B8B8 100%);
        border-radius: 8px; padding: 5px;
        display: flex; flex-direction: column; gap: 4px;
        box-shadow: inset 0 0 16px rgba(0,0,0,0.2), 0 2px 6px rgba(0,0,0,0.25);
        position: relative;
    }
    .door::before {
        content: ''; position: absolute; right: 8px; top: 50%;
        width: 4px; height: 50px; background: #5a5a5a;
        border-radius: 2px; transform: translateY(-50%);
    }
    .door-label {
        text-align: center; font-size: 9px; font-weight: 800;
        color: #555; letter-spacing: 0.15em; padding: 2px 0 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .shelf {
        background: #FAF7EC; border-radius: 4px;
        position: relative; overflow: hidden;
        border: 1px solid #E0DDC8;
        box-shadow: inset 0 -4px 8px rgba(0,0,0,0.05);
    }
    .shelf::after {
        content: ''; position: absolute;
        bottom: 0; left: 0; right: 0; height: 6px;
        background: linear-gradient(to bottom, #BBB, #888);
        box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10;
    }
    .shelf-led {
        position: absolute; top: 0; left: 4px; right: 4px; height: 3px;
        background: linear-gradient(to right, transparent, #FFE47A, #FFF5B0, #FFE47A, transparent);
        z-index: 10;
    }
    .products-row {
        display: flex; align-items: flex-end;
        height: calc(100% - 6px); padding: 4px 4px 0;
        gap: 1px; position: relative; z-index: 5; overflow: hidden;
    }
    .bottle {
        display: flex; flex-direction: column; align-items: center;
        position: relative; flex-shrink: 0;
        transition: transform .15s ease;
    }
    .bottle:hover { transform: translateY(-6px); z-index: 20; }
    .bottle:hover .tooltip { opacity: 1; transform: translateX(-50%) translateY(0); }
    .bottle-cap {
        border-radius: 3px 3px 0 0;
        background: linear-gradient(to bottom, #E8E8E8, #C0C0C0);
        border: 1px solid rgba(0,0,0,0.3); z-index: 2;
    }
    .bottle-neck { border-radius: 2px; opacity: 0.95; border: 1px solid rgba(0,0,0,0.2); border-bottom: none; }
    .bottle-body {
        border-radius: 4px 4px 0 0; position: relative; overflow: hidden;
        border: 1px solid rgba(0,0,0,0.25); border-bottom: none;
        box-shadow: inset 0 -8px 12px rgba(0,0,0,0.15);
    }
    .bottle-label {
        position: absolute; left: 50%; top: 42%;
        transform: translate(-50%, -50%);
        background: rgba(255,255,255,0.92); border-radius: 2px;
        padding: 2px 3px; text-align: center;
        font-size: 6px; font-weight: 700; color: #111;
        line-height: 1.15; max-width: 92%; overflow: hidden;
        pointer-events: none; border: 0.5px solid rgba(0,0,0,0.05);
    }
    .bottle-shine {
        position: absolute; top: 6%; left: 12%;
        width: 18%; height: 55%;
        background: linear-gradient(160deg, rgba(255,255,255,0.55), rgba(255,255,255,0));
        border-radius: 50%; pointer-events: none;
    }
    .tooltip {
        position: absolute; bottom: calc(100% + 10px);
        left: 50%; transform: translateX(-50%) translateY(4px);
        background: #1A1A1A; color: white;
        border-radius: 8px; padding: 10px 14px;
        font-size: 11px; white-space: nowrap;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        opacity: 0; pointer-events: none; transition: all .2s; z-index: 100;
    }
    .tooltip strong { display: block; font-weight: 700; font-size: 12px; margin-bottom: 4px; color: #FFCC00; }
    .tooltip em { color: #BBB; font-style: normal; font-family: 'JetBrains Mono', monospace; font-size: 10px; }
    .tooltip::after {
        content: ''; position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
        border: 5px solid transparent; border-top-color: #1A1A1A;
    }
    .nivel-tag {
        position: absolute; right: 4px; top: 6px;
        font-size: 8px; color: rgba(0,0,0,0.4);
        font-family: 'JetBrains Mono', monospace; font-weight: 700; z-index: 6;
    }
    .altura-tag {
        position: absolute; left: 4px; bottom: 8px;
        font-size: 7px; color: rgba(0,0,0,0.5);
        font-family: 'JetBrains Mono', monospace; z-index: 6;
    }
    </style>
    """

    html = css + '<div class="fridge-container"><div class="fridge">'

    for jx in range(NX):
        html += f'<div class="door"><div class="door-label">PUERTA {jx+1}</div>'

        for jy in range(NY - 1, -1, -1):
            shelf_h = alturas[jy] * PX_PER_CM
            html += f'<div class="shelf" style="width:{SHELF_W}px;height:{shelf_h}px">'
            html += '<div class="shelf-led"></div>'
            if jx == 0:
                html += f'<div class="nivel-tag">N.{jy+1}</div>'
                html += f'<div class="altura-tag">{alturas[jy]:.0f}cm</div>'

            html += '<div class="products-row">'
            prods = grid.get((jx, jy), [])
            inner_h = shelf_h - 6
            for p in prods:
                pw = max(10, int(p['ancho'] * ANCHO_SCALE))
                # Altura de la botella proporcional a su ALTO real (misma escala
                # vertical que la charola). Si falta el dato, se usa el alto del nivel.
                alto_cm = p.get('alto') or alturas[jy]
                bh = int(alto_cm * PX_PER_CM)
                bh = max(14, min(bh, int(inner_h - 2)))
                cap_h = int(bh * 0.08)
                neck_h = int(bh * 0.13)
                body_h = int(bh * 0.79)
                neck_w = int(pw * 0.4)
                color = get_color(p['nombre'], p['familia'])

                # Wrap del nombre por caracteres
                words = p['nombre'].split()
                chars_per_line = max(4, pw // 4)
                lines, line = [], ''
                for w in words:
                    test = (line + ' ' + w).strip()
                    if len(test) <= chars_per_line:
                        line = test
                    else:
                        if line:
                            lines.append(line)
                        line = w
                if line:
                    lines.append(line)
                lines = lines[:4]
                lines_html = '<br>'.join(lines)

                html += f'''<div class="bottle" style="width:{pw}px">
                    <div class="bottle-cap" style="width:{int(pw*0.3)}px;height:{cap_h}px"></div>
                    <div class="bottle-neck" style="width:{neck_w}px;height:{neck_h}px;background:{color}"></div>
                    <div class="bottle-body" style="width:{pw}px;height:{body_h}px;background:{color}">
                        <div class="bottle-shine"></div>
                        <div class="bottle-label">{lines_html}</div>
                    </div>
                    <div class="tooltip"><strong>{p['nombre']}</strong>
                    <em>{p['familia'].upper()} &middot; {p['ancho']}cm</em></div>
                </div>'''

            html += '</div></div>'  # products-row, shelf
        html += '</div>'  # door

    html += '</div></div>'  # fridge, container
    return html

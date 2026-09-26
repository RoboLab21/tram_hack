import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
from pathlib import Path

with open("events_data.json", "r", encoding="utf-8") as f:
    events = json.load(f)

json_str = json.dumps(events)

html_template = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Анализ проскальзываний и производных скорости</title>
    <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
    <style>
        body {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 16px;
        }
        .chart-box {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }
        svg {
            width: 100%;
            height: 200px;
            overflow: visible;
        }
        .btn-active {
            background-color: #0284c7;
            color: white;
            font-weight: 600;
        }
        .btn-inactive {
            background-color: #334155;
            color: #cbd5e1;
        }
    </style>
</head>
<body>
    <div class="max-w-5xl mx-auto">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
            <div>
                <h1 class="text-xl font-bold text-sky-400">Анализ моментов проскальзывания и производных скорости</h1>
                <p class="text-xs text-slate-400">Сопоставление скоростей колес v1, v2, ускорений a = dv/dt и ручки контроллера</p>
            </div>
            <div class="flex gap-2">
                <button id="btn-e1" onclick="selectEvent('event1')" class="px-3 py-1.5 rounded-lg text-xs transition btn-active">
                    ⚡ Событие 1: Пробуксовка (+26 км/ч)
                </button>
                <button id="btn-e2" onclick="selectEvent('event2')" class="px-3 py-1.5 rounded-lg text-xs transition btn-inactive">
                    🛑 Событие 2: Юз / Отказ тележки (-135 м)
                </button>
            </div>
        </div>

        <div id="event-banner" class="p-3 rounded-lg mb-4 text-xs flex justify-between items-center bg-red-950/60 border border-red-500/40 text-red-200">
            <span id="banner-text">Трамвай стоит на остановке (v_GNSS = 0), колёса раскручиваются тягой водителя (+11) до 26 км/ч.</span>
            <span id="banner-badge" class="px-2 py-0.5 rounded font-mono font-bold bg-red-500/30 text-red-300">ОШИБКА: +80.7 м</span>
        </div>

        <!-- 1. ГРАФИК СКОРОСТЕЙ -->
        <div class="chart-box">
            <div class="flex flex-wrap justify-between items-center mb-2 gap-2">
                <h2 class="text-sm font-semibold text-slate-200">1. Скорости колес и трамвая (км/ч)</h2>
                <div class="flex flex-wrap gap-4 text-xs font-mono">
                    <span class="flex items-center gap-1"><span class="w-3 h-0.5 bg-blue-400 inline-block"></span> v1 (передняя)</span>
                    <span class="flex items-center gap-1"><span class="w-3 h-0.5 bg-amber-400 inline-block"></span> v2 (задняя)</span>
                    <span class="flex items-center gap-1"><span class="w-3 h-0.5 bg-purple-400 inline-block"></span> V_ср</span>
                    <span class="flex items-center gap-1"><span class="w-3 h-0.5 bg-emerald-400 inline-block"></span> v_GNSS (истина)</span>
                </div>
            </div>
            <div class="relative">
                <svg id="svg-speed" viewBox="0 0 800 200"></svg>
            </div>
        </div>

        <!-- 2. ГРАФИК ПРОИЗВОДНЫХ (УСКОРЕНИЯ dv/dt) -->
        <div class="chart-box">
            <div class="flex flex-wrap justify-between items-center mb-2 gap-2">
                <h2 class="text-sm font-semibold text-slate-200">2. Производные скоростей — Ускорения a = dv/dt (м/с²)</h2>
                <div class="flex gap-4 text-xs font-mono">
                    <span class="flex items-center gap-1"><span class="w-3 h-0.5 bg-purple-400 inline-block"></span> a_колес (dv_одом/dt)</span>
                    <span class="flex items-center gap-1"><span class="w-3 h-0.5 bg-emerald-400 inline-block"></span> a_GNSS (истинное ускорение)</span>
                </div>
            </div>
            <div class="relative">
                <svg id="svg-acc" viewBox="0 0 800 180"></svg>
            </div>
            <p class="text-[11px] text-slate-400 mt-2">
                💡 <b>Физический маркер проскальзывания:</b> производная скорости колёс резко взлетает (до +2.5 м/с²), в то время как истинное ускорение трамвая a_GNSS остаётся строго нулевым!
            </p>
        </div>

        <!-- 3. ПРОСКАЛЬЗЫВАНИЕ И КОМАНДА ВОДИТЕЛЯ -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="chart-box !mb-0">
                <h2 class="text-sm font-semibold text-slate-200 mb-2">3. Величина проскальзывания (км/ч)</h2>
                <svg id="svg-slip" viewBox="0 0 400 130"></svg>
            </div>
            <div class="chart-box !mb-0">
                <h2 class="text-sm font-semibold text-slate-200 mb-2">4. Позиция контроллера водителя</h2>
                <svg id="svg-cmd" viewBox="0 0 400 130"></svg>
            </div>
        </div>
    </div>

    <script>
        const eventsData = """ + json_str + """;
        let currentEvent = 'event1';

        function selectEvent(evKey) {
            currentEvent = evKey;
            document.getElementById('btn-e1').className = (evKey === 'event1') ? 'px-3 py-1.5 rounded-lg text-xs transition btn-active' : 'px-3 py-1.5 rounded-lg text-xs transition btn-inactive';
            document.getElementById('btn-e2').className = (evKey === 'event2') ? 'px-3 py-1.5 rounded-lg text-xs transition btn-active' : 'px-3 py-1.5 rounded-lg text-xs transition btn-inactive';

            const banner = document.getElementById('event-banner');
            const text = document.getElementById('banner-text');
            const badge = document.getElementById('banner-badge');

            if (evKey === 'event1') {
                banner.className = 'p-3 rounded-lg mb-4 text-xs flex justify-between items-center bg-red-950/60 border border-red-500/40 text-red-200';
                text.innerText = 'Трамвай стоит на остановке (v_GNSS = 0), колёса раскручиваются тягой водителя (+11) до 26 км/ч в пробуксовке.';
                badge.className = 'px-2 py-0.5 rounded font-mono font-bold bg-red-500/30 text-red-300';
                badge.innerText = 'ОШИБКА: +80.7 м';
            } else {
                banner.className = 'p-3 rounded-lg mb-4 text-xs flex justify-between items-center bg-blue-950/60 border border-blue-500/40 text-blue-200';
                text.innerText = 'Задняя тележка v2 выпала в 0 км/ч на 30 секунд при реальном движении 17-25 км/ч. Скорость V_cp занижена в 2 раза.';
                badge.className = 'px-2 py-0.5 rounded font-mono font-bold bg-blue-500/30 text-blue-300';
                badge.innerText = 'ОШИБКА: -135 м';
            }
            renderCharts();
        }

        function buildPath(xArr, yArr, minX, maxX, minY, maxY, width, height, padY=20) {
            const padX = 40;
            const effW = width - padX * 2;
            const effH = height - padY * 2;
            let d = '';
            for (let i = 0; i < xArr.length; i++) {
                const px = padX + ((xArr[i] - minX) / (maxX - minX)) * effW;
                const py = height - padY - ((yArr[i] - minY) / (maxY - minY)) * effH;
                d += (i === 0 ? 'M' : 'L') + px.toFixed(1) + ',' + py.toFixed(1) + ' ';
            }
            return d;
        }

        function renderCharts() {
            const data = eventsData[currentEvent];
            const t = data.t;
            const minT = Math.min(...t);
            const maxT = Math.max(...t);

            // 1. Скорости
            const maxV = Math.max(...data.v1, ...data.v2, ...data.v_g, 15) * 1.15;
            const svgSpeed = document.getElementById('svg-speed');
            svgSpeed.innerHTML = `
                <line x1="40" y1="180" x2="760" y2="180" stroke="#334155" stroke-width="1" />
                <line x1="40" y1="100" x2="760" y2="100" stroke="#334155" stroke-width="0.5" stroke-dasharray="4" />
                <line x1="40" y1="20" x2="760" y2="20" stroke="#334155" stroke-width="0.5" stroke-dasharray="4" />
                <text x="32" y="183" fill="#64748b" font-size="10" text-anchor="end">0</text>
                <text x="32" y="103" fill="#64748b" font-size="10" text-anchor="end">${(maxV/2).toFixed(0)}</text>
                <text x="32" y="23" fill="#64748b" font-size="10" text-anchor="end">${maxV.toFixed(0)} км/ч</text>
                <path d="${buildPath(t, data.v1, minT, maxT, 0, maxV, 800, 200)}" fill="none" stroke="#60a5fa" stroke-width="2.2" />
                <path d="${buildPath(t, data.v2, minT, maxT, 0, maxV, 800, 200)}" fill="none" stroke="#fbbf24" stroke-width="2.2" />
                <path d="${buildPath(t, data.v_cp, minT, maxT, 0, maxV, 800, 200)}" fill="none" stroke="#c084fc" stroke-width="1.8" stroke-dasharray="4,4" />
                <path d="${buildPath(t, data.v_g, minT, maxT, 0, maxV, 800, 200)}" fill="none" stroke="#34d399" stroke-width="2.5" />
            `;

            // 2. Ускорения (dv/dt)
            const minA = Math.min(...data.a_wheels, ...data.a_gnss, -1.5) * 1.15;
            const maxA = Math.max(...data.a_wheels, ...data.a_gnss, 1.5) * 1.15;
            const svgAcc = document.getElementById('svg-acc');
            const zeroY = 180 - 20 - ((0 - minA) / (maxA - minA)) * 140;
            svgAcc.innerHTML = `
                <line x1="40" y1="${zeroY}" x2="760" y2="${zeroY}" stroke="#64748b" stroke-width="1" stroke-dasharray="2" />
                <text x="32" y="${zeroY+3}" fill="#94a3b8" font-size="10" text-anchor="end">0</text>
                <text x="32" y="23" fill="#64748b" font-size="10" text-anchor="end">+${maxA.toFixed(1)} м/с²</text>
                <text x="32" y="163" fill="#64748b" font-size="10" text-anchor="end">${minA.toFixed(1)}</text>
                <path d="${buildPath(t, data.a_wheels, minT, maxT, minA, maxA, 800, 180)}" fill="none" stroke="#c084fc" stroke-width="2" />
                <path d="${buildPath(t, data.a_gnss, minT, maxT, minA, maxA, 800, 180)}" fill="none" stroke="#34d399" stroke-width="2" stroke-dasharray="3,3" />
            `;

            // 3. Проскальзывание
            const minS = Math.min(...data.slip, -5) * 1.1;
            const maxS = Math.max(...data.slip, 5) * 1.1;
            const zeroYS = 130 - 15 - ((0 - minS) / (maxS - minS)) * 100;
            document.getElementById('svg-slip').innerHTML = `
                <line x1="30" y1="${zeroYS}" x2="380" y2="${zeroYS}" stroke="#64748b" stroke-width="1" />
                <path d="${buildPath(t, data.slip, minT, maxT, minS, maxS, 400, 130, 15)}" fill="none" stroke="#f43f5e" stroke-width="2.2" />
            `;

            // 4. Ручка водителя
            const minC = Math.min(...data.cmd, -10);
            const maxC = Math.max(...data.cmd, 12);
            const zeroYC = 130 - 15 - ((0 - minC) / (maxC - minC)) * 100;
            document.getElementById('svg-cmd').innerHTML = `
                <line x1="30" y1="${zeroYC}" x2="380" y2="${zeroYC}" stroke="#64748b" stroke-width="1" />
                <path d="${buildPath(t, data.cmd, minT, maxT, minC, maxC, 400, 130, 15)}" fill="none" stroke="#f59e0b" stroke-width="2" />
            `;
        }

        renderCharts();
    </script>
</body>
</html>
"""

target_file = Path("/home/eclipse/.gemini/antigravity/brain/50fa85e5-42d3-44a4-b1d3-9ad601380140/slip_analysis_widget.html")
target_file.write_text(html_template, encoding="utf-8")
print(f"Widget successfully written to {target_file}")

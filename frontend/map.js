/* ============================================================
   Perimetr — Mapa placu (top-down, multi-camera fusion)

   Rysuje widok z góry we wspólnym układzie metrycznym: prostokąt
   referencyjny z kalibracji, strefy world-space (_site), sfuzowane
   pozycje osób (kropka + liczba kamer, czerwona przy breachu).
   Dane: eventy "perimetr-frame" z app.js (world_persons/world_zones
   na każdej klatce) + /api/calibration dla skali.
   ============================================================ */
(function () {
    "use strict";

    const card = document.getElementById("siteMapCard");
    const canvas = document.getElementById("siteMapCanvas");
    const metaEl = document.getElementById("siteMapMeta");
    if (!card || !canvas) return;
    const ctx = canvas.getContext("2d");

    const PAD_M = 1.5;            // margines wokół sceny w metrach
    const PERSON_TTL_MS = 2000;   // znikanie kropki po utracie sygnału

    const Map = {
        refW: 3, refH: 3,         // prostokąt referencyjny (z kalibracji)
        cameras: [],
        zones: [],                // [{id, name, severity, polygon(m)}]
        persons: [],              // ostatni stan fuzji
        breachedIds: new Set(),   // fused_id z potwierdzonym breachem
        lastUpdate: 0,
    };

    async function loadCalibrations() {
        try {
            const r = await fetch("/api/calibration");
            if (!r.ok) return;
            const data = await r.json();
            Map.cameras = data.calibrations || [];
            if (Map.cameras.length) {
                Map.refW = Map.cameras[0].width_m;
                Map.refH = Map.cameras[0].height_m;
            }
        } catch (_) { /* ignore */ }
    }

    // ---------- geometria: metry -> piksele canvasa ----------

    function bounds() {
        let minX = 0, minY = 0, maxX = Map.refW, maxY = Map.refH;
        Map.zones.forEach(z => z.polygon.forEach(([x, y]) => {
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        }));
        Map.persons.forEach(p => {
            minX = Math.min(minX, p.x_m); maxX = Math.max(maxX, p.x_m);
            minY = Math.min(minY, p.y_m); maxY = Math.max(maxY, p.y_m);
        });
        return {
            minX: minX - PAD_M, minY: minY - PAD_M,
            maxX: maxX + PAD_M, maxY: maxY + PAD_M,
        };
    }

    function makeProjector(b) {
        const w = canvas.width, h = canvas.height;
        const scale = Math.min(w / (b.maxX - b.minX), h / (b.maxY - b.minY));
        const ox = (w - (b.maxX - b.minX) * scale) / 2;
        const oy = (h - (b.maxY - b.minY) * scale) / 2;
        return {
            scale,
            toPx: (x, y) => [ox + (x - b.minX) * scale,
                             oy + (y - b.minY) * scale],
        };
    }

    // ---------- render ----------

    function css(name, fallback) {
        const v = getComputedStyle(document.documentElement)
            .getPropertyValue(name).trim();
        return v || fallback;
    }

    function draw() {
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0) return;
        canvas.width = Math.round(rect.width);
        canvas.height = Math.round(rect.width * 0.62);

        const b = bounds();
        const proj = makeProjector(b);
        const red = css("--msbp-red", "#B03032");

        ctx.fillStyle = "#f7f6f4";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        drawGrid(b, proj);
        drawReferenceRect(proj);
        Map.zones.forEach(z => drawZone(z, proj, red));
        prunePersons();
        Map.persons.forEach(p => drawPerson(p, proj, red));
        drawScaleBar(proj);

        metaEl.textContent = Map.cameras.length + " kam · "
            + Map.persons.length + " os · " + Map.refW + "×" + Map.refH + " m";
    }

    function drawGrid(b, proj) {
        ctx.strokeStyle = "rgba(0,0,0,0.06)";
        ctx.lineWidth = 1;
        for (let x = Math.ceil(b.minX); x <= b.maxX; x++) {
            const [px] = proj.toPx(x, 0);
            ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, canvas.height); ctx.stroke();
        }
        for (let y = Math.ceil(b.minY); y <= b.maxY; y++) {
            const [, py] = proj.toPx(0, y);
            ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(canvas.width, py); ctx.stroke();
        }
    }

    function drawReferenceRect(proj) {
        const [x0, y0] = proj.toPx(0, 0);
        const [x1, y1] = proj.toPx(Map.refW, Map.refH);
        ctx.strokeStyle = "rgba(0,0,0,0.35)";
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(0,0,0,0.45)";
        ctx.font = "10px monospace";
        ctx.fillText("obszar referencyjny (markery)", x0 + 4, y0 - 4);
    }

    function drawZone(z, proj, red) {
        if (!z.polygon || z.polygon.length < 3) return;
        const danger = z.severity === "DANGER";
        const color = danger ? red : "#c99a2e";
        ctx.beginPath();
        z.polygon.forEach(([x, y], i) => {
            const [px, py] = proj.toPx(x, y);
            i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.globalAlpha = 0.15; ctx.fillStyle = color; ctx.fill();
        ctx.globalAlpha = 1; ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
        const [lx, ly] = proj.toPx(z.polygon[0][0], z.polygon[0][1]);
        ctx.fillStyle = color;
        ctx.font = "600 11px sans-serif";
        ctx.fillText(z.name, lx + 4, ly - 4);
    }

    function drawPerson(p, proj, red) {
        const [px, py] = proj.toPx(p.x_m, p.y_m);
        const breached = Map.breachedIds.has(p.fused_id);
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.fillStyle = breached ? red : "#2e7d32";
        ctx.fill();
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
        // badge: z ilu kamer sfuzowana pozycja
        const n = (p.cameras || []).length;
        if (n > 1) {
            ctx.beginPath();
            ctx.arc(px + 8, py - 8, 6, 0, Math.PI * 2);
            ctx.fillStyle = "#1a1a1a"; ctx.fill();
            ctx.fillStyle = "#fff"; ctx.font = "600 8px sans-serif";
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText(String(n), px + 8, py - 7.5);
            ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
        }
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.font = "10px monospace";
        ctx.fillText(p.x_m.toFixed(1) + ", " + p.y_m.toFixed(1) + " m",
                     px + 10, py + 4);
    }

    function drawScaleBar(proj) {
        const px1m = proj.scale;  // 1 m w pikselach
        const x = 12, y = canvas.height - 12;
        ctx.strokeStyle = "#1a1a1a"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + px1m, y); ctx.stroke();
        ctx.fillStyle = "#1a1a1a"; ctx.font = "10px monospace";
        ctx.fillText("1 m", x + px1m + 4, y + 3);
    }

    function prunePersons() {
        if (Date.now() - Map.lastUpdate > PERSON_TTL_MS) {
            Map.persons = [];
            Map.breachedIds.clear();
        }
    }

    // ---------- dane ----------

    document.addEventListener("perimetr-frame", (evt) => {
        const data = evt.detail || {};
        const zones = data.world_zones || [];
        const persons = data.world_persons || [];
        if (!zones.length && !persons.length && card.classList.contains("hidden")) {
            return;   // brak fuzji — karta zostaje schowana
        }
        card.classList.remove("hidden");
        if (data.calibration_active) {
            Map.zones = zones;
            Map.persons = persons;
            Map.lastUpdate = Date.now();
            (data.confirmed_world_breaches || []).forEach(bch => {
                Map.breachedIds.add(bch.person.fused_id);
                // breach gaśnie po 5 s — kropka wraca do zielonej
                setTimeout(() => Map.breachedIds.delete(bch.person.fused_id), 5000);
            });
        }
        draw();
    });

    window.addEventListener("resize", () => {
        if (!card.classList.contains("hidden")) draw();
    });

    loadCalibrations();
    // odśwież listę kamer co 30 s (nowe kalibracje pojawiają się bez reloadu)
    setInterval(loadCalibrations, 30000);
})();

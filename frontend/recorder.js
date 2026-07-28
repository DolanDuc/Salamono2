/* ============================================================
   Perimetr — panel nagrywania RTSP (recorder.js)
   Samodzielny moduł: lista kamer + przycisk Nagrywaj/Stop,
   status odpytany co 2 s. Nie zależy od app.js.
   ============================================================ */
(function () {
    "use strict";

    const listEl = document.getElementById("recorderList");
    const emptyEl = document.getElementById("recorderEmpty");
    const metaEl = document.getElementById("recorderMeta");
    if (!listEl) return;

    const overlapEl = document.getElementById("overlapBanner");
    let cameras = [];
    let statusByCam = {};   // camera_id -> { recording, elapsed_sec, size_bytes, filename }
    let liveByCam = {};     // camera_id -> { live, connected, frames, error }

    function fmtSize(bytes) {
        if (!bytes) return "0 MB";
        return (bytes / 1e6).toFixed(1) + " MB";
    }
    function fmtTime(sec) {
        sec = Math.floor(sec || 0);
        const m = Math.floor(sec / 60), s = sec % 60;
        return `${m}:${String(s).padStart(2, "0")}`;
    }

    async function api(path, opts) {
        const res = await fetch(path, opts);
        if (!res.ok) {
            let msg = res.statusText;
            try { msg = (await res.json()).detail || msg; } catch (_) {}
            throw new Error(msg);
        }
        return res.json();
    }

    async function loadCameras() {
        try {
            const data = await api("/api/recorder/cameras");
            cameras = data.cameras || [];
            if (!cameras.length) {
                emptyEl.textContent = "Brak kamer — dodaj je w data/cameras.json.";
                metaEl.textContent = "0 kamer";
                return;
            }
            render();
        } catch (e) {
            emptyEl.textContent = "Recorder niedostępny: " + e.message;
        }
    }

    async function refreshStatus() {
        if (!cameras.length) return;
        try {
            const [rec, det, ov] = await Promise.all([
                api("/api/recorder/status"),
                api("/api/detect/status"),
                api("/api/detect/overlap"),
            ]);
            const rmap = {};
            (rec.active || []).forEach((s) => { rmap[s.camera_id] = s; });
            statusByCam = rmap;
            const dmap = {};
            (det.active || []).forEach((s) => { dmap[s.camera_id] = s; });
            liveByCam = dmap;
            renderOverlap(ov.overlap);
            render();
        } catch (_) { /* serwer może wstawać — próbujemy dalej */ }
    }

    function renderOverlap(ov) {
        if (!overlapEl) return;
        if (ov && ov.cameras && ov.cameras.length >= 2) {
            overlapEl.textContent =
                `🔗 Kamery ${ov.cameras.join(" + ")} patrzą na ten sam punkt z różnych stron ` +
                `(fuzja ArUco potwierdzona — ${ov.persons} os. połączona)`;
            overlapEl.classList.remove("hidden");
        } else {
            overlapEl.classList.add("hidden");
        }
    }

    function currentMode() {
        // Tryb z przełącznika w nagłówku (Plac / Tylko strefy / Bramka).
        const active = document.querySelector("#modeSegmented button.active");
        return (active && active.dataset.mode) || "site";
    }

    async function toggleLive(cameraId) {
        const live = liveByCam[cameraId] && liveByCam[cameraId].live;
        const path = live ? "/api/detect/stop" : "/api/detect/start";
        try {
            await api(path, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ camera_id: cameraId, mode: currentMode() }),
            });
        } catch (e) {
            alert((live ? "Nie udało się zatrzymać detekcji: " : "Nie udało się włączyć detekcji: ") + e.message);
        }
        await refreshStatus();
    }

    const probeByCam = {};  // camera_id -> tekst wyniku testu

    async function testConn(cameraId) {
        probeByCam[cameraId] = "⏳ testuję…";
        render();
        try {
            const r = await api("/api/recorder/test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ camera_id: cameraId }),
            });
            if (r.reachable) {
                const res = r.width ? `${r.width}×${r.height}` : "?";
                const fps = r.fps ? ` · ${r.fps} kl/s` : "";
                const codec = r.codec ? ` · ${r.codec}` : "";
                probeByCam[cameraId] = `✅ podłączona · ${res}${fps}${codec}`;
            } else {
                probeByCam[cameraId] = "❌ brak połączenia: " + (r.error || "—");
            }
        } catch (e) {
            probeByCam[cameraId] = "❌ błąd testu: " + e.message;
        }
        render();
    }

    async function toggle(cameraId) {
        const rec = statusByCam[cameraId] && statusByCam[cameraId].recording;
        const path = rec ? "/api/recorder/stop" : "/api/recorder/start";
        try {
            await api(path, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ camera_id: cameraId }),
            });
        } catch (e) {
            alert((rec ? "Nie udało się zatrzymać: " : "Nie udało się nagrać: ") + e.message);
        }
        await refreshStatus();
    }

    function render() {
        if (emptyEl) emptyEl.remove();
        const recCount = Object.values(statusByCam).filter((s) => s.recording).length;
        const liveCount = Object.values(liveByCam).filter((s) => s.live).length;
        const tags = [];
        if (liveCount) tags.push(`👁 ${liveCount} live`);
        if (recCount) tags.push(`● ${recCount} nagrywa`);
        metaEl.textContent = tags.length ? tags.join(" · ") : `${cameras.length} kamer`;

        listEl.innerHTML = "";
        cameras.forEach((cam) => {
            const s = statusByCam[cam.id];
            const recording = s && s.recording;

            const row = document.createElement("div");
            row.className = "px-recorder-row" + (recording ? " is-recording" : "");

            const info = document.createElement("div");
            info.className = "px-recorder-info";
            const name = document.createElement("div");
            name.className = "px-recorder-name";
            name.innerHTML = `<span class="px-status-dot"></span>${cam.name}`;
            const live = liveByCam[cam.id] && liveByCam[cam.id].live;

            const meta = document.createElement("div");
            meta.className = "px-recorder-sub";
            const bits = [];
            if (recording) bits.push(`⏺ ${fmtTime(s.elapsed_sec)} · ${fmtSize(s.size_bytes)}`);
            if (live) {
                const l = liveByCam[cam.id];
                bits.push(l.connected ? `👁 live · ${l.frames} kl.` : "👁 live · łączenie…");
                if (l.error) bits.push(l.error);
            }
            if (!bits.length) bits.push(probeByCam[cam.id] || cam.rtsp_url_masked);
            meta.textContent = bits.join(" · ");
            info.appendChild(name);
            info.appendChild(meta);

            const actions = document.createElement("div");
            actions.className = "px-recorder-actions";

            const testBtn = document.createElement("button");
            testBtn.type = "button";
            testBtn.className = "px-btn";
            testBtn.textContent = "Testuj";
            testBtn.title = "Sprawdź czy kamera jest podłączona po Ethernecie";
            testBtn.disabled = recording || live;
            testBtn.addEventListener("click", () => testConn(cam.id));

            const liveBtn = document.createElement("button");
            liveBtn.type = "button";
            liveBtn.className = "px-btn" + (live ? " px-btn--solid" : "");
            liveBtn.textContent = live ? "■ Live" : "👁 Live";
            liveBtn.title = "Detekcja + kalibracja ArUco na żywo z tej kamery";
            liveBtn.addEventListener("click", () => toggleLive(cam.id));

            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "px-btn" + (recording ? " px-btn--solid" : "");
            btn.textContent = recording ? "■ Stop" : "● Nagrywaj";
            btn.addEventListener("click", () => toggle(cam.id));

            actions.appendChild(testBtn);
            actions.appendChild(liveBtn);
            actions.appendChild(btn);
            row.appendChild(info);
            row.appendChild(actions);
            listEl.appendChild(row);
        });
    }

    loadCameras().then(refreshStatus);
    setInterval(refreshStatus, 2000);
})();

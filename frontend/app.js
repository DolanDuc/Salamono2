/* ============================================================
   Perimetr — Panel kierownika, app runtime
   ============================================================ */
(function () {
    "use strict";

    // ---------- State store (prosty pub/sub) -----------------------

    const State = {
        wsStatus: "connecting",
        stats: { fps: 0, detectionMs: 0, frameCount: 0 },
        mode: "site",           // "site" | "checkpoint"
        layers: loadLayers(),   // { boxes, zones, markers, distances }
        activeAlarm: null,      // { title, meta, ts, source }
        alarmMuteUntil: 0,
        alertsCount: 0,
        alertsFilter: "",       // "" | "DANGER" | "WARNING"
        lastMarkerCount: 0,
        lastCalibrationActive: false,
        lastCameraSize: "—",
    };

    function loadLayers() {
        try {
            const raw = localStorage.getItem("perimetr:layers");
            if (raw) {
                const parsed = JSON.parse(raw);
                return {
                    boxes: parsed.boxes !== false,
                    zones: parsed.zones !== false,
                    markers: parsed.markers !== false,
                    distances: parsed.distances !== false,
                };
            }
        } catch (_) { /* ignore */ }
        return { boxes: true, zones: true, markers: true, distances: true };
    }
    function saveLayers() {
        try { localStorage.setItem("perimetr:layers", JSON.stringify(State.layers)); } catch (_) { }
    }

    // ---------- DOM refs ----------------------------------------

    const canvas = document.getElementById("liveCanvas");
    const ctx = canvas.getContext("2d");
    const noSignal = document.getElementById("noSignal");
    const wsStatusEl = document.getElementById("wsStatus");
    const connText = document.getElementById("connectionText");
    const connWrap = document.getElementById("connectionStatus");
    const frameCountEl = document.getElementById("frameCount");
    const processingTimeEl = document.getElementById("processingTime");
    const fpsLine = document.getElementById("fpsLine");
    const camMeta = document.getElementById("camMeta");
    const liveText = document.getElementById("liveText");
    const liveChip = document.getElementById("liveChip");
    const markerChip = document.getElementById("markerChip");
    const videoEl = document.getElementById("canvasStack");
    const videoStatus = document.getElementById("videoStatus");
    const alertList = document.getElementById("alertList");
    const alertsEmpty = document.getElementById("alertsEmpty");
    const alertTotal = document.getElementById("alertTotal");
    const alarmsFilters = document.getElementById("alarmsFilters");
    const alarmBanner = document.getElementById("alarmBanner");
    const alarmTitle = document.getElementById("alarmTitle");
    const alarmMeta = document.getElementById("alarmMeta");
    const alarmAckBtn = document.getElementById("alarmAckBtn");
    const alarmMuteBtn = document.getElementById("alarmMuteBtn");
    const headerAlarmTag = document.getElementById("headerAlarmTag");
    const snapshotBtn = document.getElementById("snapshotBtn");
    const pairPhoneBtn = document.getElementById("pairPhoneBtn");
    const modeSegmented = document.getElementById("modeSegmented");
    const ppeBadge = document.getElementById("ppeBadge");
    const ppeHeadline = document.getElementById("ppeHeadline");
    const ppeHardhat = document.getElementById("ppeHardhat");
    const ppeVest = document.getElementById("ppeVest");
    const layerTags = document.querySelectorAll("[data-layer]");

    // ---------- Init: layer buttons ---------------------------

    function syncLayerButtons() {
        layerTags.forEach(el => {
            const on = !!State.layers[el.dataset.layer];
            el.classList.toggle("msbp-tag--solid", on);
            el.classList.toggle("msbp-tag--outline", !on);
            if (!on) {
                el.style.color = "var(--ink-3)";
                el.style.borderColor = "var(--border-2)";
            } else {
                el.style.color = "";
                el.style.borderColor = "";
            }
        });
    }
    layerTags.forEach(el => {
        el.addEventListener("click", () => {
            const key = el.dataset.layer;
            State.layers[key] = !State.layers[key];
            saveLayers();
            syncLayerButtons();
            document.dispatchEvent(new CustomEvent("perimetr-layers", { detail: State.layers }));
        });
    });
    syncLayerButtons();

    // ---------- Init: mode segmented ---------------------------

    modeSegmented.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
            State.mode = btn.dataset.mode;
            modeSegmented.querySelectorAll("button").forEach(b => {
                b.classList.toggle("active", b === btn);
            });
            // hide PPE badge when leaving checkpoint
            if (State.mode !== "checkpoint") ppeBadge.classList.add("hidden");
            document.dispatchEvent(new CustomEvent("perimetr-mode", { detail: State.mode }));
        });
    });

    // ---------- Init: alarms filter ---------------------------

    alarmsFilters.querySelectorAll("[data-filter]").forEach(el => {
        el.addEventListener("click", () => {
            State.alertsFilter = el.dataset.filter;
            alarmsFilters.querySelectorAll("[data-filter]").forEach(x => {
                const on = x === el;
                x.classList.toggle("msbp-tag--solid", on);
                x.classList.toggle("msbp-tag--outline", !on);
                x.classList.toggle("active", on);
                if (!on) {
                    x.style.color = "var(--ink-3)";
                    x.style.borderColor = "var(--border-2)";
                } else {
                    x.style.color = "";
                    x.style.borderColor = "";
                }
            });
            filterAlertsUI();
        });
    });

    function filterAlertsUI() {
        const rows = alertList.querySelectorAll(".px-alert-row");
        let visible = 0;
        rows.forEach(row => {
            const sev = row.dataset.severity;
            const show = !State.alertsFilter || sev === State.alertsFilter;
            row.style.display = show ? "" : "none";
            if (show) visible++;
        });
        alertsEmpty.style.display = visible === 0 ? "" : "none";
    }

    // ---------- Alarm banner controls ----------------------------

    alarmAckBtn.addEventListener("click", () => clearAlarm());
    alarmMuteBtn.addEventListener("click", () => {
        State.alarmMuteUntil = Date.now() + 5 * 60 * 1000;
        clearAlarm();
    });

    function raiseAlarm(kind, description, meta) {
        if (Date.now() < State.alarmMuteUntil) return;
        State.activeAlarm = { kind, description, meta, ts: Date.now() };
        alarmTitle.textContent = description || "STOP — Naruszenie";
        alarmMeta.textContent = meta;
        alarmBanner.classList.remove("hidden");
        headerAlarmTag.classList.remove("hidden");
        videoEl.classList.add("alarm");
    }
    function clearAlarm() {
        State.activeAlarm = null;
        alarmBanner.classList.add("hidden");
        headerAlarmTag.classList.add("hidden");
        videoEl.classList.remove("alarm");
    }

    // ---------- Snapshot + phone pair ---------------------------

    snapshotBtn.addEventListener("click", () => {
        if (!canvas.width || !canvas.height) return;
        const url = canvas.toDataURL("image/png");
        const a = document.createElement("a");
        a.href = url;
        const ts = new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
        a.download = `perimetr-${ts}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    // ---------- Phone pair QR modal ----------------------------
    const pairModal = document.getElementById("pairModal");
    const pairModalClose = document.getElementById("pairModalClose");
    const pairQrImg = document.getElementById("pairQrImg");
    const pairUrlInput = document.getElementById("pairUrl");
    const pairCopyBtn = document.getElementById("pairCopyBtn");

    // Capture URL points the phone back at THIS server. `server` param makes
    // the phone POST frames here even when it opened the page from a QR that
    // was scanned off a different-origin screen (iframe pitch embed).
    function captureUrl() {
        const u = new URL("/phone/capture.html", location.origin);
        u.searchParams.set("server", location.origin);
        return u.toString();
    }

    function openPairModal() {
        const url = captureUrl();
        pairUrlInput.value = url;
        // QR rendered server-side (SVG) — no CDN dependency for the demo.
        pairQrImg.src = "/api/pair-qr?target=" + encodeURIComponent(url);
        pairModal.classList.remove("hidden");
    }
    function closePairModal() {
        pairModal.classList.add("hidden");
    }

    pairPhoneBtn.addEventListener("click", openPairModal);
    pairModalClose.addEventListener("click", closePairModal);
    pairModal.addEventListener("click", (e) => {
        if (e.target === pairModal) closePairModal();  // click backdrop
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !pairModal.classList.contains("hidden")) closePairModal();
    });
    pairCopyBtn.addEventListener("click", () => {
        const done = () => {
            const prev = pairCopyBtn.textContent;
            pairCopyBtn.textContent = "Skopiowano";
            setTimeout(() => { pairCopyBtn.textContent = prev; }, 1500);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(pairUrlInput.value).then(done, () => {
                pairUrlInput.select(); document.execCommand("copy"); done();
            });
        } else {
            pairUrlInput.select(); document.execCommand("copy"); done();
        }
    });

    // ---------- WebSocket ----------------------------------

    let ws = null;
    let lastRenderedTs = 0;

    function setWsStatus(state) {
        State.wsStatus = state;
        const labels = { connecting: "łączenie", connected: "połączono", error: "błąd" };
        wsStatusEl.textContent = labels[state] || state;
        connText.textContent = labels[state] || state;
        connWrap.classList.toggle("err", state !== "connected");
        if (state !== "connected") {
            liveText.textContent = "OFFLINE";
        }
    }

    function connectWebSocket() {
        setWsStatus("connecting");
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(proto + "//" + location.host + "/ws/live");
        ws.onopen  = () => setWsStatus("connected");
        ws.onerror = () => { setWsStatus("error"); try { ws.close(); } catch (_) {} };
        ws.onclose = () => { setWsStatus("error"); setTimeout(connectWebSocket, 2000); };
        ws.onmessage = onWsMessage;
    }

    function onWsMessage(evt) {
        let data; try { data = JSON.parse(evt.data); } catch (_) { return; }
        const ts = data.timestamp || 0;
        if (ts && ts < lastRenderedTs) return;
        lastRenderedTs = ts;

        renderFrame(data);
        updateStats(data);
        updateStatusChips(data);
        updateAlerts(data);
        updatePPE(data);
        maybeRaiseAlarm(data);

        // For zones.js — draws marker-zone polygons and other overlays
        document.dispatchEvent(new CustomEvent("perimetr-frame", { detail: data }));
    }

    // ---------- Frame render ---------------------------------

    function renderFrame(data) {
        if (!data.frame_jpeg_b64) return;
        noSignal.classList.add("hidden");
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            State.lastCameraSize = img.width + "×" + img.height;
        };
        img.src = "data:image/jpeg;base64," + data.frame_jpeg_b64;
    }

    // ---------- Stats + chips ------------------------------

    let fpsFrames = 0, fpsLastTime = performance.now();
    function updateStats(data) {
        State.stats.frameCount = data.frame_id || (State.stats.frameCount + 1);
        State.stats.detectionMs = data.processing_ms || 0;
        frameCountEl.textContent = State.stats.frameCount;
        processingTimeEl.textContent = Math.round(State.stats.detectionMs);
        fpsFrames++;
        const now = performance.now();
        const elapsed = now - fpsLastTime;
        if (elapsed >= 1000) {
            const fps = (fpsFrames / elapsed) * 1000;
            State.stats.fps = fps;
            fpsLine.textContent = fps.toFixed(1) + " kl/s";
            camMeta.textContent = State.lastCameraSize + " · " + fps.toFixed(1) + " kl/s";
            const shortText = fps.toFixed(1) + " kl/s · " + Math.round(State.stats.detectionMs) + " ms";
            connText.textContent = "Połączono · " + shortText;
            fpsFrames = 0;
            fpsLastTime = now;
        }
    }

    function updateStatusChips(data) {
        const time = data.timestamp ?
            new Date(data.timestamp * 1000).toLocaleTimeString("pl-PL") : "—";
        const activeAlarm = !!State.activeAlarm;
        const cam = data.camera_id || "KAM-01";
        liveText.textContent = (activeAlarm ? "ALARM" : "LIVE") + " · " + cam + " · " + time;

        const markers = (data.markers || []).length;
        State.lastMarkerCount = markers;
        State.lastCalibrationActive = !!data.calibration_active;
        const homo = data.calibration_active ? "homografia OK" : "brak homografii";
        markerChip.textContent = "ArUco " + markers + " · " + homo;
    }

    // ---------- Alerts list --------------------------------

    const MAX_ALERTS_IN_UI = 100;

    function updateAlerts(data) {
        const items = [];
        (data.confirmed_zone_breaches || []).forEach(b => items.push({
            severity: b.severity,
            time: b.timestamp,
            desc: "Wejście w strefę: " + (b.zone_name || "?"),
            kind: "zone_breach",
            thumb: b.frame_thumbnail_url,
        }));
        (data.confirmed_alerts || []).forEach(a => items.push({
            severity: a.severity,
            time: a.timestamp,
            desc: a.rule_name === "person_vehicle_overlap"
                ? "Osoba w strefie pojazdu"
                : "Osoba blisko pojazdu",
            kind: "site_hazard",
            thumb: a.frame_thumbnail_url,
        }));
        (data.ppe_checks || []).forEach(c => {
            if (c.severity !== "DANGER" || !c.missing || !c.missing.length) return;
            const parts = c.missing.map(m => m === "hardhat" ? "kaska" : (m === "vest" ? "kamizelki" : m));
            items.push({
                severity: "DANGER",
                time: c.timestamp,
                desc: "Brak PPE: " + parts.join(" + "),
                kind: "ppe_missing",
                thumb: c.frame_thumbnail_url,
            });
        });
        items.forEach(prependAlert);
    }

    function prependAlert(a) {
        const row = document.createElement("div");
        row.className = "px-alert-row " + (a.severity === "WARNING" ? "warning" : "danger");
        row.dataset.severity = a.severity;

        const thumb = document.createElement("div");
        thumb.className = "px-alert-thumb";
        if (a.thumb) thumb.style.backgroundImage = "url(" + a.thumb + ")";
        else         thumb.textContent = "KAM-01";

        const body = document.createElement("div");
        body.className = "px-alert-body";
        const line = document.createElement("div");
        line.className = "px-alert-line";
        const tag = document.createElement("span");
        tag.className = "msbp-tag " + (a.severity === "WARNING" ? "msbp-tag--warning" : "msbp-tag--red");
        tag.textContent = a.severity === "WARNING" ? "Warning" : "Danger";
        const desc = document.createElement("span");
        desc.className = "px-alert-desc";
        desc.textContent = a.desc;
        line.appendChild(tag);
        line.appendChild(desc);
        const meta = document.createElement("span");
        meta.className = "px-alert-meta";
        const t = a.time ? new Date(a.time * 1000).toLocaleTimeString("pl-PL") : "—";
        meta.textContent = t + " · " + a.kind;
        body.appendChild(line);
        body.appendChild(meta);

        row.appendChild(thumb);
        row.appendChild(body);
        alertsEmpty.style.display = "none";
        // insert on top; alertsEmpty is kept at the end and hidden when there are rows
        alertList.insertBefore(row, alertList.firstChild);
        const rows = alertList.querySelectorAll(".px-alert-row");
        if (rows.length > MAX_ALERTS_IN_UI) rows[rows.length - 1].remove();
        State.alertsCount++;
        alertTotal.textContent = State.alertsCount;
        filterAlertsUI();
    }

    // ---------- PPE badge (state 1d) --------------------

    let ppeHideTimeout = null;
    function updatePPE(data) {
        if (State.mode !== "checkpoint") {
            ppeBadge.classList.add("hidden");
            return;
        }
        const checks = data.ppe_checks || [];
        if (!checks.length) return;
        const c = checks[0];
        const ok = c.severity === "OK";
        ppeBadge.classList.remove("hidden", "ok", "fail");
        ppeBadge.classList.add(ok ? "ok" : "fail");
        ppeHeadline.textContent = ok
            ? "PPE OK"
            : "Brak: " + c.missing.map(x => x.toUpperCase()).join(" + ");
        setPpeItem(ppeHardhat, c.has_hardhat);
        setPpeItem(ppeVest, c.has_vest);
        if (ppeHideTimeout) clearTimeout(ppeHideTimeout);
        ppeHideTimeout = setTimeout(() => { ppeBadge.classList.add("hidden"); }, 6000);
    }
    function setPpeItem(el, ok) {
        el.classList.remove("ok", "fail");
        el.classList.add(ok ? "ok" : "fail");
        el.querySelector(".px-ppe-mark").textContent = ok ? "✓" : "✗";
    }

    // ---------- Alarm banner (state 1c) trigger ------------

    function maybeRaiseAlarm(data) {
        if (State.activeAlarm) return;

        const zoneBreach = (data.confirmed_zone_breaches || []).find(b => b.severity === "DANGER");
        if (zoneBreach) {
            const t = zoneBreach.timestamp ?
                new Date(zoneBreach.timestamp * 1000).toLocaleTimeString("pl-PL") : "—";
            raiseAlarm(
                "zone_breach",
                "STOP — Osoba w strefie: " + (zoneBreach.zone_name || "?"),
                t + " · KAM-01 · reguła zone_breach"
            );
            return;
        }

        const siteHazard = (data.confirmed_alerts || []).find(a => a.severity === "DANGER");
        if (siteHazard) {
            const t = siteHazard.timestamp ?
                new Date(siteHazard.timestamp * 1000).toLocaleTimeString("pl-PL") : "—";
            raiseAlarm(
                "site_hazard",
                "STOP — Osoba w strefie pojazdu",
                t + " · KAM-01 · reguła " + siteHazard.rule_name
            );
            return;
        }

        const ppeFail = (data.ppe_checks || []).find(c => c.severity === "DANGER" && c.missing && c.missing.length);
        if (ppeFail && State.mode === "checkpoint") {
            const missing = ppeFail.missing.map(m => m === "hardhat" ? "kaska" : (m === "vest" ? "kamizelki" : m));
            const t = ppeFail.timestamp ?
                new Date(ppeFail.timestamp * 1000).toLocaleTimeString("pl-PL") : "—";
            raiseAlarm(
                "ppe_missing",
                "BRAK " + missing.join(" + ").toUpperCase() + " — WEJŚCIE WSTRZYMANE",
                t + " · KAM-01 · bramka główna · komunikat głosowy odtworzony"
            );
        }
    }

    // ---------- Bootstrap ---------------------------

    filterAlertsUI();
    connectWebSocket();

})();

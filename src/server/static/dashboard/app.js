(function () {
  const STORAGE_KEY = "bil_server_dashboard_targets_v1";
  const POLL_INTERVAL_MS = 15000;
  const WS_RECONNECT_MS = 2500;
  const LIVE_REFRESH_DEBOUNCE_MS = 600;

  class TargetStore {
    constructor() {
      this.targets = [];
      this.activeId = null;
      this.load();
      if (this.targets.length === 0) {
        this.add({
          name: "Local Server 8000",
          protocol: "http",
          host: "127.0.0.1",
          port: "8000",
          notes: "Default local target",
        });
      }
      if (!this.activeId && this.targets[0]) {
        this.activeId = this.targets[0].id;
      }
    }

    load() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) {
          return;
        }
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed.targets)) {
          return;
        }
        this.targets = parsed.targets;
        this.activeId = parsed.activeId || null;
      } catch (_err) {
        this.targets = [];
        this.activeId = null;
      }
    }

    save() {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          targets: this.targets,
          activeId: this.activeId,
        })
      );
    }

    add(targetInput) {
      const item = {
        id: this._newId(),
        name: targetInput.name.trim(),
        protocol: targetInput.protocol,
        host: targetInput.host.trim(),
        port: String(targetInput.port).trim(),
        notes: (targetInput.notes || "").trim(),
      };
      this.targets.push(item);
      this.activeId = item.id;
      this.save();
      return item;
    }

    remove(id) {
      this.targets = this.targets.filter((target) => target.id !== id);
      if (this.activeId === id) {
        this.activeId = this.targets[0] ? this.targets[0].id : null;
      }
      this.save();
    }

    setActive(id) {
      this.activeId = id;
      this.save();
    }

    getActive() {
      return this.targets.find((target) => target.id === this.activeId) || null;
    }

    _newId() {
      if (window.crypto && window.crypto.randomUUID) {
        return window.crypto.randomUUID();
      }
      return `target_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
    }
  }

  class DashboardApp {
    constructor() {
      this.store = new TargetStore();
      this.state = {
        snapshots: new Map(),
        timer: null,
        ws: null,
        wsTargetId: null,
        wsReconnectTimer: null,
        refreshDebounceTimer: null,
      };
      this.dom = this._getDom();
    }

    start() {
      this._bindEvents();
      this._renderTargets();
      this.refreshSelected();
      this._connectLiveEvents();
      this.state.timer = setInterval(() => this.refreshSelected(), POLL_INTERVAL_MS);
    }

    _bindEvents() {
      this.dom.targetForm.addEventListener("submit", (event) => {
        event.preventDefault();
        const name = this.dom.targetName.value.trim();
        const host = this.dom.targetHost.value.trim();
        const port = this.dom.targetPort.value.trim();
        if (!name || !host || !port) {
          return;
        }
        this.store.add({
          name,
          protocol: this.dom.targetProtocol.value,
          host,
          port,
          notes: this.dom.targetNotes.value,
        });
        this.dom.targetForm.reset();
        this.dom.targetHost.value = "127.0.0.1";
        this.dom.targetPort.value = "8000";
        this._renderTargets();
        this._renderMain();
        this.refreshSelected();
        this._connectLiveEvents();
      });

      this.dom.refreshSelectedBtn.addEventListener("click", () => this.refreshSelected());
      this.dom.refreshAllBtn.addEventListener("click", () => this.refreshAll());
    }

    async refreshAll() {
      await Promise.all(this.store.targets.map((target) => this._loadSnapshot(target)));
      this._renderTargets();
      this._renderMain();
    }

    async refreshSelected() {
      const active = this.store.getActive();
      if (!active) {
        this._renderMain();
        return;
      }
      await this._loadSnapshot(active);
      this._renderTargets();
      this._renderMain();
    }

    _scheduleLiveRefresh() {
      if (this.state.refreshDebounceTimer) {
        return;
      }
      this.state.refreshDebounceTimer = setTimeout(async () => {
        this.state.refreshDebounceTimer = null;
        await this.refreshSelected();
      }, LIVE_REFRESH_DEBOUNCE_MS);
    }

    _setLiveStatus(text, ok) {
      this.dom.liveStatus.textContent = text;
      this.dom.liveStatus.classList.remove("ok", "err");
      this.dom.liveStatus.classList.add(ok ? "ok" : "err");
    }

    _closeLiveSocket() {
      if (this.state.wsReconnectTimer) {
        clearTimeout(this.state.wsReconnectTimer);
        this.state.wsReconnectTimer = null;
      }
      if (this.state.ws) {
        this.state.ws.onopen = null;
        this.state.ws.onclose = null;
        this.state.ws.onmessage = null;
        this.state.ws.onerror = null;
        this.state.ws.close();
        this.state.ws = null;
      }
      this.state.wsTargetId = null;
    }

    _connectLiveEvents() {
      const target = this.store.getActive();
      this._closeLiveSocket();

      if (!target) {
        this._setLiveStatus("Live disconnected", false);
        return;
      }

      const wsUrl = `${this._baseWsUrl(target)}/ws/dashboard-events`;
      this.state.wsTargetId = target.id;
      const ws = new WebSocket(wsUrl);
      this.state.ws = ws;
      this._setLiveStatus("Live connecting...", false);

      ws.onopen = () => {
        this._setLiveStatus("Live connected", true);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type !== "connected") {
            this._scheduleLiveRefresh();
          }
        } catch (_err) {
          this._scheduleLiveRefresh();
        }
      };

      ws.onerror = () => {
        this._setLiveStatus("Live error", false);
      };

      ws.onclose = () => {
        this._setLiveStatus("Live disconnected", false);
        const active = this.store.getActive();
        if (!active || active.id !== this.state.wsTargetId) {
          return;
        }
        this.state.wsReconnectTimer = setTimeout(
          () => this._connectLiveEvents(),
          WS_RECONNECT_MS
        );
      };
    }

    async _loadSnapshot(target) {
      const base = this._baseUrl(target);
      const [healthResult, alertsResult, edgesResult, logsResult] = await Promise.allSettled([
        this._fetchJson(`${base}/`),
        this._fetchJson(`${base}/api/alerts?limit=60`),
        this._fetchJson(`${base}/api/heartbeat`),
        this._fetchJson(`${base}/api/logs?limit=200`),
      ]);

      const snapshot = {
        fetchedAt: new Date().toISOString(),
        baseUrl: base,
        ok: false,
        health: null,
        alerts: [],
        edges: [],
        logs: [],
        error: null,
      };

      if (healthResult.status === "fulfilled") {
        snapshot.health = healthResult.value;
      } else {
        snapshot.error = healthResult.reason ? String(healthResult.reason) : "Health check failed";
      }
      if (alertsResult.status === "fulfilled") {
        snapshot.alerts = Array.isArray(alertsResult.value.alerts) ? alertsResult.value.alerts : [];
      }
      if (edgesResult.status === "fulfilled") {
        snapshot.edges = Array.isArray(edgesResult.value.edges) ? edgesResult.value.edges : [];
      }
      if (logsResult.status === "fulfilled") {
        snapshot.logs = Array.isArray(logsResult.value.logs) ? logsResult.value.logs : [];
      }

      snapshot.ok = snapshot.health !== null;
      this.state.snapshots.set(target.id, snapshot);
    }

    async _fetchJson(url) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      try {
        const response = await fetch(url, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }
        return await response.json();
      } finally {
        clearTimeout(timeout);
      }
    }

    _renderTargets() {
      this.dom.targetList.innerHTML = "";
      for (const target of this.store.targets) {
        const card = document.createElement("div");
        card.className = "target-item";
        if (target.id === this.store.activeId) {
          card.classList.add("active");
        }
        const snapshot = this.state.snapshots.get(target.id);
        const statusClass = snapshot && snapshot.ok ? "ok" : "err";
        const statusText = snapshot ? (snapshot.ok ? "Online" : "Offline") : "Unknown";

        card.innerHTML = [
          `<h3>${this._escape(target.name)}</h3>`,
          `<p>${this._escape(this._baseUrl(target))}</p>`,
          `<span class="tag ${statusClass}">${statusText}</span>`,
          target.notes ? `<p>${this._escape(target.notes)}</p>` : "",
          '<div class="target-actions">',
          '<button class="btn ghost" data-action="select" type="button">Open</button>',
          '<button class="btn ghost" data-action="delete" type="button">Remove</button>',
          "</div>",
        ].join("");

        card.querySelector("[data-action='select']").addEventListener("click", () => {
          this.store.setActive(target.id);
          this._renderTargets();
          this._renderMain();
          this.refreshSelected();
          this._connectLiveEvents();
        });
        card.querySelector("[data-action='delete']").addEventListener("click", () => {
          this.store.remove(target.id);
          this.state.snapshots.delete(target.id);
          this._renderTargets();
          this._renderMain();
          this._connectLiveEvents();
        });

        this.dom.targetList.appendChild(card);
      }
    }

    _renderMain() {
      const target = this.store.getActive();
      if (!target) {
        this.dom.activeTargetTitle.textContent = "No target selected";
        this.dom.activeTargetMeta.textContent = "Add a target to begin monitoring.";
        this.dom.lastUpdated.textContent = "Not updated yet";
        this.dom.healthValue.textContent = "Unknown";
        this.dom.alertsCount.textContent = "0";
        this.dom.edgeCount.textContent = "0";
        this.dom.logsCount.textContent = "0";
        this.dom.alertsList.innerHTML = '<div class="empty">No alerts to show.</div>';
        this.dom.edgeList.innerHTML = '<div class="empty">No edge status to show.</div>';
        this.dom.logsList.innerHTML = '<div class="empty">No logs to show.</div>';
        this._setLiveStatus("Live disconnected", false);
        return;
      }

      const snapshot = this.state.snapshots.get(target.id);
      this.dom.activeTargetTitle.textContent = target.name;
      this.dom.activeTargetMeta.textContent = `${this._baseUrl(target)}${target.notes ? ` - ${target.notes}` : ""}`;

      if (!snapshot) {
        this.dom.lastUpdated.textContent = "Waiting for first poll...";
        return;
      }

      this.dom.lastUpdated.textContent = `Last updated: ${this._fmtTs(snapshot.fetchedAt)}`;
      this.dom.healthValue.textContent = snapshot.ok ? "Online" : "Offline";
      this.dom.alertsCount.textContent = String(snapshot.alerts.length);
      this.dom.edgeCount.textContent = String(snapshot.edges.length);
      this.dom.logsCount.textContent = String(snapshot.logs.length);

      this._renderAlerts(snapshot, target);
      this._renderEdges(snapshot);
      this._renderLogs(snapshot);
    }

    _renderAlerts(snapshot, target) {
      if (snapshot.alerts.length === 0) {
        this.dom.alertsList.innerHTML = '<div class="empty">No alerts found.</div>';
        return;
      }

      const sorted = [...snapshot.alerts].sort((a, b) => {
        const ta = new Date(a.timestamp || 0).getTime();
        const tb = new Date(b.timestamp || 0).getTime();
        return tb - ta;
      });

      this.dom.alertsList.innerHTML = sorted.slice(0, 60).map((alert) => {
        const imageBlock = alert.image_path
          ? `<img src="${this._baseUrl(target)}/api/alerts/${encodeURIComponent(alert.id)}/image" alt="Alert image">`
          : "";
        const detections = Array.isArray(alert.detections) ? alert.detections : [];
        const detectionText = detections
          .map((detection) => {
            const className = detection.class || detection.class_ || "unknown";
            const confidence = Number(detection.confidence || 0).toFixed(2);
            return `${className} (${confidence})`;
          })
          .join(", ");

        return [
          '<div class="alert-item">',
          imageBlock,
          `<p class="kv"><strong>ID:</strong> ${this._escape(alert.id)}</p>`,
          `<p class="kv"><strong>Site:</strong> ${this._escape(alert.site_id || "unknown")}</p>`,
          `<p class="kv"><strong>Camera:</strong> ${this._escape(alert.camera_id || "unknown")}</p>`,
          `<p class="kv"><strong>Edge:</strong> ${this._escape(alert.edge_pc_id || "unknown")}</p>`,
          `<p class="kv"><strong>Time:</strong> ${this._fmtTs(alert.timestamp)}</p>`,
          `<p class="kv"><strong>Detections:</strong> ${this._escape(detectionText || "none")}</p>`,
          "</div>",
        ].join("");
      }).join("");
    }

    _renderEdges(snapshot) {
      if (snapshot.edges.length === 0) {
        this.dom.edgeList.innerHTML = '<div class="empty">No edge heartbeats recorded.</div>';
        return;
      }

      this.dom.edgeList.innerHTML = snapshot.edges.map((edge) => [
        '<div class="edge-item">',
        `<p class="kv"><strong>Edge:</strong> ${this._escape(edge.edge_pc_id || "unknown")}</p>`,
        `<p class="kv"><strong>Site:</strong> ${this._escape(edge.site_name || "unknown")}</p>`,
        `<p class="kv"><strong>Status:</strong> ${this._escape(edge.status || "unknown")}</p>`,
        `<p class="kv"><strong>Last Heartbeat:</strong> ${this._fmtTs(edge.last_heartbeat)}</p>`,
        "</div>",
      ].join("")).join("");
    }

    _renderLogs(snapshot) {
      if (snapshot.logs.length === 0) {
        this.dom.logsList.innerHTML = '<div class="empty">No logs available from this target.</div>';
        return;
      }

      this.dom.logsList.innerHTML = snapshot.logs.slice(-200).reverse().map((log) => {
        const level = String(log.level || "INFO").toUpperCase();
        const css = level.startsWith("ERR") ? "err" : (level.startsWith("WARN") ? "warn" : "ok");
        return [
          `<div class="log-item ${css}">`,
          `<p class="kv"><strong>${this._escape(level)}</strong> ${this._fmtTs(log.timestamp)}</p>`,
          `<p class="kv"><strong>${this._escape(log.logger || "server")}</strong></p>`,
          `<p class="kv">${this._escape(log.message || "")}</p>`,
          "</div>",
        ].join("");
      }).join("");
    }

    _baseUrl(target) {
      return `${target.protocol}://${target.host}:${target.port}`;
    }

    _baseWsUrl(target) {
      const scheme = target.protocol === "https" ? "wss" : "ws";
      return `${scheme}://${target.host}:${target.port}`;
    }

    _fmtTs(ts) {
      if (!ts) {
        return "unknown";
      }
      const date = new Date(ts);
      if (Number.isNaN(date.getTime())) {
        return String(ts);
      }
      return date.toLocaleString();
    }

    _escape(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    _getDom() {
      return {
        targetForm: document.getElementById("target-form"),
        targetName: document.getElementById("target-name"),
        targetProtocol: document.getElementById("target-protocol"),
        targetHost: document.getElementById("target-host"),
        targetPort: document.getElementById("target-port"),
        targetNotes: document.getElementById("target-notes"),
        targetList: document.getElementById("target-list"),
        refreshSelectedBtn: document.getElementById("refresh-selected-btn"),
        refreshAllBtn: document.getElementById("refresh-all-btn"),
        activeTargetTitle: document.getElementById("active-target-title"),
        activeTargetMeta: document.getElementById("active-target-meta"),
        liveStatus: document.getElementById("live-status"),
        lastUpdated: document.getElementById("last-updated"),
        healthValue: document.getElementById("health-value"),
        alertsCount: document.getElementById("alerts-count"),
        edgeCount: document.getElementById("edge-count"),
        logsCount: document.getElementById("logs-count"),
        alertsList: document.getElementById("alerts-list"),
        edgeList: document.getElementById("edge-list"),
        logsList: document.getElementById("logs-list"),
      };
    }
  }

  window.addEventListener("DOMContentLoaded", () => {
    const app = new DashboardApp();
    app.start();
  });
})();

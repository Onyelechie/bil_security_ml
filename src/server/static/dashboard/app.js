(function () {
  const STORAGE_KEY = "bil_server_dashboard_targets_v1";
  const VISITED_STORAGE_KEY = "bil_server_dashboard_alert_visits_v1";
  const ALERT_VIEW_STORAGE_KEY = "bil_server_dashboard_alert_view_v1";
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
        selectedAlerts: new Map(),
        visitedAlerts: this._loadVisitedAlerts(),
        alertListView: this._loadAlertListView(),
      };
      this.dom = this._getDom();
    }

    start() {
      this._bindEvents();
      this._renderTargets();
      this._renderAlertViewToggle();
      this._renderMain();
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
      this.dom.alertsViewCompact.addEventListener("click", () => this._setAlertListView("compact"));
      this.dom.alertsViewList.addEventListener("click", () => this._setAlertListView("list"));
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
      this.dom.liveStatus.className = `status-pill ${ok ? "ok" : "err"}`;
    }

    _setSelectedAlertVisitState(text, cssClass) {
      this.dom.selectedAlertVisitState.textContent = text;
      this.dom.selectedAlertVisitState.className = `status-pill ${cssClass}`;
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
      this._pruneVisitedAlerts(target.id, snapshot.alerts);
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

    _loadVisitedAlerts() {
      try {
        const raw = localStorage.getItem(VISITED_STORAGE_KEY);
        if (!raw) {
          return {};
        }
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : {};
      } catch (_err) {
        return {};
      }
    }

    _loadAlertListView() {
      try {
        const raw = localStorage.getItem(ALERT_VIEW_STORAGE_KEY);
        return raw === "list" ? "list" : "compact";
      } catch (_err) {
        return "compact";
      }
    }

    _saveVisitedAlerts() {
      localStorage.setItem(VISITED_STORAGE_KEY, JSON.stringify(this.state.visitedAlerts));
    }

    _setAlertListView(view) {
      this.state.alertListView = view === "list" ? "list" : "compact";
      localStorage.setItem(ALERT_VIEW_STORAGE_KEY, this.state.alertListView);
      this._renderAlertViewToggle();
      this._renderMain();
    }

    _renderAlertViewToggle() {
      const isListView = this.state.alertListView === "list";
      this.dom.alertsList.classList.toggle("list-view", isListView);
      this.dom.alertsViewCompact.classList.toggle("is-active", !isListView);
      this.dom.alertsViewList.classList.toggle("is-active", isListView);
    }

    _getVisitedAlerts(targetId) {
      const visited = this.state.visitedAlerts[targetId];
      return Array.isArray(visited) ? new Set(visited) : new Set();
    }

    _isAlertVisited(targetId, alertId) {
      return this._getVisitedAlerts(targetId).has(alertId);
    }

    _markAlertVisited(targetId, alertId) {
      if (!targetId || !alertId) {
        return;
      }
      const visited = this._getVisitedAlerts(targetId);
      if (visited.has(alertId)) {
        return;
      }
      visited.add(alertId);
      this.state.visitedAlerts[targetId] = Array.from(visited).slice(-500);
      this._saveVisitedAlerts();
    }

    _pruneVisitedAlerts(targetId, alerts) {
      if (!targetId || !Array.isArray(alerts)) {
        return;
      }
      const current = this._getVisitedAlerts(targetId);
      if (current.size === 0) {
        return;
      }
      const liveIds = new Set(alerts.map((alert) => alert.id));
      const pruned = Array.from(current).filter((alertId) => liveIds.has(alertId)).slice(-500);
      if (pruned.length === current.size) {
        return;
      }
      this.state.visitedAlerts[targetId] = pruned;
      this._saveVisitedAlerts();
    }

    _setSelectedAlert(targetId, alertId, markVisited) {
      if (!targetId || !alertId) {
        return;
      }
      const openedAsNew = markVisited && !this._isAlertVisited(targetId, alertId);
      this.state.selectedAlerts.set(targetId, {
        id: alertId,
        openedAsNew,
      });
      if (markVisited) {
        this._markAlertVisited(targetId, alertId);
      }
    }

    _ensureSelectedAlert(targetId, snapshot) {
      if (!snapshot || !Array.isArray(snapshot.alerts) || snapshot.alerts.length === 0) {
        this.state.selectedAlerts.delete(targetId);
        return null;
      }

      const sorted = this._getSortedAlerts(snapshot.alerts);
      const selectedState = this.state.selectedAlerts.get(targetId);
      if (selectedState && selectedState.id) {
        const selected = sorted.find((alert) => alert.id === selectedState.id);
        if (selected) {
          return selected;
        }
      }

      this.state.selectedAlerts.set(targetId, {
        id: sorted[0].id,
        openedAsNew: false,
      });
      return sorted[0];
    }

    _getSortedAlerts(alerts) {
      return [...alerts].sort((a, b) => {
        const ta = new Date(a.timestamp || 0).getTime();
        const tb = new Date(b.timestamp || 0).getTime();
        return tb - ta;
      });
    }

    _renderTargets() {
      this.dom.targetList.innerHTML = "";

      if (this.store.targets.length === 0) {
        this.dom.targetList.innerHTML = '<div class="empty">No targets added yet.</div>';
        return;
      }

      for (const target of this.store.targets) {
        const card = document.createElement("div");
        card.className = "target-item";
        if (target.id === this.store.activeId) {
          card.classList.add("active");
        }

        const snapshot = this.state.snapshots.get(target.id);
        const statusClass = snapshot && snapshot.ok ? "ok" : "err";
        const statusText = snapshot ? (snapshot.ok ? "Online" : "Offline") : "Unknown";
        const alertCount = snapshot ? `${snapshot.alerts.length} alerts` : "Waiting for poll";
        const updateText = snapshot ? `Updated ${this._fmtTs(snapshot.fetchedAt)}` : "No snapshot yet";

        card.innerHTML = [
          `<h3>${this._escape(target.name)}</h3>`,
          `<p>${this._escape(this._baseUrl(target))}</p>`,
          `<span class="status-pill ${statusClass}">${statusText}</span>`,
          `<p>${this._escape(alertCount)} - ${this._escape(updateText)}</p>`,
          target.notes ? `<p>${this._escape(target.notes)}</p>` : "",
          '<div class="target-actions">',
          '<button class="btn ghost" data-action="select" type="button">Focus</button>',
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
          this.state.selectedAlerts.delete(target.id);
          delete this.state.visitedAlerts[target.id];
          this._saveVisitedAlerts();
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
        this.dom.alertFeedCount.textContent = "0 total";
        this.dom.alertsList.innerHTML = '<div class="empty">No alerts to show.</div>';
        this.dom.edgeList.innerHTML = '<div class="empty">No edge status to show.</div>';
        this.dom.logsList.innerHTML = '<div class="empty">No logs to show.</div>';
        this._renderViewerPlaceholder(
          "No alert selected",
          "Select a target and click an alert from the right rail to inspect it.",
          "No image loaded",
          "Alert details will appear below once you open an alert.",
          "Standby",
          "neutral"
        );
        this._setLiveStatus("Live disconnected", false);
        return;
      }

      const snapshot = this.state.snapshots.get(target.id);
      this.dom.activeTargetTitle.textContent = target.name;
      this.dom.activeTargetMeta.textContent = `${this._baseUrl(target)}${target.notes ? ` - ${target.notes}` : ""}`;

      if (!snapshot) {
        this.dom.lastUpdated.textContent = "Waiting for first poll...";
        this.dom.healthValue.textContent = "Polling";
        this.dom.alertsCount.textContent = "0";
        this.dom.edgeCount.textContent = "0";
        this.dom.logsCount.textContent = "0";
        this.dom.alertFeedCount.textContent = "0 total";
        this.dom.alertsList.innerHTML = '<div class="empty">Waiting for alert data from this target.</div>';
        this.dom.edgeList.innerHTML = '<div class="empty">Waiting for edge heartbeat data.</div>';
        this.dom.logsList.innerHTML = '<div class="empty">Waiting for server logs.</div>';
        this._renderViewerPlaceholder(
          "Waiting for alerts",
          "The center console will fill with the latest alert image after the next refresh.",
          "Awaiting first image",
          "Alert details will appear here after the first snapshot arrives.",
          "Awaiting review",
          "pending"
        );
        return;
      }

      this.dom.lastUpdated.textContent = `Last updated: ${this._fmtTs(snapshot.fetchedAt)}`;
      this.dom.healthValue.textContent = snapshot.ok ? "Online" : "Offline";
      this.dom.alertsCount.textContent = String(snapshot.alerts.length);
      this.dom.edgeCount.textContent = String(snapshot.edges.length);
      this.dom.logsCount.textContent = String(snapshot.logs.length);
      this.dom.alertFeedCount.textContent = `${snapshot.alerts.length} total`;

      const selectedAlert = this._ensureSelectedAlert(target.id, snapshot);
      this._renderSelectedAlert(snapshot, target, selectedAlert);
      this._renderAlerts(snapshot, target, selectedAlert);
      this._renderEdges(snapshot);
      this._renderLogs(snapshot);
    }

    _renderSelectedAlert(snapshot, target, selectedAlert) {
      if (!selectedAlert) {
        const title = snapshot.ok ? "No alerts available" : "Alert feed unavailable";
        const summary = snapshot.ok
          ? "This target is online, but it has not reported any alerts yet."
          : (snapshot.error || "The dashboard could not fetch the alert feed from this target.");
        this._renderViewerPlaceholder(
          title,
          summary,
          "No image available",
          "As alerts arrive, their image and detail cards will render here.",
          "Awaiting review",
          "pending"
        );
        return;
      }

      const visited = this._isAlertVisited(target.id, selectedAlert.id);
      const selectedState = this.state.selectedAlerts.get(target.id);
      const openedAsNew =
        selectedState &&
        selectedState.id === selectedAlert.id &&
        selectedState.openedAsNew;
      const visitLabel = openedAsNew
        ? "Viewing now"
        : (visited ? "Visited" : "Awaiting review");
      const visitClass = openedAsNew
        ? "neutral"
        : (visited ? "visited" : "pending");
      const detailCards = [
        { label: "Alert ID", value: selectedAlert.id || "unknown" },
        { label: "Detected At", value: this._fmtTs(selectedAlert.timestamp) },
        { label: "Site", value: selectedAlert.site_id || "unknown" },
        { label: "Camera", value: selectedAlert.camera_id || "unknown" },
        { label: "Edge PC", value: selectedAlert.edge_pc_id || "unknown" },
        { label: "Review State", value: openedAsNew ? "First review" : visitLabel },
        { label: "Detections", value: this._formatDetections(selectedAlert) || "none" },
        { label: "Image", value: selectedAlert.image_path ? "Available" : "No image attached" },
      ];

      this.dom.selectedAlertTitle.textContent = this._alertLabel(selectedAlert);
      this.dom.selectedAlertSummary.textContent =
        `${this._fmtTs(selectedAlert.timestamp)} - ${this._formatDetections(selectedAlert) || "No detections reported"}`;
      this._setSelectedAlertVisitState(visitLabel, visitClass);

      if (selectedAlert.image_path) {
        const imageUrl = `${this._baseUrl(target)}/api/alerts/${encodeURIComponent(selectedAlert.id)}/image`;
        this.dom.selectedAlertStage.classList.remove("empty");
        this.dom.selectedAlertStage.innerHTML = [
          `<div class="viewer-badge"><span class="status-pill ${visitClass}">${visitLabel}</span></div>`,
          `<img class="viewer-image" src="${this._escape(imageUrl)}" alt="Alert image for ${this._escape(selectedAlert.id)}">`,
        ].join("");
      } else {
        this.dom.selectedAlertStage.classList.add("empty");
        this.dom.selectedAlertStage.innerHTML = [
          `<div class="viewer-badge"><span class="status-pill ${visitClass}">${visitLabel}</span></div>`,
          '<div class="viewer-placeholder">',
          '<span class="viewer-placeholder-label">No image available</span>',
          "<p>This alert was received without an attached image. The details remain available below.</p>",
          "</div>",
        ].join("");
      }

      this._renderDetailCards(detailCards);
    }

    _renderViewerPlaceholder(title, summary, stageLabel, detailsMessage, visitLabel, visitClass) {
      this.dom.selectedAlertTitle.textContent = title;
      this.dom.selectedAlertSummary.textContent = summary;
      this._setSelectedAlertVisitState(visitLabel, visitClass);
      this.dom.selectedAlertStage.classList.add("empty");
      this.dom.selectedAlertStage.innerHTML = [
        '<div class="viewer-placeholder">',
        `<span class="viewer-placeholder-label">${this._escape(stageLabel)}</span>`,
        `<p>${this._escape(detailsMessage)}</p>`,
        "</div>",
      ].join("");
      this._renderDetailCards([{ label: "Details", value: detailsMessage }]);
    }

    _renderDetailCards(cards) {
      this.dom.selectedAlertDetails.innerHTML = cards.map((card) => [
        '<article class="detail-card">',
        `<span class="detail-label">${this._escape(card.label)}</span>`,
        `<p class="detail-value">${this._escape(card.value)}</p>`,
        "</article>",
      ].join("")).join("");
    }

    _renderAlerts(snapshot, target, selectedAlert) {
      this.dom.alertsList.innerHTML = "";
      this._renderAlertViewToggle();

      if (snapshot.alerts.length === 0) {
        this.dom.alertsList.innerHTML = '<div class="empty">No alerts found for this target.</div>';
        return;
      }

      const sortedAlerts = this._getSortedAlerts(snapshot.alerts).slice(0, 60);
      const visitedAlerts = this._getVisitedAlerts(target.id);
      const isListView = this.state.alertListView === "list";

      for (const alert of sortedAlerts) {
        const isVisited = visitedAlerts.has(alert.id);
        const button = document.createElement("button");
        button.type = "button";
        button.className = "alert-row";
        if (selectedAlert && alert.id === selectedAlert.id) {
          button.classList.add("active");
        }
        if (isVisited) {
          button.classList.add("visited");
        }

        button.innerHTML = isListView
          ? [
              '<div class="alert-row-top">',
              '<div class="alert-row-list-main">',
              `<p class="alert-row-title">${this._escape(alert.site_id || "Unknown site")}</p>`,
              `<p class="alert-row-time">${this._escape(this._fmtTs(alert.timestamp))}</p>`,
              "</div>",
              `<span class="status-pill ${isVisited ? "visited" : "pending"}">${isVisited ? "Visited" : "New"}</span>`,
              "</div>",
            ].join("")
          : [
              '<div class="alert-row-top">',
              "<div>",
              `<p class="alert-row-title">${this._escape(this._alertLabel(alert))}</p>`,
              `<p class="alert-row-subtitle">${this._escape(alert.id || "unknown alert")}</p>`,
              "</div>",
              `<span class="status-pill ${isVisited ? "visited" : "pending"}">${isVisited ? "Visited" : "New"}</span>`,
              "</div>",
              `<p class="alert-row-meta">${this._escape(this._fmtTs(alert.timestamp))}</p>`,
              `<p class="alert-row-meta">${this._escape(this._formatDetections(alert) || "No detections reported")}</p>`,
              `<p class="alert-row-foot">${this._escape(alert.edge_pc_id || "unknown edge")} - ${alert.image_path ? "image ready" : "no image"}</p>`,
            ].join("");

        button.addEventListener("click", () => {
          this._setSelectedAlert(target.id, alert.id, true);
          this._renderMain();
        });

        this.dom.alertsList.appendChild(button);
      }
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
        `<p class="kv"><strong>Last Heartbeat:</strong> ${this._escape(this._fmtTs(edge.last_heartbeat))}</p>`,
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
          `<p class="kv"><strong>${this._escape(level)}</strong> ${this._escape(this._fmtTs(log.timestamp))}</p>`,
          `<p class="kv"><strong>${this._escape(log.logger || "server")}</strong></p>`,
          `<p class="kv">${this._escape(log.message || "")}</p>`,
          "</div>",
        ].join("");
      }).join("");
    }

    _alertLabel(alert) {
      const site = alert.site_id || "Unknown site";
      const camera = alert.camera_id || "unknown camera";
      return `${site} / ${camera}`;
    }

    _formatDetections(alert) {
      const detections = Array.isArray(alert.detections) ? alert.detections : [];
      return detections
        .map((detection) => {
          const className = detection.class || detection.class_ || "unknown";
          const confidenceValue = Number(detection.confidence);
          const confidence = Number.isFinite(confidenceValue) ? confidenceValue.toFixed(2) : "0.00";
          return `${className} (${confidence})`;
        })
        .join(", ");
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
        selectedAlertTitle: document.getElementById("selected-alert-title"),
        selectedAlertSummary: document.getElementById("selected-alert-summary"),
        selectedAlertVisitState: document.getElementById("selected-alert-visit-state"),
        selectedAlertStage: document.getElementById("selected-alert-stage"),
        selectedAlertDetails: document.getElementById("selected-alert-details"),
        alertFeedCount: document.getElementById("alert-feed-count"),
        alertsViewCompact: document.getElementById("alerts-view-compact"),
        alertsViewList: document.getElementById("alerts-view-list"),
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

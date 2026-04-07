(function () {
  const STORAGE_KEY = "bil_server_dashboard_targets_v1";
  const VISITED_STORAGE_KEY = "bil_server_dashboard_alert_visits_v1";
  const ALERT_VIEW_STORAGE_KEY = "bil_server_dashboard_alert_view_v1";
  const ALERT_SORT_STORAGE_KEY = "bil_server_dashboard_alert_sort_v1";
  const AUTH_STORAGE_KEY = "bil_server_dashboard_admin_tokens_v1";
  const SETTINGS_VIEW_STORAGE_KEY = "bil_server_dashboard_settings_view_v1";
  const ALERT_TIME_ZONE = "America/Winnipeg";
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
        alertSortBy: this._loadAlertSortBy(),
        authTokens: this._loadAuthTokens(),
        settingsView: this._loadSettingsView(),
        managedDevices: [],
        managedDevicesLoadedFor: null,
        pendingDeviceRevoke: null,
      };
      this.dom = this._getDom();
      this.dom.settingsSiteSelect = document.getElementById("settings-site-select");
      this.dom.settingsRetentionHours = document.getElementById("settings-retention-hours");
      this.dom.settingsSaveSite = document.getElementById("settings-save-site");
      this.dom.settingsAuthState = document.getElementById("settings-auth-state");
      this.dom.settingsAdminPassword = document.getElementById("settings-admin-password");
      this.dom.settingsAuthLogin = document.getElementById("settings-auth-login");
      this.dom.settingsAuthClear = document.getElementById("settings-auth-clear");
      this.dom.settingsAuthMessage = document.getElementById("settings-auth-message");
      this.dom.settingsDeviceId = document.getElementById("settings-device-id");
      this.dom.settingsPublicKey = document.getElementById("settings-public-key");
      this.dom.settingsEnrollDevice = document.getElementById("settings-enroll-device");
      this.dom.settingsEnrollMessage = document.getElementById("settings-enroll-message");
      this.dom.settingsDeviceCount = document.getElementById("settings-device-count");
      this.dom.settingsDeviceRefresh = document.getElementById("settings-device-refresh");
      this.dom.settingsDeviceMessage = document.getElementById("settings-device-message");
      this.dom.settingsDeviceList = document.getElementById("settings-device-list");
      this.dom.alertsSortReceived = document.getElementById("alerts-sort-received");
      this.dom.alertsSortTimestamp = document.getElementById("alerts-sort-timestamp");
      this.dom.settingsViewButtons = Array.from(
        document.querySelectorAll(".settings-subnav-btn[data-settings-view]")
      );
      this.dom.settingsPanels = Array.from(document.querySelectorAll("[data-settings-panel]"));
      this.dom.settingsActiveTarget = document.getElementById("settings-active-target");
      this.dom.settingsActiveBase = document.getElementById("settings-active-base");
      this.dom.settingsActiveAdmin = document.getElementById("settings-active-admin");
      this.dom.settingsActiveSection = document.getElementById("settings-active-section");
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
      if (this.dom.alertsSortReceived) {
        this.dom.alertsSortReceived.addEventListener("click", () =>
          this._setAlertSortBy("received_at")
        );
      }
      if (this.dom.alertsSortTimestamp) {
        this.dom.alertsSortTimestamp.addEventListener("click", () =>
          this._setAlertSortBy("timestamp")
        );
      }
      if (this.dom.settingsSaveSite) {
        this.dom.settingsSaveSite.addEventListener("click", async () => {
          const sel = this.dom.settingsSiteSelect;
          const hoursEl = this.dom.settingsRetentionHours;
          if (!sel || !hoursEl) return;
          const site = sel.value;
          const hours = Number(hoursEl.value) || null;
          if (!site) return;
          try {
            const active = this.store.getActive();
            const base = active ? this._baseUrl(active) : window.location.origin;
            const resp = await fetch(`${base}/api/sites/${encodeURIComponent(site)}/settings`, {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ image_retention_hours: hours }),
            });
            if (!resp.ok) throw new Error("Failed to save site settings");
            alert("Site settings saved");
          } catch (err) {
            console.error(err);
            alert("Failed to save site settings");
          }
        });
      }
      if (this.dom.settingsSiteSelect) {
        this.dom.settingsSiteSelect.addEventListener("change", async () => {
          const sel = this.dom.settingsSiteSelect;
          const hoursEl = this.dom.settingsRetentionHours;
          if (!sel || !hoursEl) return;
          const site = sel.value;
          if (!site) return;
          try {
            const active = this.store.getActive();
            const base = active ? this._baseUrl(active) : window.location.origin;
            const resp = await fetch(`${base}/api/sites/${encodeURIComponent(site)}/settings`);
            if (!resp.ok) throw new Error("Failed to fetch site settings");
            const data = await resp.json();
            hoursEl.value = data.image_retention_hours || hoursEl.value || 24;
          } catch (err) {
            console.error(err);
          }
        });
      }
      if (this.dom.settingsAuthLogin) {
        this.dom.settingsAuthLogin.addEventListener("click", async () => {
          const active = this.store.getActive();
          const password = (this.dom.settingsAdminPassword?.value || "").trim();
          if (!active || !password) {
            this._setSettingsMessage("Enter an admin password for the active target.", true);
            return;
          }
          const base = this._baseUrl(active);
          try {
            const body = new URLSearchParams({ username: "admin", password });
            const response = await fetch(`${base}/api/auth/token`, {
              method: "POST",
              headers: { "Content-Type": "application/x-www-form-urlencoded" },
              body,
            });
            if (!response.ok) throw new Error("Invalid admin credentials");
            const data = await response.json();
            this._setAuthToken(active.id, data.access_token);
            if (this.dom.settingsAdminPassword) {
              this.dom.settingsAdminPassword.value = "";
            }
            this._renderSettingsAdminState();
            this._setSettingsMessage("Admin access unlocked for this target.", false);
            await this.refreshSelected();
            await this._refreshManagedDevices(true);
          } catch (err) {
            console.error(err);
            this._setSettingsMessage("Failed to unlock admin access.", true);
          }
        });
      }
      if (this.dom.settingsAuthClear) {
        this.dom.settingsAuthClear.addEventListener("click", async () => {
          const active = this.store.getActive();
          if (!active) return;
          this._handleAuthExpired(active, "Stored admin token cleared for this target.", false);
          this._setSettingsMessage("Stored admin token cleared for this target.", false);
          await this.refreshSelected();
        });
      }
      if (this.dom.settingsEnrollDevice) {
        this.dom.settingsEnrollDevice.addEventListener("click", async () => {
          const active = this.store.getActive();
          if (!active) {
            this._setEnrollMessage("Select an active target first.", true);
            return;
          }
          const token = this._getAuthToken(active.id);
          if (!token) {
            this._setEnrollMessage("Unlock admin access before enrolling a device.", true);
            return;
          }
          const deviceId = (this.dom.settingsDeviceId?.value || "").trim();
          const publicKey = (this.dom.settingsPublicKey?.value || "").trim();
          if (!deviceId || !publicKey) {
            this._setEnrollMessage("Enter both the edge/device id and the public key.", true);
            return;
          }
          try {
            const response = await fetch(`${this._baseUrl(active)}/api/devices/enroll`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                ...this._authHeaders(active),
              },
              body: JSON.stringify({
                device_id: deviceId,
                public_key_b64: publicKey,
              }),
            });
            const payload = await response.json().catch(() => ({}));
            if (response.status === 401) {
              this._handleAuthExpired(active, "Admin token expired. Unlock admin again.", true);
              this._setEnrollMessage("Admin token expired. Unlock admin again.", true);
              return;
            }
            if (!response.ok) {
              throw new Error(payload.detail || "Failed to enroll device");
            }
            this._setEnrollMessage(`Device ${deviceId} enrolled successfully.`, false);
            await this._refreshManagedDevices(true);
          } catch (err) {
            console.error(err);
            this._setEnrollMessage(err.message || "Failed to enroll device.", true);
          }
        });
      }
      if (this.dom.settingsDeviceRefresh) {
        this.dom.settingsDeviceRefresh.addEventListener("click", async () => {
          await this._refreshManagedDevices(true);
        });
      }
      if (this.dom.settingsDeviceList) {
        this.dom.settingsDeviceList.addEventListener("click", async (event) => {
          const button = event.target.closest("[data-device-revoke]");
          const cancelButton = event.target.closest("[data-device-revoke-cancel]");
          if (!button && !cancelButton) return;
          if (cancelButton) {
            this.state.pendingDeviceRevoke = null;
            this._setDeviceMessage("Revocation cancelled.", false);
            this._renderManagedDevices();
            return;
          }
          const deviceId = button.dataset.deviceRevoke || "";
          if (!deviceId) return;
          const active = this.store.getActive();
          if (!active) {
            this._setDeviceMessage("Select an active target first.", true);
            return;
          }
          const token = this._getAuthToken(active.id);
          if (!token) {
            this._setDeviceMessage("Unlock admin access before revoking a device.", true);
            return;
          }
          if (this.state.pendingDeviceRevoke !== deviceId) {
            this.state.pendingDeviceRevoke = deviceId;
            this._setDeviceMessage(
              `Click \"Confirm revoke\" to revoke ${deviceId}, or cancel to keep it active.`,
              true
            );
            this._renderManagedDevices();
            return;
          }
          try {
            const response = await fetch(
              `${this._baseUrl(active)}/api/devices/${encodeURIComponent(deviceId)}/revoke`,
              {
                method: "POST",
                headers: this._authHeaders(active),
              }
            );
            const payload = await response.json().catch(() => ({}));
            if (response.status === 401) {
              this._handleAuthExpired(active, "Admin token expired. Unlock admin again.", true);
              this._setDeviceMessage("Admin token expired. Unlock admin again.", true);
              return;
            }
            if (!response.ok) {
              throw new Error(payload.detail || "Failed to revoke device");
            }
            this.state.pendingDeviceRevoke = null;
            this._setDeviceMessage(`Device ${deviceId} revoked successfully.`, false);
            await this._refreshManagedDevices(true);
          } catch (err) {
            console.error(err);
            this._setDeviceMessage(err.message || "Failed to revoke device.", true);
          }
        });
      }
      this.dom.settingsViewButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
          this._setSettingsView(btn.dataset.settingsView);
        });
      });
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
      if (document.body.getAttribute("data-view") === "settings") {
        await this._refreshManagedDevices();
      }
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
      const [healthResult, alertsResult, edgesResult, logsResult, serverInfoResult] = await Promise.allSettled([
        this._fetchJson(`${base}/`),
        this._fetchJson(
          `${base}/api/alerts?limit=60&sort_by=${encodeURIComponent(this.state.alertSortBy || "received_at")}`
        ),
        this._fetchJson(`${base}/api/heartbeat`),
        this._fetchJson(`${base}/api/logs?limit=200`, {
          headers: this._authHeaders(target),
          authTarget: target,
        }),
        this._fetchJson(`${base}/api/server-info`),
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
      if (serverInfoResult.status === "fulfilled") {
        snapshot.serverInfo = serverInfoResult.value;
      } else {
        snapshot.serverInfo = null;
      }
      this.state.snapshots.set(target.id, snapshot);
      this._pruneVisitedAlerts(target.id, snapshot.alerts);
    }

    async _fetchJson(url, options = {}) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      try {
        const response = await fetch(url, {
          ...options,
          headers: {
            ...(options.headers || {}),
          },
          signal: controller.signal,
        });
        if (response.status === 401 && options.authTarget) {
          this._handleAuthExpired(
            options.authTarget,
            "Admin token expired. Unlock admin again.",
            true
          );
          throw new Error("Admin token expired");
        }
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

    _loadAlertSortBy() {
      try {
        const raw = localStorage.getItem(ALERT_SORT_STORAGE_KEY);
        return raw === "timestamp" ? "timestamp" : "received_at";
      } catch (_err) {
        return "received_at";
      }
    }

    _loadAuthTokens() {
      try {
        const raw = localStorage.getItem(AUTH_STORAGE_KEY);
        if (!raw) {
          return {};
        }
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : {};
      } catch (_err) {
        return {};
      }
    }

    _loadSettingsView() {
      try {
        const raw = localStorage.getItem(SETTINGS_VIEW_STORAGE_KEY);
        return ["cleanup", "security", "guide"].includes(raw) ? raw : "security";
      } catch (_err) {
        return "security";
      }
    }

    _saveAuthTokens() {
      localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(this.state.authTokens));
    }

    _getAuthToken(targetId) {
      return targetId ? this.state.authTokens[targetId] || "" : "";
    }

    _setAuthToken(targetId, token) {
      if (!targetId) return;
      this.state.authTokens[targetId] = token;
      this._saveAuthTokens();
    }

    _clearAuthToken(targetId) {
      if (!targetId) return;
      delete this.state.authTokens[targetId];
      this._saveAuthTokens();
    }

    _handleAuthExpired(target, message, isError = true) {
      if (!target) return;
      this._clearAuthToken(target.id);
      this.state.managedDevices = [];
      this.state.managedDevicesLoadedFor = null;
      this.state.pendingDeviceRevoke = null;
      this._renderSettingsAdminState();
      this._renderManagedDevices();
      if (message) {
        this._setSettingsMessage(message, isError);
      }
    }

    _authHeaders(target) {
      const token = target ? this._getAuthToken(target.id) : "";
      return token ? { Authorization: `Bearer ${token}` } : {};
    }

    _setSettingsView(view) {
      const nextView = ["cleanup", "security", "guide"].includes(view) ? view : "security";
      this.state.settingsView = nextView;
      localStorage.setItem(SETTINGS_VIEW_STORAGE_KEY, nextView);
      this._renderSettingsView();
      this._renderSettingsSummary();
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
      this._renderAlertSortToggle();
    }

    _setAlertSortBy(sortBy) {
      const nextSort = sortBy === "timestamp" ? "timestamp" : "received_at";
      this.state.alertSortBy = nextSort;
      localStorage.setItem(ALERT_SORT_STORAGE_KEY, nextSort);
      this._renderAlertSortToggle();
      this.refreshSelected();
    }

    _renderAlertSortToggle() {
      const sortBy = this.state.alertSortBy || "received_at";
      if (this.dom.alertsSortReceived) {
        this.dom.alertsSortReceived.classList.toggle("is-active", sortBy === "received_at");
      }
      if (this.dom.alertsSortTimestamp) {
        this.dom.alertsSortTimestamp.classList.toggle("is-active", sortBy === "timestamp");
      }
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
      const sortBy = this.state.alertSortBy || "received_at";
      return [...alerts].sort((a, b) => {
        const ta = this._alertSortValue(a, sortBy);
        const tb = this._alertSortValue(b, sortBy);
        return tb - ta;
      });
    }

    _alertSortValue(alert, sortBy) {
      const rawValue =
        sortBy === "timestamp"
          ? (alert.timestamp || alert.received_at)
          : (alert.received_at || alert.timestamp);
      const value = new Date(rawValue || 0).getTime();
      return Number.isFinite(value) ? value : 0;
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
          this._renderSettingsAdminState();
          this._renderSettingsSummary();
          void this._refreshManagedDevices(true);
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
          this._renderSettingsAdminState();
          this._renderSettingsSummary();
          void this._refreshManagedDevices(true);
          this._connectLiveEvents();
        });

        this.dom.targetList.appendChild(card);
      }
    }

    _renderMain() {
      const target = this.store.getActive();
      this._renderSettingsAdminState();
      this._renderSettingsSummary();
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

      // render server info if available
      try {
        if (snapshot.serverInfo) {
          const si = snapshot.serverInfo;
          this.dom.serverInfo.textContent = `Host: ${si.host}:${si.port} — edges: ${si.edge_count}`;
        } else {
          this.dom.serverInfo.textContent = "";
        }
      } catch (_err) {
        this.dom.serverInfo.textContent = "";
      }

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
        { label: "Detected At", value: this._fmtAlertTs(selectedAlert.timestamp) },
        {
          label: "Received At",
          value: this._fmtAlertTs(selectedAlert.received_at || selectedAlert.timestamp),
        },
        { label: "Site", value: selectedAlert.site_id || "unknown" },
        { label: "Camera", value: selectedAlert.camera_id || "unknown" },
        { label: "Edge PC", value: selectedAlert.edge_pc_id || "unknown" },
        { label: "Review State", value: openedAsNew ? "First review" : visitLabel },
        { label: "Detections", value: this._formatDetections(selectedAlert) || "none" },
        { label: "Image", value: selectedAlert.image_path ? "Available" : "No image attached" },
      ];

      this.dom.selectedAlertTitle.textContent = this._alertLabel(selectedAlert);
      this.dom.selectedAlertSummary.textContent =
        `${this._fmtAlertTs(selectedAlert.timestamp)} - ${this._formatDetections(selectedAlert) || "No detections reported"}`;
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
              `<p class="alert-row-time">Alert time: ${this._escape(this._fmtAlertTs(alert.timestamp))}</p>`,
              `<p class="alert-row-time">Received: ${this._escape(this._fmtAlertTs(alert.received_at || alert.timestamp))}</p>`,
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
              `<p class="alert-row-meta">Alert time: ${this._escape(this._fmtAlertTs(alert.timestamp))}</p>`,
              `<p class="alert-row-meta">Received: ${this._escape(this._fmtAlertTs(alert.received_at || alert.timestamp))}</p>`,
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

    _populateSettingsSites() {
      try {
        const sel = this.dom.settingsSiteSelect;
        if (!sel) return;
        const sites = new Set();
        for (const snapshot of this.state.snapshots.values()) {
          if (Array.isArray(snapshot.edges)) {
            snapshot.edges.forEach((e) => e && e.site_name && sites.add(e.site_name));
          }
          if (Array.isArray(snapshot.alerts)) {
            snapshot.alerts.forEach((a) => a && a.site_id && sites.add(a.site_id));
          }
        }
        const arr = Array.from(sites).filter(Boolean).sort();
        sel.innerHTML = "";
        arr.forEach((s) => {
          const opt = document.createElement("option");
          opt.value = s;
          opt.textContent = s;
          sel.appendChild(opt);
        });
        // trigger change to load settings for first site
        if (arr.length > 0) {
          sel.value = arr[0];
          sel.dispatchEvent(new Event("change"));
        }
      } catch (_err) {
        // ignore
      }
    }

    _renderSettingsAdminState() {
      const active = this.store.getActive();
      const hasToken = !!(active && this._getAuthToken(active.id));
      if (this.dom.settingsAuthState) {
        this.dom.settingsAuthState.textContent = hasToken ? "Admin unlocked" : "Admin locked";
        this.dom.settingsAuthState.className = `status-pill ${hasToken ? "ok" : "err"}`;
      }
      if (this.dom.settingsAuthMessage) {
        if (!active) {
          this.dom.settingsAuthMessage.textContent = "Choose an active target before authenticating.";
        } else if (hasToken) {
          this.dom.settingsAuthMessage.textContent = `Protected API access is enabled for ${active.name}.`;
        } else {
          this.dom.settingsAuthMessage.textContent = `No admin token stored for ${active.name}.`;
        }
      }
      this._renderSettingsSummary();
    }

    _renderSettingsView() {
      const activeView = this.state.settingsView || "security";
      const sectionLabels = {
        cleanup: "Cleanup",
        security: "Security",
        guide: "Guide",
      };

      this.dom.settingsViewButtons.forEach((btn) => {
        const isActive = btn.dataset.settingsView === activeView;
        btn.classList.toggle("is-active", isActive);
        btn.setAttribute("aria-selected", isActive ? "true" : "false");
      });

      this.dom.settingsPanels.forEach((panel) => {
        const show = panel.dataset.settingsPanel === activeView;
        panel.classList.toggle("hidden", !show);
      });

      if (this.dom.settingsActiveSection) {
        this.dom.settingsActiveSection.textContent = sectionLabels[activeView] || "Security";
      }
    }

    _renderSettingsSummary() {
      const active = this.store.getActive();
      const hasToken = !!(active && this._getAuthToken(active.id));
      const labels = {
        cleanup: "Cleanup",
        security: "Security",
        guide: "Guide",
      };

      if (this.dom.settingsActiveTarget) {
        this.dom.settingsActiveTarget.textContent = active ? active.name : "No target selected";
      }
      if (this.dom.settingsActiveBase) {
        this.dom.settingsActiveBase.textContent = active
          ? `${this._baseUrl(active)}${active.notes ? ` - ${active.notes}` : ""}`
          : "Choose a dashboard target to begin administrative work.";
      }
      if (this.dom.settingsActiveAdmin) {
        this.dom.settingsActiveAdmin.textContent = hasToken ? "Unlocked" : "Locked";
      }
      if (this.dom.settingsActiveSection) {
        this.dom.settingsActiveSection.textContent =
          labels[this.state.settingsView] || "Security";
      }
      this._renderManagedDevices();
    }

    async _refreshManagedDevices(force = false) {
      const active = this.store.getActive();
      const hasToken = !!(active && this._getAuthToken(active.id));
      if (!active || !hasToken) {
        this.state.managedDevices = [];
        this.state.managedDevicesLoadedFor = null;
        this.state.pendingDeviceRevoke = null;
        this._renderManagedDevices();
        return;
      }
      if (!force && this.state.managedDevicesLoadedFor === active.id) {
        this._renderManagedDevices();
        return;
      }
      try {
        const response = await fetch(`${this._baseUrl(active)}/api/devices`, {
          headers: this._authHeaders(active),
        });
        const payload = await response.json().catch(() => ({}));
        if (response.status === 401) {
          this._handleAuthExpired(active, "Admin token expired. Unlock admin again.", true);
          this._setDeviceMessage("Admin token expired. Unlock admin again.", true);
          return;
        }
        if (!response.ok) {
          throw new Error(payload.detail || "Failed to load devices");
        }
        this.state.managedDevices = Array.isArray(payload.devices) ? payload.devices : [];
        this.state.managedDevicesLoadedFor = active.id;
        this.state.pendingDeviceRevoke = null;
        this._renderManagedDevices();
      } catch (err) {
        console.error(err);
        this.state.managedDevices = [];
        this.state.managedDevicesLoadedFor = null;
        this.state.pendingDeviceRevoke = null;
        this._setDeviceMessage(err.message || "Failed to load devices.", true);
        this._renderManagedDevices();
      }
    }

    _renderManagedDevices() {
      const active = this.store.getActive();
      const hasToken = !!(active && this._getAuthToken(active.id));
      const devices = Array.isArray(this.state.managedDevices)
        ? [...this.state.managedDevices]
        : [];
      devices.sort((a, b) => {
        if (a.active !== b.active) {
          return a.active ? -1 : 1;
        }
        return String(a.device_id || "").localeCompare(String(b.device_id || ""));
      });

      if (this.dom.settingsDeviceCount) {
        this.dom.settingsDeviceCount.textContent = `${devices.length} device${devices.length === 1 ? "" : "s"}`;
      }

      if (!this.dom.settingsDeviceList) {
        return;
      }

      if (!active) {
        this.dom.settingsDeviceList.innerHTML = '<div class="empty">No target selected.</div>';
        this._setDeviceMessage("Select an active target to manage devices.", true);
        return;
      }
      if (!hasToken) {
        this.dom.settingsDeviceList.innerHTML = '<div class="empty">Admin access is locked.</div>';
        this._setDeviceMessage("Unlock admin access to load and manage devices.", true);
        return;
      }
      if (devices.length === 0) {
        this.dom.settingsDeviceList.innerHTML = '<div class="empty">No enrolled devices yet.</div>';
        this._setDeviceMessage(`No enrolled devices found for ${active.name}.`, false);
        return;
      }

      this.dom.settingsDeviceList.innerHTML = devices.map((device) => {
        const isActive = !!device.active;
        const statusLabel = isActive ? "Active" : "Revoked";
        const statusClass = isActive ? "ok" : "err";
        const isPendingRevoke = this.state.pendingDeviceRevoke === device.device_id;
        const revokeButton = isActive
          ? `<button class="btn ghost${isPendingRevoke ? " danger" : ""}" type="button" data-device-revoke="${this._escape(device.device_id)}">${isPendingRevoke ? "Confirm revoke" : "Revoke"}</button>`
          : '<button class="btn ghost" type="button" disabled>Revoked</button>';
        const cancelButton = isActive && isPendingRevoke
          ? '<button class="btn ghost" type="button" data-device-revoke-cancel="true">Cancel</button>'
          : "";
        return [
          '<article class="device-card">',
          '<div class="section-heading device-card-heading">',
          '<div>',
          `<p class="eyebrow">${isActive ? "Authorized Edge" : "Revoked Edge"}</p>`,
          `<h3>${this._escape(device.device_id || "unknown device")}</h3>`,
          "</div>",
          `<span class="status-pill ${statusClass}">${statusLabel}</span>`,
          "</div>",
          '<div class="device-card-meta">',
          `<p><strong>Enrolled:</strong> ${this._escape(this._fmtTs(device.enrolled_at))}</p>`,
          `<p><strong>Revoked:</strong> ${this._escape(this._fmtTs(device.revoked_at))}</p>`,
          `<p><strong>Last key rotation:</strong> ${this._escape(this._fmtTs(device.last_key_rotation_at))}</p>`,
          "</div>",
          '<div class="action-row">',
          revokeButton,
          cancelButton,
          "</div>",
          "</article>",
        ].join("");
      }).join("");

      this._setDeviceMessage(`Showing enrolled devices for ${active.name}.`, false);
    }

    _setSettingsMessage(message, isError) {
      if (!this.dom.settingsAuthMessage) return;
      this.dom.settingsAuthMessage.textContent = message;
      this.dom.settingsAuthMessage.classList.toggle("danger-text", !!isError);
    }

    _setEnrollMessage(message, isError) {
      if (!this.dom.settingsEnrollMessage) return;
      this.dom.settingsEnrollMessage.textContent = message;
      this.dom.settingsEnrollMessage.classList.toggle("danger-text", !!isError);
      this.dom.settingsEnrollMessage.classList.toggle("ok-text", !isError);
    }

    _setDeviceMessage(message, isError) {
      if (!this.dom.settingsDeviceMessage) return;
      this.dom.settingsDeviceMessage.textContent = message;
      this.dom.settingsDeviceMessage.classList.toggle("danger-text", !!isError);
      this.dom.settingsDeviceMessage.classList.toggle("ok-text", !isError);
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

    _fmtAlertTs(ts) {
      if (!ts) {
        return "unknown";
      }
      const date = new Date(ts);
      if (Number.isNaN(date.getTime())) {
        return String(ts);
      }
      return new Intl.DateTimeFormat(undefined, {
        timeZone: ALERT_TIME_ZONE,
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
        timeZoneName: "short",
      }).format(date);
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
        serverInfo: document.getElementById("server-info"),
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
    // --- Simple client-side view routing (Overview / Alerts / Settings) ---
    const NAV_KEY = "dashboard_view";
    const navButtons = Array.from(document.querySelectorAll(".nav-btn[data-view]"));

    function setView(view) {
      const allowed = ["overview", "alerts", "settings"];
      if (!allowed.includes(view)) view = "overview";
      document.body.setAttribute("data-view", view);
      // toggle nav active state
      navButtons.forEach((btn) => {
        btn.classList.toggle("is-active", btn.dataset.view === view);
      });
      // show/hide sections based on data-pages attribute
      document.querySelectorAll("[data-pages]").forEach((el) => {
        const pages = (el.dataset.pages || "").trim().split(/\s+/).filter(Boolean);
        const show = pages.length === 0 ? true : pages.includes(view);
        el.classList.toggle("hidden", !show);
      });
      localStorage.setItem(NAV_KEY, view);
      // populate settings UI when opening settings
      if (view === "settings") {
        try {
          app._populateSettingsSites();
          app._renderSettingsAdminState();
          app._renderSettingsView();
          app._renderSettingsSummary();
          void app._refreshManagedDevices();
        } catch (_e) {
          // ignore
        }
      }
    }

    navButtons.forEach((btn) => {
      btn.addEventListener("click", () => setView(btn.dataset.view));
    });

    const saved = localStorage.getItem(NAV_KEY);
    const defaultBtn = document.querySelector(".nav-btn.is-active[data-view]");
    const initial = saved || (defaultBtn && defaultBtn.dataset.view) || "overview";
    app._renderSettingsView();
    app._renderSettingsSummary();
    void app._refreshManagedDevices();
    setView(initial);
  });
})();

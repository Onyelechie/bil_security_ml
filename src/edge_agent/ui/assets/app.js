(async function () {
  const runtimePill = document.getElementById("runtime-pill");
  const pipelineMode = document.getElementById("pipeline-mode");
  const streamState = document.getElementById("stream-state");
  const senderStatus = document.getElementById("sender-status");
  const ringCount = document.getElementById("ring-count");
  const previewStage = document.getElementById("preview-stage");
  const settingsGroups = document.getElementById("settings-groups");

  let latestSettings = null;

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function fieldMeta() {
    return {
      site_id: "Site identifier used when the edge sends heartbeats and alerts.",
      site_name: "Human-friendly location name shown to operators.",
      edge_pc_id: "Unique identity for this edge PC.",
      device_id: "Authenticated device id. Usually match this with the edge PC id.",
      server_base_url: "Office server address where heartbeats and alerts are sent.",
      default_camera_id: "Camera id used for local motion events and live tests.",

      rtsp_url_low: "Low-resolution RTSP stream used for analysis.",
      ring_buffer_seconds: "How many seconds of recent video stay available in memory.",
      analysis_fps: "How many frames per second are sampled from the stream for analysis.",
      frame_width: "Target analysis width after resizing.",
      frame_height: "Target analysis height after resizing.",

      enable_tcp_motion: "Use motion events coming from an external source over TCP.",
      enable_local_motion: "Use motion detected directly from the edge video stream.",
      motion_fps: "How often motion scoring runs per second.",
      motion_pixel_delta: "How much a pixel must change to count as motion.",
      motion_threshold: "How much overall changed area is needed before motion is accepted.",
      trigger_cooldown_sec: "Minimum wait time before allowing another accepted trigger.",
      trigger_merge_window_sec: "Merge very close motion events into one activity period.",

      incident_quiet_sec: "How long the scene must calm down before an incident closes.",
      incident_max_sec: "Maximum allowed incident length before it is finalized.",
      incident_tick_interval_sec: "How often incident state is updated.",
      window_pre_sec: "How much video before motion is kept in the evidence window.",
      window_post_sec: "How much video after motion is kept in the evidence window.",
      window_target_fps: "How densely frames are selected inside the evidence window.",
      window_max_frames: "Maximum number of selected frames per evidence window.",
      window_wait_grace_sec: "Extra wait time to collect trailing frames after an incident.",

      detector_model: "Detection model used for person and vehicle evaluation.",
      detector_weights: "Optional custom weights path. Leave blank to use model defaults.",
      detector_person_conf: "Minimum confidence required before a person can become an alert.",
      detector_vehicle_conf: "Minimum confidence required before a vehicle can become an alert.",
      detector_allowed_classes: "Which alert classes are allowed, separated by commas.",

      ptz_global_motion_threshold: "How much full-frame change is needed before camera movement is suspected.",
      ptz_consecutive_frames: "How many consecutive PTZ-like checks are required before suppression begins.",
      ptz_suppress_sec: "How long local motion is paused after likely camera movement is detected.",

      motion_include_polygons: "Only motion inside these regions is considered active.",
      motion_exclude_polygons: "Motion inside these ignored regions is excluded.",

      heartbeat_interval_sec: "How often the edge reports heartbeat status.",
      update_interval_sec: "Reserved background update interval.",
      retry_interval_sec: "How often the edge retries queued alerts."
    };
  }

  function sectionOrder() {
    return [
      ["Identity and connection", "identity"],
      ["Camera and stream behavior", "stream"],
      ["Motion behavior", "motion"],
      ["Incident and evidence window", "incidents"],
      ["Detection behavior", "detection"],
      ["PTZ protection", "ptz"],
      ["Zones", "zones"],
      ["Background timing", "timers"],
    ];
  }

  function renderRuntime(runtime) {
    pipelineMode.textContent = runtime.pipeline_mode || "-";
    streamState.textContent = runtime.stream_state || "-";
    senderStatus.textContent = runtime.sender_status || "-";
    ringCount.textContent = String(runtime.ring_buffer_frames ?? "-");

    const ok = runtime.stream_state === "streaming" || runtime.pipeline_mode === "running";
    runtimePill.textContent = ok ? "Active" : "Idle";
    runtimePill.className = `status-pill ${ok ? "ok" : "neutral"}`;
  }

  function renderPreview(preview) {
    if (!preview.available) {
      previewStage.classList.add("empty");
      previewStage.innerHTML = `<p>${escapeHtml(preview.message || "No preview frame yet.")}</p>`;
      return;
    }

    previewStage.classList.remove("empty");
    previewStage.innerHTML = `
      <img
        class="preview-image"
        src="data:image/jpeg;base64,${preview.image_jpeg_b64}"
        alt="Latest edge preview frame"
      >
    `;
  }

  async function restartPipeline() {
    let msg = document.getElementById("save-settings-msg");

    try {
        const response = await fetch("/api/runtime/restart", {
        method: "POST",
        });

        const result = await response.json();

        if (!response.ok || !result.accepted) {
        throw new Error(result.message || "Restart failed.");
        }

        msg.textContent = result.message || "Restart requested.";
        msg.className = "ok-text";
    } catch (err) {
        msg.textContent = err.message || "Restart failed.";
        msg.className = "danger-text";
    }
  }

  function inputTypeForValue(value) {
    if (typeof value === "boolean") return "checkbox";
    if (typeof value === "number") return "number";
    if (Array.isArray(value) || (value && typeof value === "object")) return "textarea";
    return "text";
  }

  function renderField(key, value) {
    const meta = fieldMeta()[key] || key;
    const type = inputTypeForValue(value);

    if (type === "checkbox") {
      return `
        <label class="setting-field">
          <div class="setting-copy">
            <span class="setting-label">${escapeHtml(meta)}</span>
            <span class="setting-key">${escapeHtml(key)}</span>
          </div>
          <input data-setting-key="${escapeHtml(key)}" type="checkbox" ${value ? "checked" : ""}>
        </label>
      `;
    }

    if (type === "textarea") {
      return `
        <label class="setting-field setting-field-block">
          <div class="setting-copy">
            <span class="setting-label">${escapeHtml(meta)}</span>
            <span class="setting-key">${escapeHtml(key)}</span>
          </div>
          <textarea data-setting-key="${escapeHtml(key)}" rows="4">${escapeHtml(JSON.stringify(value, null, 2))}</textarea>
        </label>
      `;
    }

    return `
      <label class="setting-field setting-field-block">
        <div class="setting-copy">
          <span class="setting-label">${escapeHtml(meta)}</span>
          <span class="setting-key">${escapeHtml(key)}</span>
        </div>
        <input
          data-setting-key="${escapeHtml(key)}"
          type="${type}"
          value="${escapeHtml(value ?? "")}"
          step="any"
        >
      </label>
    `;
  }

  function renderSettings(settings) {
    latestSettings = settings;

    settingsGroups.innerHTML = sectionOrder().map(([title, groupKey]) => {
        const values = settings[groupKey] || {};
        const rows = Object.entries(values).map(([key, value]) => renderField(key, value)).join("");

        return `
        <article class="settings-card">
            <div class="settings-card-head">
            <h3>${escapeHtml(title)}</h3>
            </div>
            <div class="setting-list">${rows}</div>
        </article>
        `;
    }).join("") + `
        <div class="settings-actions">
        <button id="save-settings-btn" class="btn primary" type="button">Save Changes</button>
        <button id="reset-settings-btn" class="btn ghost" type="button">Reset Unsaved Changes</button>
        <button id="restart-pipeline-btn" class="btn ghost" type="button">Restart Edge Pipeline</button>
        <span id="save-settings-msg" class="muted">Change fields and save them to the edge PC.</span>
        </div>
    `;

    const saveBtn = document.getElementById("save-settings-btn");
    const resetBtn = document.getElementById("reset-settings-btn");
    const restartBtn = document.getElementById("restart-pipeline-btn");

    if (saveBtn) {
        saveBtn.addEventListener("click", saveSettings);
    }

    if (resetBtn) {
        resetBtn.addEventListener("click", resetSettings);
    }

    if (restartBtn) {
        restartBtn.addEventListener("click", restartPipeline);
    }
  }

  function collectSettingsFromForm() {
  const payload = {};
  const original = flattenSettings(latestSettings || {});

  document.querySelectorAll("[data-setting-key]").forEach((el) => {
    const key = el.getAttribute("data-setting-key");
    let value;

    if (el.type === "checkbox") {
      value = el.checked;
    } else if (el.tagName === "TEXTAREA") {
      try {
        value = JSON.parse(el.value);
      } catch (_err) {
        value = el.value;
      }
    } else if (el.type === "number") {
      value = el.value === "" ? "" : Number(el.value);
    } else {
      value = el.value;
    }

    const before = original[key];

    if (JSON.stringify(value) !== JSON.stringify(before)) {
      payload[key] = value;
    }
  });

  return payload;
}

  async function saveSettings() {
    let msg = document.getElementById("save-settings-msg");
    const payload = collectSettingsFromForm();

    if (Object.keys(payload).length === 0) {
        msg.textContent = "No changes to save.";
        msg.className = "muted";
        return;
    }

    try {
        const response = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        cache: "no-store",
        });

        const result = await response.json();

        if (!response.ok) {
        throw new Error(result.detail || "Failed to save settings.");
        }

        latestSettings = mergeSettingsWithPayload(latestSettings, payload);
        renderSettings(latestSettings);
        msg = document.getElementById("save-settings-msg");

        const appliedCount = Array.isArray(result.applied_keys)
        ? result.applied_keys.length
        : 0;
        const restartCount = Array.isArray(result.restart_required_keys)
        ? result.restart_required_keys.length
        : 0;

        if (appliedCount > 0 && restartCount === 0) {
        msg.textContent = "Changes applied immediately.";
        } else if (appliedCount === 0 && restartCount > 0) {
        msg.textContent = "Changes saved. Restart required.";
        } else if (appliedCount > 0 && restartCount > 0) {
        msg.textContent =
            "Some changes applied immediately. Restart required for the rest.";
        } else {
        msg.textContent = result.message || "Settings saved.";
        }

        msg.className = "ok-text";
    } catch (err) {
        msg.textContent = err.message || "Failed to save settings.";
        msg.className = "danger-text";
    }
  }

  async function resetSettings() {
    let msg = document.getElementById("save-settings-msg");

    try {
        const refreshed = await fetch(`/api/settings?ts=${Date.now()}`, {
        cache: "no-store",
        }).then((r) => r.json());

        latestSettings = refreshed;
        renderSettings(refreshed);

        msg = document.getElementById("save-settings-msg");
        msg.textContent = "Unsaved changes were discarded.";
        msg.className = "muted";
    } catch (err) {
        msg.textContent = err.message || "Failed to reset settings.";
        msg.className = "danger-text";
    }
  }

  function flattenSettings(settings) {
    const flat = {};
    for (const [, groupKey] of sectionOrder()) {
        const group = settings[groupKey] || {};
        for (const [key, value] of Object.entries(group)) {
        flat[key] = value;
        }
    }
    return flat;
 }

 function mergeSettingsWithPayload(settings, payload) {
  const next = JSON.parse(JSON.stringify(settings || {}));

  for (const [, groupKey] of sectionOrder()) {
    if (!next[groupKey]) {
      next[groupKey] = {};
    }
  }

  for (const [key, value] of Object.entries(payload)) {
    let placed = false;

    for (const [, groupKey] of sectionOrder()) {
      if (
        next[groupKey] &&
        Object.prototype.hasOwnProperty.call(next[groupKey], key)
      ) {
        next[groupKey][key] = value;
        placed = true;
        break;
      }
    }

    if (!placed) {
      next.identity[key] = value;
    }
  }

  return next;
}

  async function loadAll() {
    const [runtimeRes, previewRes, settingsRes] = await Promise.all([
        fetch(`/api/runtime?ts=${Date.now()}`, { cache: "no-store" }),
        fetch(`/api/preview?ts=${Date.now()}`, { cache: "no-store" }),
        fetch(`/api/settings?ts=${Date.now()}`, { cache: "no-store" }),
    ]);

    const runtime = await runtimeRes.json();
    const preview = await previewRes.json();
    const settings = await settingsRes.json();

    renderRuntime(runtime);
    renderPreview(preview);
    renderSettings(settings);
 }

  await loadAll();
  setInterval(async () => {
    try {
        const runtime = await fetch("/api/runtime").then((r) => r.json());
        const preview = await fetch("/api/preview").then((r) => r.json());
        renderRuntime(runtime);
        renderPreview(preview);
    } catch (_err) {
        runtimePill.textContent = "Offline";
        runtimePill.className = "status-pill err";
    }
  }, 3000);
})();
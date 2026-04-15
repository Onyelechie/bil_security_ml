(async function () {
  const runtimePill = document.getElementById("runtime-pill");
  const pipelineMode = document.getElementById("pipeline-mode");
  const streamState = document.getElementById("stream-state");
  const senderStatus = document.getElementById("sender-status");
  const ringCount = document.getElementById("ring-count");
  const previewStage = document.getElementById("preview-stage");
  const settingsGroups = document.getElementById("settings-groups");

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
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

  function renderSettings(settings) {
    const sections = [
      ["Identity and connection", settings.identity],
      ["Camera and stream behavior", settings.stream],
      ["Motion behavior", settings.motion],
      ["Incident and evidence window", settings.incidents],
      ["Detection behavior", settings.detection],
      ["PTZ protection", settings.ptz],
      ["Zones", settings.zones],
      ["Background timing", settings.timers],
    ];

    settingsGroups.innerHTML = sections.map(([title, values]) => {
      const rows = Object.entries(values || {}).map(([key, value]) => `
        <div class="setting-row">
          <span class="setting-key">${escapeHtml(key)}</span>
          <span class="setting-value">${escapeHtml(JSON.stringify(value))}</span>
        </div>
      `).join("");

      return `
        <article class="settings-card">
          <h3>${escapeHtml(title)}</h3>
          <div class="setting-list">${rows}</div>
        </article>
      `;
    }).join("");
  }

  async function loadAll() {
    const [runtimeRes, previewRes, settingsRes] = await Promise.all([
      fetch("/api/runtime"),
      fetch("/api/preview"),
      fetch("/api/settings"),
    ]);

    const runtime = await runtimeRes.json();
    const preview = await previewRes.json();
    const settings = await settingsRes.json();

    renderRuntime(runtime);
    renderPreview(preview);
    renderSettings(settings);
  }

  await loadAll();
  setInterval(loadAll, 3000);
})();
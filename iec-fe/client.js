(() => {
  let websocket = null;
  let recorder = null;
  let mediaStream = null;
  let isRecording = false;
  const chunkMs = 100; // send small chunks frequently

  const wsUrlInput = document.getElementById("wsUrl");
  const wsDefaultSpan = document.getElementById("wsDefault");
  const toggleBtn = document.getElementById("toggleBtn");
  const statusEl = document.getElementById("status");
  const logEl = document.getElementById("log");

  // Reconfig controls
  const apiBaseUrlInput = document.getElementById("apiBaseUrl");
  const apiKeyInput = document.getElementById("apiKey");
  const languageInput = document.getElementById("language");
  const modelInput = document.getElementById("model");
  const targetLanguageInput = document.getElementById("targetLanguage");
  const backendInput = document.getElementById("backend");
  const taskInput = document.getElementById("task");
  const btnSwitch = document.getElementById("btnSwitch");
  const btnReinit = document.getElementById("btnReinit");
  const btnInfo = document.getElementById("btnInfo");

  function computeDefaultWsUrl() {
    const host = window.location.hostname || "localhost";
    const port = window.location.port;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    return `${protocol}://${host}${port ? ":" + port : ""}/stream/1`;
  }

  function computeDefaultApiBase() {
    const origin = window.location.origin || "http://localhost:8000";
    return origin.replace("ws://", "http://").replace("wss://", "https://");
  }

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function logLine(line) {
    const ts = new Date().toISOString();
    logEl.textContent += `[${ts}] ${line}\n`;
    logEl.scrollTop = logEl.scrollHeight;
    console.log(line);
  }

  async function start() {
    if (isRecording) return;

    const url = wsUrlInput.value.trim();
    if (!url.startsWith("ws://") && !url.startsWith("wss://")) {
      setStatus("Invalid WS URL");
      return;
    }

    try {
      setStatus("Connecting WS...");
      websocket = new WebSocket(url);

      await new Promise((resolve, reject) => {
        websocket.onopen = () => {
          setStatus("WS connected");
          resolve();
        };
        websocket.onerror = (e) => {
          setStatus("WS error");
          reject(e);
        };
        websocket.onclose = () => {
          logLine("WebSocket closed");
          if (isRecording) stop();
        };
        websocket.onmessage = (event) => {
          logLine(`Message: ${event.data}`);
        };
      });

      setStatus("Getting microphone...");
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      let mimeType = "audio/webm;codecs=opus";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "audio/webm";
      }
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "";
      }

      recorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
          websocket.send(e.data);
        }
      };

      recorder.onstop = () => {
        logLine("Recorder stopped");
      };

      recorder.start(chunkMs);
      isRecording = true;
      toggleBtn.textContent = "Stop";
      setStatus("Recording...");
      logLine(`Streaming audio chunks every ${chunkMs}ms`);
    } catch (err) {
      console.error(err);
      setStatus("Failed to start");
      logLine(`Error: ${err?.message || err}`);
      if (recorder && recorder.state !== "inactive") recorder.stop();
      recorder = null;
      if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
      }
      if (websocket && websocket.readyState === WebSocket.OPEN) websocket.close();
      websocket = null;
      isRecording = false;
      toggleBtn.textContent = "Start";
    }
  }

  function sendFinalEmptyChunk() {
    try {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        const emptyBlob = new Blob([], { type: "audio/webm" });
        websocket.send(emptyBlob);
        logLine("Sent final empty chunk");
      }
    } catch {}
  }

  function stop() {
    if (!isRecording) return;

    setStatus("Stopping...");
    sendFinalEmptyChunk();

    try {
      if (recorder && recorder.state !== "inactive") recorder.stop();
    } catch {}
    recorder = null;

    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }

    if (websocket) {
      try { websocket.close(); } catch {}
      websocket = null;
    }

    isRecording = false;
    toggleBtn.textContent = "Start";
    setStatus("Idle");
  }

  // REST helpers for backend reconfiguration
  async function apiRequest(path, method, body) {
    const base = (apiBaseUrlInput.value || computeDefaultApiBase()).trim().replace(/\/$/, "");
    const url = `${base}${path}`;
    const headers = { "Content-Type": "application/json" };
    const apiKey = apiKeyInput.value.trim();
    if (apiKey) headers["X-API-Key"] = apiKey;

    const res = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    let text;
    try { text = await res.text(); } catch { text = ""; }
    let data;
    try { data = text ? JSON.parse(text) : null; } catch { data = text; }

    if (!res.ok) {
      const msg = typeof data === "string" ? data : (data?.detail || JSON.stringify(data));
      throw new Error(`${res.status} ${res.statusText}: ${msg}`);
    }
    return data;
  }

  async function switchLanguage() {
    const language = languageInput.value.trim();
    if (!language) { logLine("Please enter language to switch."); return; }

    const payload = {};
    const model = modelInput.value.trim();
    const target_language = targetLanguageInput.value.trim();
    const backend = backendInput.value.trim();
    const task = taskInput.value.trim();
    if (model) payload.model = model;
    if (target_language) payload.target_language = target_language;
    if (backend) payload.backend = backend;
    if (task) payload.task = task;

    try {
      logLine(`Switching language to ${language} ...`);
      const data = await apiRequest(`/reconfig_model/switch/${encodeURIComponent(language)}`, "POST", payload);
      logLine(`Switch success: ${JSON.stringify(data)}`);
    } catch (e) {
      logLine(`Switch failed: ${e.message}`);
    }
  }

  async function reinitializeEngine() {
    const language = languageInput.value.trim();
    if (!language) { logLine("Please enter language to reinitialize."); return; }

    const payload = {};
    const model = modelInput.value.trim();
    const target_language = targetLanguageInput.value.trim();
    const backend = backendInput.value.trim();
    const task = taskInput.value.trim();
    if (model) payload.model = model;
    if (target_language) payload.target_language = target_language;
    if (backend) payload.backend = backend;
    if (task) payload.task = task;

    try {
      logLine(`Reinitializing engine for ${language} ...`);
      const data = await apiRequest(`/reconfig_model/reinitialize/${encodeURIComponent(language)}`, "POST", payload);
      logLine(`Reinit success: ${JSON.stringify(data)}`);
    } catch (e) {
      logLine(`Reinit failed: ${e.message}`);
    }
  }

  async function getEngineInfo() {
    try {
      const data = await apiRequest(`/reconfig_model/info`, "GET");
      logLine(`Engine info: ${JSON.stringify(data)}`);
    } catch (e) {
      logLine(`Info failed: ${e.message}`);
    }
  }

  // UI wiring
  toggleBtn.addEventListener("click", () => {
    if (!isRecording) start();
    else stop();
  });

  btnSwitch?.addEventListener("click", switchLanguage);
  btnReinit?.addEventListener("click", reinitializeEngine);
  btnInfo?.addEventListener("click", getEngineInfo);

  // Init defaults
  const defaultWs = computeDefaultWsUrl();
  wsDefaultSpan.textContent = defaultWs;
  wsUrlInput.value = defaultWs;

  const defaultApi = computeDefaultApiBase();
  apiBaseUrlInput.value = defaultApi;

  if (window.location.hostname === "0.0.0.0") {
    logLine("Note: Some browsers block mic access on 0.0.0.0. Prefer localhost.");
  }
})(); 
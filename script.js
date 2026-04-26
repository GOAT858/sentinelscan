const MODEL_FEATURE_ORDER = ["od", "ratio", "quality", "separation_norm", "dynamic_range"];

class TrainedModelRuntime {
  constructor() {
    this.ready = false;
    this.usable = false;
    this.model = null;
    this.featureNames = [...MODEL_FEATURE_ORDER];
    this.means = new Array(MODEL_FEATURE_ORDER.length).fill(0);
    this.stds = new Array(MODEL_FEATURE_ORDER.length).fill(1);
    this.weights = new Array(MODEL_FEATURE_ORDER.length).fill(0);
    this.intercept = 0;
    this.threshold = 0.5;
  }

  static sigmoid(z) {
    const clipped = Math.max(-60, Math.min(60, z));
    return 1 / (1 + Math.exp(-clipped));
  }

  static asNum(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  validate(payload) {
    if (!payload || typeof payload !== "object") {
      throw new Error("Invalid model payload");
    }

    const features = Array.isArray(payload.features) ? payload.features : MODEL_FEATURE_ORDER;
    const coeffs = payload.coefficients;
    const means = payload?.scaler?.means;
    const stds = payload?.scaler?.stds;

    if (!Array.isArray(coeffs) || coeffs.length !== features.length) {
      throw new Error("Model coefficients mismatch feature dimensions");
    }

    if (!Array.isArray(means) || !Array.isArray(stds) || means.length !== features.length || stds.length !== features.length) {
      throw new Error("Model scaler is malformed");
    }
  }

  async load(url = "model/sentinel_model.json") {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`Model fetch failed (${res.status})`);
    }

    const payload = await res.json();
    this.validate(payload);

    this.model = payload;
    this.featureNames = payload.features;
    this.means = payload.scaler.means.map((v) => TrainedModelRuntime.asNum(v, 0));
    this.stds = payload.scaler.stds.map((v) => {
      const n = TrainedModelRuntime.asNum(v, 1);
      return Math.abs(n) < 1e-9 ? 1 : n;
    });
    this.weights = payload.coefficients.map((v) => TrainedModelRuntime.asNum(v, 0));
    this.intercept = TrainedModelRuntime.asNum(payload.intercept, 0);
    this.threshold = TrainedModelRuntime.asNum(payload.threshold, 0.5);

    const valSamples = Number(payload.validation_samples ?? payload.samples?.validation ?? 0);
    const accuracy = Number(payload.metrics?.accuracy ?? 0);

    this.ready = true;
    this.usable = Number.isFinite(valSamples) && valSamples > 0 && Number.isFinite(accuracy) && accuracy > 0;
    return payload;
  }

  get isUsable() {
    return this.ready && this.usable;
  }

  predict(featureMap) {
    if (!this.isUsable) {
      throw new Error("Trained model not usable");
    }

    let z = this.intercept;
    for (let i = 0; i < this.featureNames.length; i += 1) {
      const name = this.featureNames[i];
      const raw = TrainedModelRuntime.asNum(featureMap[name], 0);
      const norm = (raw - this.means[i]) / this.stds[i];
      z += this.weights[i] * norm;
    }

    return TrainedModelRuntime.sigmoid(z);
  }
}

class SentinelVisionModel {
  static clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }

  static smooth(values, radius = 5) {
    const out = new Array(values.length).fill(0);
    for (let i = 0; i < values.length; i += 1) {
      let sum = 0;
      let count = 0;
      for (let k = -radius; k <= radius; k += 1) {
        const idx = i + k;
        if (idx >= 0 && idx < values.length) {
          sum += values[idx];
          count += 1;
        }
      }
      out[i] = count ? sum / count : values[i];
    }
    return out;
  }

  static percentile(values, p) {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const idx = Math.floor(this.clamp(p, 0, 1) * (sorted.length - 1));
    return sorted[idx];
  }

  static detectPeaks(signal, minDistance = 18, topN = 4) {
    const peaks = [];
    for (let i = 1; i < signal.length - 1; i += 1) {
      if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
        peaks.push({ index: i, value: signal[i] });
      }
    }

    peaks.sort((a, b) => b.value - a.value);
    const selected = [];
    for (const peak of peaks) {
      if (selected.every((s) => Math.abs(s.index - peak.index) >= minDistance)) {
        selected.push(peak);
      }
      if (selected.length >= topN) break;
    }

    return selected.sort((a, b) => a.index - b.index);
  }

  static projection(imageData, roi, axis = "x") {
    const { data, width } = imageData;
    const values = [];

    if (axis === "x") {
      for (let x = roi.x; x < roi.x + roi.w; x += 1) {
        let sum = 0;
        for (let y = roi.y; y < roi.y + roi.h; y += 1) {
          const idx = (y * width + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          sum += 255 - luma;
        }
        values.push(sum / roi.h);
      }
    } else {
      for (let y = roi.y; y < roi.y + roi.h; y += 1) {
        let sum = 0;
        for (let x = roi.x; x < roi.x + roi.w; x += 1) {
          const idx = (y * width + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          sum += 255 - luma;
        }
        values.push(sum / roi.w);
      }
    }

    return values;
  }

  static candidate(imageData, c) {
    const raw = this.projection(imageData, c.roi, c.axis);
    const smooth = this.smooth(raw, 5);
    const baseline = this.percentile(smooth, 0.44) || 1;
    const centered = smooth.map((v) => Math.max(0, v - baseline));

    const minDistance = Math.max(14, Math.floor(centered.length * 0.08));
    const peaks = this.detectPeaks(centered, minDistance, 4);
    if (peaks.length < 2) {
      return { ok: false, quality: 0 };
    }

    const strongest = [...peaks].sort((a, b) => b.value - a.value);
    const control = strongest[0];
    const test = strongest[1];

    const separationNorm = Math.abs(control.index - test.index) / centered.length;
    const dynamicRange = this.clamp((this.percentile(smooth, 0.92) - baseline) / (baseline + 1e-6), 0, 3);
    const prominence = this.clamp(test.value / (baseline + 1e-6), 0, 3);

    const quality = this.clamp(0.2 + 0.34 * separationNorm + 0.3 * (prominence / 3) + 0.16 * (dynamicRange / 3), 0, 1);

    return {
      ok: true,
      axis: c.axis,
      label: c.label,
      baseline,
      control,
      test,
      separationNorm,
      dynamicRange,
      quality
    };
  }

  static analyze(imageData, modelRuntime = null) {
    const { width, height } = imageData;

    const candidates = [
      {
        label: "horizontal-strip",
        axis: "x",
        roi: {
          x: Math.floor(width * 0.1),
          y: Math.floor(height * 0.34),
          w: Math.max(24, Math.floor(width * 0.8)),
          h: Math.max(24, Math.floor(height * 0.3))
        }
      },
      {
        label: "vertical-strip",
        axis: "y",
        roi: {
          x: Math.floor(width * 0.34),
          y: Math.floor(height * 0.1),
          w: Math.max(24, Math.floor(width * 0.3)),
          h: Math.max(24, Math.floor(height * 0.8))
        }
      }
    ];

    const valid = candidates
      .map((c) => this.candidate(imageData, c))
      .filter((c) => c.ok)
      .sort((a, b) => b.quality - a.quality);

    if (!valid.length || valid[0].quality < 0.28) {
      return {
        ok: false,
        message: "Unable to resolve cassette lines. Improve lighting and alignment, then scan again."
      };
    }

    const best = valid[0];
    const epsilon = 1e-6;
    const transmittance = this.clamp((best.baseline + epsilon) / (best.baseline + best.test.value + epsilon), 0.01, 0.99);
    const od = -Math.log10(transmittance);
    const ratio = this.clamp(best.test.value / (best.control.value + epsilon), 0, 3);

    const features = {
      od,
      ratio,
      quality: best.quality,
      separation_norm: best.separationNorm,
      dynamic_range: best.dynamicRange
    };

    let malignancyProb;
    let source = "Heuristic Engine";
    let threshold = 0.5;

    if (modelRuntime?.isUsable) {
      malignancyProb = modelRuntime.predict(features);
      source = "Trained Model";
      threshold = modelRuntime.threshold;
    } else {
      const heuristic = this.clamp((ratio * 6.4) + (od * 2.3) + (best.quality * 1.15), 0, 10);
      malignancyProb = 1 / (1 + Math.exp(-1.04 * (heuristic - 4.7)));
    }

    const sentinelScore = this.clamp(malignancyProb * 10, 0, 10);
    const confidence = this.clamp(0.3 + (best.quality * 0.44) + (Math.abs(malignancyProb - threshold) * 0.4), 0.2, 0.99);

    return {
      ok: true,
      od,
      sentinelScore,
      malignancyProb,
      confidence,
      source,
      threshold,
      features,
      orientation: best.axis === "x" ? "landscape" : "portrait",
      quality: best.quality
    };
  }
}

class CameraEngine {
  constructor(videoElement) {
    this.videoElement = videoElement;
    this.stream = null;
    this.devices = [];
    this.currentDeviceId = null;
    this.currentLabel = "";
    this.isStarting = false;
  }

  static isSecureOrigin() {
    return (
      window.isSecureContext ||
      location.hostname === "localhost" ||
      location.hostname === "127.0.0.1" ||
      location.hostname === "::1"
    );
  }

  static isSupported() {
    return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  async getPermissionState() {
    if (!navigator.permissions?.query) return "unknown";
    try {
      const result = await navigator.permissions.query({ name: "camera" });
      return result.state;
    } catch {
      return "unknown";
    }
  }

  async refreshDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) {
      this.devices = [];
      return;
    }

    const all = await navigator.mediaDevices.enumerateDevices();
    this.devices = all.filter((d) => d.kind === "videoinput");

    if (!this.currentDeviceId && this.devices.length) {
      const rear = this.devices.find((d) => /rear|back|environment/i.test(d.label));
      this.currentDeviceId = rear?.deviceId || this.devices[0].deviceId;
      this.currentLabel = rear?.label || this.devices[0].label || "Camera";
    }
  }

  hasMultipleCameras() {
    return this.devices.length > 1;
  }

  stop() {
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
    this.videoElement.srcObject = null;
  }

  constraints() {
    const attempts = [];

    if (this.currentDeviceId) {
      attempts.push({
        video: {
          deviceId: { exact: this.currentDeviceId },
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false
      });
    }

    attempts.push({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      },
      audio: false
    });

    attempts.push({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    attempts.push({ video: true, audio: false });
    return attempts;
  }

  async start() {
    if (this.isStarting) {
      throw new Error("Camera is already starting.");
    }

    if (!CameraEngine.isSupported()) {
      throw new Error("Browser does not support camera APIs.");
    }

    if (!CameraEngine.isSecureOrigin()) {
      throw new Error("Camera requires HTTPS or localhost.");
    }

    this.isStarting = true;
    this.stop();
    await this.refreshDevices();

    const errors = [];
    for (const constraints of this.constraints()) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        this.stream = stream;
        this.videoElement.srcObject = stream;
        await this.videoElement.play();

        const track = stream.getVideoTracks()[0];
        const settings = track?.getSettings?.() || {};
        if (settings.deviceId) {
          this.currentDeviceId = settings.deviceId;
        }

        await this.refreshDevices();
        this.currentLabel = this.devices.find((d) => d.deviceId === this.currentDeviceId)?.label || "Camera";
        this.isStarting = false;
        return { ok: true, label: this.currentLabel || "Camera" };
      } catch (error) {
        errors.push(error);
      }
    }

    this.isStarting = false;
    const preferred = errors.find((e) => e?.name === "NotAllowedError") || errors[0];
    const message = preferred?.message || "Unknown camera error";
    throw new Error(message);
  }

  async switchCamera() {
    await this.refreshDevices();
    if (this.devices.length <= 1) {
      throw new Error("No secondary camera found.");
    }

    const currentIndex = this.devices.findIndex((d) => d.deviceId === this.currentDeviceId);
    const next = this.devices[(currentIndex + 1) % this.devices.length];
    this.currentDeviceId = next.deviceId;
    this.currentLabel = next.label || "Camera";
    return this.start();
  }
}

class SentinelApp {
  constructor() {
    this.tabButtons = Array.from(document.querySelectorAll(".tab"));
    this.tabPanels = Array.from(document.querySelectorAll(".tab-panel"));

    this.runtimeStatus = document.getElementById("runtime-status");

    this.video = document.getElementById("camera-preview");
    this.canvas = document.getElementById("analysis-canvas");
    this.scanOverlay = document.getElementById("scan-overlay");
    this.cameraPlaceholder = document.getElementById("camera-placeholder");

    this.startBtn = document.getElementById("start-camera-btn");
    this.switchBtn = document.getElementById("switch-camera-btn");
    this.captureBtn = document.getElementById("capture-btn");
    this.uploadInput = document.getElementById("upload-input");
    this.retakeBtn = document.getElementById("retake-btn");
    this.analyzeBtn = document.getElementById("analyze-btn");

    this.cameraNote = document.getElementById("camera-note");
    this.previewWrap = document.getElementById("preview-wrap");
    this.previewImage = document.getElementById("captured-preview");
    this.resolution = document.getElementById("resolution");

    this.systemStandby = document.getElementById("system-standby");
    this.biomarkerSaturation = document.getElementById("biomarker-saturation");

    this.metricOd = document.getElementById("metric-od");
    this.metricScore = document.getElementById("metric-score");
    this.metricProb = document.getElementById("metric-prob");
    this.metricSource = document.getElementById("metric-source");

    this.latentCoordinates = document.getElementById("latent-coordinates");
    this.signatureIndex = document.getElementById("signature-index");
    this.modelValidation = document.getElementById("model-validation");

    this.vaeRuntime = document.getElementById("vae-runtime");

    this.directiveCard = document.getElementById("directive-card");
    this.clinicalDirective = document.getElementById("clinical-directive");
    this.clinicalSupport = document.getElementById("clinical-support");

    this.camera = new CameraEngine(this.video);
    this.model = new TrainedModelRuntime();
    this.selectedDataUrl = null;

    this.bindTabs();
    this.bindEvents();
    this.initialize();

    window.addEventListener("beforeunload", () => this.camera.stop());
  }

  bindTabs() {
    for (const button of this.tabButtons) {
      button.addEventListener("click", () => this.setActiveTab(button.dataset.tab));
    }
  }

  bindEvents() {
    this.startBtn.addEventListener("click", () => this.startCamera());
    this.switchBtn.addEventListener("click", () => this.switchCamera());
    this.captureBtn.addEventListener("click", () => this.captureFrame());
    this.retakeBtn.addEventListener("click", () => this.resetSelection());
    this.analyzeBtn.addEventListener("click", () => this.analyzeSelection());
    this.uploadInput.addEventListener("change", (event) => this.handleUpload(event));

    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") {
        this.camera.stop();
        this.cameraNote.textContent = "Camera paused while app is in background.";
      }
    });

    if (navigator.mediaDevices?.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", async () => {
        try {
          await this.camera.refreshDevices();
          this.switchBtn.classList.toggle("hidden", !this.camera.hasMultipleCameras());
        } catch {
          this.switchBtn.classList.add("hidden");
        }
      });
    }
  }

  setActiveTab(tabName) {
    for (const button of this.tabButtons) {
      const active = button.dataset.tab === tabName;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    }

    for (const panel of this.tabPanels) {
      panel.classList.toggle("active", panel.id === tabName);
    }
  }

  async initialize() {
    this.runtimeStatus.textContent = "Runtime: loading model and camera...";

    await Promise.all([this.initializeModel(), this.initializeCamera()]);

    this.runtimeStatus.textContent = "Runtime: ready.";
  }

  async initializeModel() {
    try {
      const payload = await this.model.load("model/sentinel_model.json");
      const accuracy = Number(payload.metrics?.accuracy ?? 0);
      const train = payload.samples?.train ?? "--";
      const val = payload.samples?.validation ?? payload.validation_samples ?? "--";

      if (this.model.isUsable) {
        this.modelValidation.textContent = `Model loaded | Accuracy ${(accuracy * 100).toFixed(2)}% | Train ${train} | Val ${val}`;
        this.vaeRuntime.textContent = `${payload.model_name || "Sentinel model"} active. Decision threshold ${(payload.threshold ?? 0.5).toFixed(3)}.`;
      } else {
        this.modelValidation.textContent = "Model file is a placeholder. Using heuristic fallback.";
        this.vaeRuntime.textContent = "Placeholder model loaded. Add validated samples for full model output.";
      }
    } catch (error) {
      this.modelValidation.textContent = `Model load failed (${error.message}). Using heuristic fallback.`;
      this.vaeRuntime.textContent = "Heuristic mode active. Add model/sentinel_model.json to enable trained inference.";
    }
  }

  async initializeCamera() {
    if (!CameraEngine.isSecureOrigin()) {
      this.cameraPlaceholder.classList.remove("hidden");
      this.cameraNote.textContent = "Camera blocked: open this app on HTTPS or localhost.";
      return;
    }

    const permission = await this.camera.getPermissionState();
    if (permission === "denied") {
      this.cameraPlaceholder.classList.remove("hidden");
      this.cameraNote.textContent = "Camera permission denied. Enable access in browser settings.";
      return;
    }

    this.cameraPlaceholder.classList.remove("hidden");
    this.cameraNote.textContent = "Press Start Camera to begin.";

    try {
      await this.camera.refreshDevices();
      this.switchBtn.classList.toggle("hidden", !this.camera.hasMultipleCameras());
    } catch {
      this.switchBtn.classList.add("hidden");
    }
  }

  async startCamera() {
    this.startBtn.disabled = true;
    this.switchBtn.disabled = true;
    this.cameraNote.textContent = "Starting camera...";

    try {
      const info = await this.camera.start();
      this.cameraPlaceholder.classList.add("hidden");
      this.scanOverlay.classList.remove("hidden");
      this.cameraNote.textContent = `Camera active: ${info.label}. Center the strip and capture.`;
      this.switchBtn.classList.toggle("hidden", !this.camera.hasMultipleCameras());
    } catch (error) {
      this.cameraPlaceholder.classList.remove("hidden");
      this.cameraNote.textContent = `Camera startup failed: ${error.message}`;
    } finally {
      this.startBtn.disabled = false;
      this.switchBtn.disabled = false;
    }
  }

  async switchCamera() {
    this.switchBtn.disabled = true;
    this.cameraNote.textContent = "Switching camera...";
    try {
      const info = await this.camera.switchCamera();
      this.cameraPlaceholder.classList.add("hidden");
      this.cameraNote.textContent = `Now using: ${info.label}`;
    } catch (error) {
      this.cameraNote.textContent = `Switch failed: ${error.message}`;
    } finally {
      this.switchBtn.disabled = false;
    }
  }

  captureFrame() {
    if (!this.video.videoWidth || !this.video.videoHeight) {
      this.cameraPlaceholder.classList.remove("hidden");
      this.cameraNote.textContent = "No live camera feed yet. Start camera first.";
      return;
    }

    const ctx = this.canvas.getContext("2d");
    this.canvas.width = this.video.videoWidth;
    this.canvas.height = this.video.videoHeight;
    ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

    this.selectedDataUrl = this.canvas.toDataURL("image/jpeg", 0.95);
    this.previewImage.src = this.selectedDataUrl;
    this.resolution.textContent = `${this.canvas.width} × ${this.canvas.height}`;
    this.previewWrap.classList.remove("hidden");
    this.retakeBtn.classList.remove("hidden");
    this.analyzeBtn.classList.remove("hidden");
    this.systemStandby.textContent = "Frame captured. Ready for analysis.";
    this.cameraNote.textContent = "Captured. Press Analyze.";
  }

  handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      this.selectedDataUrl = reader.result;
      this.previewImage.src = this.selectedDataUrl;
      this.resolution.textContent = "Uploaded image";
      this.previewWrap.classList.remove("hidden");
      this.retakeBtn.classList.remove("hidden");
      this.analyzeBtn.classList.remove("hidden");
      this.systemStandby.textContent = "Uploaded image ready for analysis.";
      this.cameraNote.textContent = "Upload complete. Press Analyze to continue.";
    };
    reader.readAsDataURL(file);
  }

  async analyzeSelection() {
    if (!this.selectedDataUrl) {
      this.cameraNote.textContent = "Capture or upload a frame first.";
      return;
    }

    try {
      const image = await this.loadImage(this.selectedDataUrl);
      const sample = this.sampleImage(image);
      const result = SentinelVisionModel.analyze(sample, this.model);

      if (!result.ok) {
        this.systemStandby.textContent = result.message;
        this.biomarkerSaturation.textContent = "Awaiting scan...";
        this.cameraNote.textContent = "Could not read strip clearly. Re-align and try again.";
        return;
      }

      this.applyResult(result);
      this.cameraNote.textContent = "Analysis complete. Review Profile, Model, and Protocol tabs.";
    } catch (error) {
      this.cameraNote.textContent = `Analysis failed: ${error.message}`;
    }
  }

  loadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("Unable to decode image"));
      image.src = src;
    });
  }

  sampleImage(image) {
    const maxSide = 1400;
    let width = image.width;
    let height = image.height;

    if (Math.max(width, height) > maxSide) {
      const ratio = maxSide / Math.max(width, height);
      width = Math.round(width * ratio);
      height = Math.round(height * ratio);
    }

    const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.canvas.width = width;
    this.canvas.height = height;
    ctx.drawImage(image, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    return {
      data: imageData.data,
      width,
      height
    };
  }

  animateNumber(element, to, decimals = 2, suffix = "", duration = 950) {
    const from = 0;
    const start = performance.now();

    const tick = (now) => {
      const p = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      const value = from + (to - from) * eased;
      element.textContent = `${value.toFixed(decimals)}${suffix}`;
      if (p < 1) {
        requestAnimationFrame(tick);
      }
    };

    requestAnimationFrame(tick);
  }

  applyResult(result) {
    this.systemStandby.textContent = "Analysis complete.";
    this.biomarkerSaturation.textContent = `${(result.malignancyProb * 100).toFixed(1)}% signal detected.`;

    this.animateNumber(this.metricOd, result.od, 3);
    this.animateNumber(this.metricScore, result.sentinelScore, 2);
    this.animateNumber(this.metricProb, result.malignancyProb * 100, 1, "%");
    this.metricSource.textContent = result.source;

    const f = result.features;
    const z1 = (f.od * 1.48 - f.ratio * 0.62 + f.quality * 0.4).toFixed(3);
    const z2 = (f.dynamic_range * 0.9 - f.separation_norm * 0.45 + f.ratio * 0.2).toFixed(3);
    const z3 = (result.malignancyProb * 1.7 - 0.8).toFixed(3);

    this.latentCoordinates.textContent = `z = [${z1}, ${z2}, ${z3}] | orientation: ${result.orientation}`;
    this.signatureIndex.textContent = `Signature index ${(result.sentinelScore / 10).toFixed(3)} | quality ${(result.quality * 100).toFixed(1)}%`;

    this.applyClinicalDirective(result);
  }

  applyClinicalDirective(result) {
    const prob = result.malignancyProb;
    const threshold = result.threshold;

    this.directiveCard.classList.remove("low", "medium", "high");

    if (prob < threshold) {
      this.directiveCard.classList.add("low");
      this.clinicalDirective.textContent = "Routine monitoring recommended.";
      this.clinicalSupport.textContent = "No elevated signature detected in this scan.";
    } else if (prob < threshold + 0.18) {
      this.directiveCard.classList.add("medium");
      this.clinicalDirective.textContent = "Repeat scan and review with a clinician.";
      this.clinicalSupport.textContent = "Borderline signature detected. Confirm with a second scan and clinical context.";
    } else {
      this.directiveCard.classList.add("high");
      this.clinicalDirective.textContent = "Prompt clinical follow-up recommended.";
      this.clinicalSupport.textContent = "Strong signature detected. Escalation is advised.";
    }
  }

  resetSelection() {
    this.selectedDataUrl = null;
    this.uploadInput.value = "";
    this.previewImage.src = "";
    this.previewWrap.classList.add("hidden");
    this.retakeBtn.classList.add("hidden");
    this.analyzeBtn.classList.add("hidden");

    this.systemStandby.textContent = "Waiting for a scan to generate model features.";
    this.biomarkerSaturation.textContent = "Awaiting scan...";
    this.cameraNote.textContent = "Selection cleared. Capture or upload a new frame.";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.sentinelApp = new SentinelApp();
});

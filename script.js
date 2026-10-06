(() => {
  "use strict";

  const qs = (selector, root = document) => root.querySelector(selector);
  const qsa = (selector, root = document) => [...root.querySelectorAll(selector)];

  const state = {
    cameraStream: null,
    faceMesh: null,
    cv: null,
    activeSource: null,
    processing: false,
    frameBusy: false,
    frameRequest: 0,
    runToken: 0,
    lastSentAt: 0,
    frameCount: 0,
    uploadUrl: null,
    uploadFile: null,
    dependencies: null,
  };

  const elements = {};

  function setText(element, value) {
    if (element) element.textContent = value;
  }

  function rememberTheme(theme) {
    try {
      window.localStorage.setItem("ssr-theme", theme);
    } catch (error) {
      // Storage can be unavailable in private or restricted browsing modes.
    }
  }

  function readTheme() {
    try {
      return window.localStorage.getItem("ssr-theme");
    } catch (error) {
      return null;
    }
  }

  function applyTheme(theme, persist = true) {
    const nextTheme = theme === "light" ? "light" : "dark";
    document.documentElement.dataset.theme = nextTheme;
    if (elements.themeToggle) {
      elements.themeToggle.setAttribute("aria-pressed", String(nextTheme === "light"));
      elements.themeToggle.setAttribute("aria-label", nextTheme === "light" ? "Switch to dark mode" : "Switch to light mode");
      setText(qs(".theme-label", elements.themeToggle), nextTheme === "light" ? "Dark mode" : "Light mode");
    }
    if (persist) rememberTheme(nextTheme);
  }

  function closeMobileNav() {
    if (!elements.menuToggle || !elements.mobileNav) return;
    elements.menuToggle.setAttribute("aria-expanded", "false");
    elements.mobileNav.hidden = true;
    document.body.classList.remove("menu-open");
  }

  function setupNavigation() {
    if (elements.menuToggle && elements.mobileNav) {
      elements.menuToggle.addEventListener("click", () => {
        const open = elements.menuToggle.getAttribute("aria-expanded") === "true";
        elements.menuToggle.setAttribute("aria-expanded", String(!open));
        elements.mobileNav.hidden = open;
        document.body.classList.toggle("menu-open", !open);
      });

      qsa("a", elements.mobileNav).forEach((link) => link.addEventListener("click", closeMobileNav));
    }

    const sections = ["overview", "demo", "architecture", "results", "team"]
      .map((id) => qs(`#${id}`))
      .filter(Boolean);
    const navLinks = qsa(".desktop-nav a");
    if ("IntersectionObserver" in window && sections.length) {
      const navObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          navLinks.forEach((link) => {
            link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`);
          });
        });
      }, { rootMargin: "-35% 0px -55%" });
      sections.forEach((section) => navObserver.observe(section));
    }
  }

  function setupRevealAnimations() {
    const revealItems = qsa("[data-reveal]");
    if (!("IntersectionObserver" in window)) {
      revealItems.forEach((item) => item.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver((entries, instance) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        instance.unobserve(entry.target);
      });
    }, { threshold: 0.08 });
    revealItems.forEach((item) => observer.observe(item));
  }

  function loadScript(id, source) {
    const existing = document.getElementById(id);
    if (existing) {
      if (existing.dataset.loaded === "true") return Promise.resolve();
      return new Promise((resolve, reject) => {
        existing.addEventListener("load", resolve, { once: true });
        existing.addEventListener("error", () => reject(new Error(`Unable to load ${source}`)), { once: true });
      });
    }

    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.id = id;
      script.src = source;
      script.async = true;
      script.onload = () => {
        script.dataset.loaded = "true";
        resolve();
      };
      script.onerror = () => reject(new Error(`Unable to load ${source}`));
      document.head.appendChild(script);
    });
  }

  function ensureFaceMeshScript() {
    if (window.FaceMesh) return Promise.resolve();
    if (state.dependencies?.faceMeshScript) return state.dependencies.faceMeshScript;
    state.dependencies = state.dependencies || {};
    state.dependencies.faceMeshScript = loadScript(
      "mediapipe-face-mesh-script",
      "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/face_mesh.js",
    );
    return state.dependencies.faceMeshScript;
  }

  function ensureOpenCvScript() {
    if (window.cv && window.cv.Mat) return Promise.resolve(window.cv);
    if (state.dependencies?.openCvScript) return state.dependencies.openCvScript;
    state.dependencies = state.dependencies || {};
    state.dependencies.openCvScript = loadScript("opencv-script", "https://docs.opencv.org/4.x/opencv.js")
      .then(() => new Promise((resolve, reject) => {
        const startedAt = performance.now();
        const checkReady = () => {
          if (window.cv && window.cv.Mat) {
            resolve(window.cv);
            return;
          }
          if (performance.now() - startedAt > 12000) {
            reject(new Error("OpenCV.js runtime did not initialize"));
            return;
          }
          window.setTimeout(checkReady, 80);
        };
        checkReady();
      }));
    return state.dependencies.openCvScript;
  }

  async function prepareVision() {
    if (!state.faceMesh) {
      await ensureFaceMeshScript();
      ensureOpenCvScript().then((cv) => { state.cv = cv; }).catch(() => {
        state.cv = null;
      });

      const FaceMesh = window.FaceMesh;
      if (!FaceMesh) throw new Error("MediaPipe Face Mesh is unavailable");
      state.faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`,
      });
      state.faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.55,
        minTrackingConfidence: 0.55,
      });
      state.faceMesh.onResults(handleFaceMeshResults);
    }
    return state.faceMesh;
  }

  function setStageStatus(status, label) {
    if (!elements.stageStatus) return;
    elements.stageStatus.dataset.state = status;
    setText(elements.stageStatusText, label);
  }

  function resetInference(copy = "A trained CNN-LSTM model and selected vocabulary are not included in the supplied report, so no class or confidence is fabricated.") {
    setText(elements.predictionValue, "Awaiting sequence");
    setText(elements.confidenceValue, "--");
    setText(elements.predictionNote, copy);
    setText(elements.readoutPulse, " waiting");
    if (elements.readoutPulse) elements.readoutPulse.prepend(document.createElement("i"));
    setText(elements.inferenceDetection, "Waiting for input");
    setText(elements.inferenceRoi, "ROI not captured");
    setText(elements.inferenceModel, "CNN-LSTM weights pending");
    qsa(".inference-step", elements.predictionPanel).forEach((step, index) => {
      step.classList.toggle("is-ready", index === 0);
    });
  }

  function resetStats() {
    state.frameCount = 0;
    setText(elements.landmarkValue, "waiting");
    setText(elements.roiValue, "waiting");
    setText(elements.sequenceValue, "0 captured");
    setText(elements.liveCaption, "WAITING FOR INPUT");
    setText(elements.uploadCaption, "VIDEO READY");
    clearCanvas(elements.liveOverlay);
    clearCanvas(elements.uploadOverlay);
    resetInference();
  }

  function clearCanvas(canvas) {
    if (!canvas) return;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  function activeVideo() {
    return state.activeSource === "live" ? elements.liveVideo : elements.uploadVideo;
  }

  function activeOverlay() {
    return state.activeSource === "live" ? elements.liveOverlay : elements.uploadOverlay;
  }

  function resizeCanvasToVideo(canvas, video) {
    if (!canvas || !video) return;
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 360;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
  }

  const lipLandmarkIndices = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310,
    415,
  ];

  function getMouthRoi(landmarks, width, height) {
    const points = lipLandmarkIndices.map((index) => landmarks[index]).filter(Boolean);
    if (!points.length) return null;
    const xValues = points.map((point) => point.x * width);
    const yValues = points.map((point) => point.y * height);
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);
    const minY = Math.min(...yValues);
    const maxY = Math.max(...yValues);
    const padX = Math.max(18, (maxX - minX) * 0.24);
    const padY = Math.max(14, (maxY - minY) * 0.7);
    const x = Math.max(0, Math.floor(minX - padX));
    const y = Math.max(0, Math.floor(minY - padY));
    const right = Math.min(width, Math.ceil(maxX + padX));
    const bottom = Math.min(height, Math.ceil(maxY + padY));
    return { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y) };
  }

  function drawDetection(canvas, landmarks, roi, width, height) {
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, width, height);

    context.fillStyle = "rgba(115, 231, 244, 0.72)";
    landmarks.forEach((point, index) => {
      if (index % 8 !== 0) return;
      context.beginPath();
      context.arc(point.x * width, point.y * height, Math.max(1.3, width / 520), 0, Math.PI * 2);
      context.fill();
    });

    context.strokeStyle = "#b8f36a";
    context.lineWidth = Math.max(1.5, width / 430);
    context.setLineDash([5, 5]);
    context.strokeRect(roi.x, roi.y, roi.width, roi.height);
    context.setLineDash([]);
    context.fillStyle = "#b8f36a";
    context.font = `${Math.max(10, width / 82)}px monospace`;
    context.fillText("MOUTH ROI", roi.x + 7, Math.max(15, roi.y - 8));

    context.fillStyle = "#ff8b6a";
    lipLandmarkIndices.forEach((index) => {
      const point = landmarks[index];
      if (!point) return;
      context.beginPath();
      context.arc(point.x * width, point.y * height, Math.max(2, width / 290), 0, Math.PI * 2);
      context.fill();
    });
  }

  function preprocessRoi(video, roi) {
    const buffer = elements.roiBuffer;
    if (!buffer) return "local ROI";
    const context = buffer.getContext("2d", { willReadFrequently: true });
    context.clearRect(0, 0, 96, 96);
    context.drawImage(video, roi.x, roi.y, roi.width, roi.height, 0, 0, 96, 96);

    if (!state.cv) return "canvas ROI";
    try {
      const source = state.cv.imread(buffer);
      const resized = new state.cv.Mat();
      const normalized = new state.cv.Mat();
      state.cv.resize(source, resized, new state.cv.Size(96, 96), 0, 0, state.cv.INTER_AREA);
      resized.convertTo(normalized, state.cv.CV_32F, 1 / 255);
      source.delete();
      resized.delete();
      normalized.delete();
      return "OpenCV ROI";
    } catch (error) {
      return "canvas ROI";
    }
  }

  function updatePredictionState() {
    setText(elements.predictionValue, "Prediction unavailable");
    setText(elements.confidenceValue, "--");
    setText(elements.readoutPulse, " model pending");
    if (elements.readoutPulse) elements.readoutPulse.prepend(document.createElement("i"));
    setText(elements.predictionNote, "A visual sequence was captured, but the report does not provide trained CNN-LSTM weights, a class vocabulary or measured prediction confidence.");
  }

  function handleFaceMeshResults(results) {
    const video = activeVideo();
    const canvas = activeOverlay();
    if (!video || !canvas || state.activeSource === null) return;
    resizeCanvasToVideo(canvas, video);
    const width = canvas.width;
    const height = canvas.height;
    const landmarks = results.multiFaceLandmarks?.[0];
    if (!landmarks) {
      clearCanvas(canvas);
      setText(elements.landmarkValue, "not found");
      setText(elements.roiValue, "waiting");
      setText(elements.inferenceDetection, "Face not detected");
      setText(elements.inferenceRoi, "ROI not captured");
      qsa(".inference-step", elements.predictionPanel).forEach((step, index) => step.classList.toggle("is-ready", index === 0));
      return;
    }

    const roi = getMouthRoi(landmarks, width, height);
    if (!roi) return;
    drawDetection(canvas, landmarks, roi, width, height);
    const preprocessingStatus = preprocessRoi(video, roi);
    state.frameCount = Math.min(24, state.frameCount + 1);
    setText(elements.landmarkValue, "tracked");
    setText(elements.roiValue, "captured");
    setText(elements.sequenceValue, `${state.frameCount} captured`);
    setText(elements.inferenceDetection, "MediaPipe landmarks tracked");
    setText(elements.inferenceRoi, `${preprocessingStatus} / normalized`);
    setText(elements.inferenceModel, "CNN-LSTM weights pending");
    qsa(".inference-step", elements.predictionPanel).forEach((step) => step.classList.add("is-ready"));
    setText(state.activeSource === "live" ? elements.liveCaption : elements.uploadCaption, "LANDMARKS TRACKED");
    updatePredictionState();
  }

  function processingError(message) {
    state.processing = false;
    if (state.frameRequest) cancelAnimationFrame(state.frameRequest);
    setStageStatus("error", "vision error");
    setText(elements.cameraHint, message);
    setText(elements.inferenceDetection, "Vision module unavailable");
  }

  async function processFrameLoop(video, source, token) {
    if (!state.processing || token !== state.runToken || state.activeSource !== source) return;
    const now = performance.now();
    if (!state.frameBusy && video.readyState >= 2 && now - state.lastSentAt >= 90) {
      state.frameBusy = true;
      state.lastSentAt = now;
      try {
        await state.faceMesh.send({ image: video });
      } catch (error) {
        processingError("The vision runtime could not process this frame.");
      } finally {
        state.frameBusy = false;
      }
    }
    if (state.processing && token === state.runToken) {
      state.frameRequest = requestAnimationFrame(() => processFrameLoop(video, source, token));
    }
  }

  function startFrameLoop(video, source) {
    state.activeSource = source;
    state.processing = true;
    state.lastSentAt = 0;
    const token = ++state.runToken;
    state.frameRequest = requestAnimationFrame(() => processFrameLoop(video, source, token));
  }

  function stopFrameLoop() {
    state.processing = false;
    state.runToken += 1;
    if (state.frameRequest) cancelAnimationFrame(state.frameRequest);
    state.frameRequest = 0;
  }

  function cameraErrorMessage(error) {
    if (error?.name === "NotAllowedError" || error?.name === "SecurityError") return "Camera permission was denied. Allow access and try again.";
    if (error?.name === "NotFoundError") return "No camera was found on this device.";
    if (error?.name === "NotReadableError") return "The camera is already in use by another application.";
    if (!window.isSecureContext) return "Camera access needs HTTPS or localhost. Serve this project locally instead of opening a restricted file.";
    return "Camera access could not be started. Check browser permissions and try again.";
  }

  async function startCamera() {
    stopCamera(true);
    resetStats();
    state.activeSource = "live";
    setStageStatus("working", "requesting camera");
    setText(elements.cameraHint, "Requesting local camera permission...");
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera API unavailable");
      state.cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      elements.liveVideo.srcObject = state.cameraStream;
      await elements.liveVideo.play();
      elements.liveStage.classList.add("is-live");
      elements.liveEmpty.classList.add("hidden");
      elements.startCamera.hidden = true;
      elements.stopCamera.hidden = false;
      setText(elements.liveCaption, "LOADING VISION MODULE");
      setStageStatus("working", "loading vision");
      await prepareVision();
      setStageStatus("working", "tracking");
      setText(elements.cameraHint, state.cv ? "MediaPipe + OpenCV ready. No audio is used." : "MediaPipe ready. OpenCV fallback is active.");
      startFrameLoop(elements.liveVideo, "live");
    } catch (error) {
      stopCamera(true);
      setStageStatus("error", "camera unavailable");
      setText(elements.cameraHint, /vision|mediapipe|face_mesh/i.test(error?.message || "") ? "MediaPipe could not be loaded. Check the network connection and try again." : cameraErrorMessage(error));
    }
  }

  function stopCamera(silent = false) {
    stopFrameLoop();
    if (state.cameraStream) state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
    if (elements.liveVideo) {
      elements.liveVideo.pause();
      elements.liveVideo.srcObject = null;
    }
    if (elements.liveStage) elements.liveStage.classList.remove("is-live");
    if (elements.liveEmpty) elements.liveEmpty.classList.remove("hidden");
    if (elements.startCamera) elements.startCamera.hidden = false;
    if (elements.stopCamera) elements.stopCamera.hidden = true;
    if (state.activeSource === "live") state.activeSource = null;
    if (!silent) {
      setStageStatus("idle", "ready");
      setText(elements.cameraHint, "Camera permission is required.");
      setText(elements.liveCaption, "WAITING FOR INPUT");
    }
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "--";
    const units = ["B", "KB", "MB", "GB"];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${(bytes / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds)) return "--";
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.floor(seconds % 60).toString().padStart(2, "0");
    return `${minutes}:${remainder}`;
  }

  function showUploadError(message) {
    setText(elements.uploadError, message);
    setStageStatus("error", "upload error");
  }

  function loadVideoFile(file) {
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      showUploadError("Please choose a video file.");
      return;
    }
    stopCamera(true);
    stopFrameLoop();
    if (state.uploadUrl) URL.revokeObjectURL(state.uploadUrl);
    state.uploadUrl = URL.createObjectURL(file);
    state.uploadFile = file;
    elements.uploadVideo.src = state.uploadUrl;
    elements.uploadVideo.load();
    elements.uploadZone.hidden = true;
    elements.uploadPreview.hidden = false;
    setText(elements.fileName, file.name);
    setText(elements.fileDetails, `${file.type.replace("video/", "").toUpperCase() || "VIDEO"} / ${formatBytes(file.size)}`);
    setText(elements.uploadError, "");
    resetStats();
    setStageStatus("idle", "video ready");
  }

  async function analyzeUpload() {
    if (!state.uploadFile) {
      showUploadError("Choose a video before analyzing the sequence.");
      return;
    }
    stopCamera(true);
    stopFrameLoop();
    state.activeSource = "upload";
    resetStats();
    setText(elements.uploadCaption, "LOADING VISION MODULE");
    setText(elements.uploadError, "");
    setStageStatus("working", "loading vision");
    try {
      await prepareVision();
      elements.uploadVideo.currentTime = 0;
      await elements.uploadVideo.play();
      setStageStatus("working", "analyzing");
      setText(elements.cameraHint, state.cv ? "Analyzing locally with MediaPipe + OpenCV." : "Analyzing locally with MediaPipe and canvas fallback.");
      startFrameLoop(elements.uploadVideo, "upload");
    } catch (error) {
      setStageStatus("error", "analysis error");
      setText(elements.uploadError, "This video could not be analyzed. Check the file and network connection.");
    }
  }

  function removeUpload() {
    stopFrameLoop();
    if (elements.uploadVideo) {
      elements.uploadVideo.pause();
      elements.uploadVideo.removeAttribute("src");
      elements.uploadVideo.load();
    }
    if (state.uploadUrl) URL.revokeObjectURL(state.uploadUrl);
    state.uploadUrl = null;
    state.uploadFile = null;
    if (state.activeSource === "upload") state.activeSource = null;
    elements.uploadInput.value = "";
    elements.uploadZone.hidden = false;
    elements.uploadPreview.hidden = true;
    setText(elements.uploadError, "");
    resetStats();
    setStageStatus("idle", "ready");
  }

  function setInputMode(mode) {
    const live = mode === "live";
    elements.liveTab.classList.toggle("is-active", live);
    elements.uploadTab.classList.toggle("is-active", !live);
    elements.liveTab.setAttribute("aria-selected", String(live));
    elements.uploadTab.setAttribute("aria-selected", String(!live));
    elements.liveSource.hidden = !live;
    elements.uploadSource.hidden = live;
    if (live) {
      stopFrameLoop();
      if (state.activeSource === "upload") {
        state.activeSource = null;
        elements.uploadVideo.pause();
      }
      setStageStatus("idle", "ready");
    } else {
      stopCamera(true);
      setStageStatus(state.uploadFile ? "idle" : "idle", state.uploadFile ? "video ready" : "ready");
    }
  }

  function setupUploadHandlers() {
    elements.uploadInput.addEventListener("change", (event) => loadVideoFile(event.target.files?.[0]));
    ["dragenter", "dragover"].forEach((eventName) => elements.uploadZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.uploadZone.classList.add("is-dragging");
    }));
    ["dragleave", "drop"].forEach((eventName) => elements.uploadZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.uploadZone.classList.remove("is-dragging");
    }));
    elements.uploadZone.addEventListener("drop", (event) => loadVideoFile(event.dataTransfer.files?.[0]));
    elements.uploadVideo.addEventListener("loadedmetadata", () => setText(elements.uploadDuration, formatDuration(elements.uploadVideo.duration)));
    elements.uploadVideo.addEventListener("ended", () => {
      if (state.activeSource !== "upload") return;
      stopFrameLoop();
      setStageStatus("pending", "sequence ready");
      setText(elements.uploadCaption, "SEQUENCE CAPTURED");
      setText(elements.inferenceModel, "CNN-LSTM weights pending");
    });
  }

  function setupDemo() {
    elements.liveTab.addEventListener("click", () => setInputMode("live"));
    elements.uploadTab.addEventListener("click", () => setInputMode("upload"));
    elements.startCamera.addEventListener("click", startCamera);
    elements.stopCamera.addEventListener("click", () => stopCamera());
    elements.analyzeUpload.addEventListener("click", analyzeUpload);
    elements.removeUpload.addEventListener("click", removeUpload);
    elements.clearSession.addEventListener("click", () => {
      stopCamera(true);
      removeUpload();
      resetStats();
      setText(elements.cameraHint, "Camera permission is required.");
    });
    setupUploadHandlers();
  }

  function setupScrollControls() {
    const floatingTop = qs("#floating-top");
    const updateTopButton = () => floatingTop?.classList.toggle("visible", window.scrollY > 650);
    window.addEventListener("scroll", updateTopButton, { passive: true });
    floatingTop?.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
    qsa("a.back-top").forEach((link) => link.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" })));
    updateTopButton();
  }

  function cacheElements() {
    elements.themeToggle = qs("#theme-toggle");
    elements.menuToggle = qs("#menu-toggle");
    elements.mobileNav = qs("#mobile-nav");
    elements.stageStatus = qs("#stage-status");
    elements.stageStatusText = qs("#stage-status-text");
    elements.liveTab = qs("#live-tab");
    elements.uploadTab = qs("#upload-tab");
    elements.liveSource = qs("#live-source");
    elements.uploadSource = qs("#upload-source");
    elements.liveVideo = qs("#live-video");
    elements.liveOverlay = qs("#live-overlay");
    elements.liveStage = qs("#live-stage");
    elements.liveEmpty = qs("#live-empty");
    elements.liveCaption = qs("#live-caption");
    elements.startCamera = qs("#start-camera");
    elements.stopCamera = qs("#stop-camera");
    elements.cameraHint = qs("#camera-hint");
    elements.uploadInput = qs("#video-upload");
    elements.uploadZone = qs("#upload-zone");
    elements.uploadPreview = qs("#upload-preview");
    elements.uploadVideo = qs("#upload-video");
    elements.uploadOverlay = qs("#upload-overlay");
    elements.uploadCaption = qs("#upload-caption");
    elements.uploadDuration = qs("#upload-duration");
    elements.uploadError = qs("#upload-error");
    elements.fileName = qs("#file-name");
    elements.fileDetails = qs("#file-details");
    elements.analyzeUpload = qs("#analyze-upload");
    elements.removeUpload = qs("#remove-upload");
    elements.landmarkValue = qs("#landmark-value");
    elements.roiValue = qs("#roi-value");
    elements.sequenceValue = qs("#sequence-value");
    elements.roiBuffer = qs("#roi-buffer");
    elements.predictionPanel = qs(".prediction-panel");
    elements.predictionValue = qs("#prediction-value");
    elements.confidenceValue = qs("#confidence-value");
    elements.predictionNote = qs("#prediction-note");
    elements.readoutPulse = qs(".readout-pulse");
    elements.inferenceDetection = qs("#inference-detection");
    elements.inferenceRoi = qs("#inference-roi");
    elements.inferenceModel = qs("#inference-model");
    elements.clearSession = qs("#clear-session");
  }

  function setupTheme() {
    applyTheme(readTheme() || document.documentElement.dataset.theme || "dark", false);
    elements.themeToggle?.addEventListener("click", () => {
      const nextTheme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
      applyTheme(nextTheme);
    });
  }

  function init() {
    cacheElements();
    setupTheme();
    setupNavigation();
    setupRevealAnimations();
    setupScrollControls();
    resetStats();
    if (elements.liveTab && elements.uploadInput) setupDemo();
    window.addEventListener("beforeunload", () => stopCamera(true), { once: true });
  }

  document.addEventListener("DOMContentLoaded", init, { once: true });
})();

/* ==========================================================================
   PlantCare AI - frontend logic
   One file, split into small sections. Each page runs only its own part,
   chosen by <body data-page="...">.
   ========================================================================== */

const MAX_MB = 10;
const ALLOWED_TYPES = ["image/jpeg", "image/png"];
const ALLOWED_EXT = ["jpg", "jpeg", "png"];

const $ = (sel, root = document) => root.querySelector(sel);

/* ---------- Shared helpers ---------- */

/** Show a short message at the bottom of the screen. */
function toast(message) {
  const el = $("[data-toast]");
  if (!el) return;
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove("show"), 3000);
}

/**
 * Call the Flask API and always return { ok, status, data }.
 * A network failure (server stopped) becomes a friendly error.
 */
async function api(url, options = {}) {
  try {
    const res = await fetch(url, options);
    let data = {};
    try { data = await res.json(); } catch { data = {}; }
    if (!res.ok && !data.error) data.error = `Server error (${res.status}).`;
    return { ok: res.ok && data.success !== false, status: res.status, data };
  } catch {
    return {
      ok: false, status: 0,
      data: { error: "Cannot reach the PlantCare AI server. Make sure it is running (python app.py) and try again." },
    };
  }
}

function formatDate(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

function formatSize(bytes) {
  return bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

function statusTagClass(status) {
  return { Healthy: "tag-ok", Diseased: "tag-bad" }[status] || "tag-warn";
}

/** Create an element with text (never innerHTML for server data). */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ---------- Mobile navigation ---------- */
(function initNav() {
  const btn = $("[data-nav-toggle]");
  const links = $("[data-nav-links]");
  if (!btn || !links) return;
  btn.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    btn.setAttribute("aria-expanded", open);
  });
})();

/* ==========================================================================
   SCAN PAGE
   ========================================================================== */
function initScanPage() {
  const galleryInput = $("[data-gallery-input]");
  const captureInput = $("[data-capture-input]");
  const preview = $("[data-preview]");
  const previewImg = $("[data-preview-img]");
  const fileName = $("[data-file-name]");
  const fileSize = $("[data-file-size]");
  const detectBtn = $("[data-detect]");
  const loading = $("[data-loading]");
  const errorBox = $("[data-error]");
  const errorText = $("[data-error-text]");
  const picker = $("[data-source-picker]");
  const dropzone = $("[data-dropzone]");

  let selectedFile = null;
  let previewUrl = null;
  let lastSource = "gallery";

  function showError(msg) {
    errorText.textContent = msg;
    errorBox.hidden = false;
    errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  function clearError() { errorBox.hidden = true; }

  /** Check type and size in the browser before uploading. */
  function validate(file) {
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    const typeOk = ALLOWED_TYPES.includes(file.type) || (!file.type && ALLOWED_EXT.includes(ext));
    if (!typeOk) return "Unsupported file format. Please choose a JPG, JPEG or PNG image.";
    if (file.size > MAX_MB * 1024 * 1024) return `Image is too large (${formatSize(file.size)}). Maximum size is ${MAX_MB} MB.`;
    if (file.size === 0) return "The selected file is empty.";
    return null;
  }

  function setFile(file) {
    clearError();
    const problem = validate(file);
    if (problem) { showError(problem); return; }

    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    previewImg.src = previewUrl;
    previewImg.onerror = () => { removeFile(); showError("This image could not be opened. It may be corrupted."); };
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);
    preview.hidden = false;
    picker.hidden = true;
    dropzone.hidden = true;
    detectBtn.disabled = false;
  }

  function removeFile() {
    selectedFile = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    previewImg.removeAttribute("src");
    preview.hidden = true;
    picker.hidden = false;
    dropzone.hidden = false;
    detectBtn.disabled = true;
    galleryInput.value = "";
    captureInput.value = "";
  }

  galleryInput.addEventListener("change", () => {
    lastSource = "gallery";
    if (galleryInput.files[0]) setFile(galleryInput.files[0]);
  });
  captureInput.addEventListener("change", () => {
    lastSource = "camera";
    const f = captureInput.files[0];
    if (f) setFile(f.type ? f : new File([f], "camera-photo.jpg", { type: "image/jpeg" }));
  });

  $("[data-remove]").addEventListener("click", removeFile);
  $("[data-change]").addEventListener("click", () => {
    if (lastSource === "camera") openCamera(); else galleryInput.click();
  });

  // Drag and drop (desktop)
  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) { lastSource = "gallery"; setFile(f); }
  });

  /* ----- Camera ----- */
  const modal = $("[data-camera-modal]");
  const video = $("[data-camera-video]");
  let stream = null;
  let facingMode = "environment";

  // Live camera needs HTTPS or localhost. Otherwise we fall back to the
  // phone's own camera app via <input capture>, which works over plain HTTP.
  const liveCameraPossible = window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia;

  async function startStream() {
    stopStream();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facingMode }, width: { ideal: 1920 }, height: { ideal: 1440 } },
      audio: false,
    });
    video.srcObject = stream;
  }

  function stopStream() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  async function openCamera() {
    clearError();
    lastSource = "camera";
    if (!liveCameraPossible) { captureInput.click(); return; }
    try {
      modal.hidden = false;
      await startStream();
    } catch (err) {
      modal.hidden = true;
      stopStream();
      // Permission denied or no camera: use the device camera app / file picker instead
      toast(err && err.name === "NotAllowedError" ? "Camera permission denied. Opening file picker instead." : "Live camera unavailable. Opening your camera app instead.");
      captureInput.click();
    }
  }

  function closeCamera() { stopStream(); modal.hidden = true; }

  $("[data-open-camera]").addEventListener("click", openCamera);
  $("[data-camera-close]").addEventListener("click", closeCamera);
  $("[data-camera-flip]").addEventListener("click", async () => {
    facingMode = facingMode === "environment" ? "user" : "environment";
    try { await startStream(); } catch { toast("Could not switch camera."); }
  });
  $("[data-camera-shoot]").addEventListener("click", () => {
    if (!video.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      closeCamera();
      setFile(new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" }));
    }, "image/jpeg", 0.92);
  });
  modal.addEventListener("click", (e) => { if (e.target === modal) closeCamera(); });

  /* ----- Detect ----- */
  detectBtn.addEventListener("click", async () => {
    if (!selectedFile) { showError("No image selected. Please capture or upload a leaf photo first."); return; }
    clearError();
    detectBtn.disabled = true;
    detectBtn.lastChild.textContent = " Analysing…";
    loading.hidden = false;

    const form = new FormData();
    form.append("image", selectedFile, selectedFile.name);
    const { ok, data } = await api("/api/predict", { method: "POST", body: form });

    if (ok && data.scan_id) {
      window.location.href = `/result/${data.scan_id}`;
      return;
    }
    loading.hidden = true;
    detectBtn.disabled = false;
    detectBtn.lastChild.textContent = " Detect Disease";
    showError(data.error || "Prediction failed. Please try again.");
  });

  // Warn early if the backend is not reachable
  api("/api/health").then(({ status }) => {
    if (status === 0) showError("Cannot reach the PlantCare AI server. Make sure it is running (python app.py).");
  });
}

/* ==========================================================================
   RESULT PAGE
   ========================================================================== */
async function initResultPage() {
  const root = $("[data-scan-id]");
  const id = root.dataset.scanId;
  const loading = $("[data-state-loading]");
  const errorState = $("[data-state-error]");
  const result = $("[data-result]");

  const { ok, data } = await api(`/api/history/${id}`);
  loading.hidden = true;
  if (!ok) {
    $("[data-error-text]", errorState).textContent = data.error || "This result could not be loaded.";
    errorState.hidden = false;
    return;
  }

  const r = data;
  const uncertain = r.status === "Uncertain";
  const healthy = r.status === "Healthy";

  const plantDisplay = r.local_name ? `${r.local_name} (${r.plant})` : r.plant;
  document.title = `${uncertain ? "Uncertain result" : `${plantDisplay} · ${healthy ? "Healthy" : r.disease}`} · PlantCare AI`;
  $("[data-scan-date]").textContent = `Scanned ${formatDate(r.scan_date)}`;
  $("[data-title]").textContent = uncertain ? "Result uncertain" : healthy ? `Healthy ${plantDisplay} leaf` : r.disease;

  const pill = $("[data-status-pill]");
  pill.textContent = r.status;
  pill.classList.add(`status-${r.status.toLowerCase()}`);

  $("[data-image]").src = r.image_url;
  $("[data-plant]").textContent = r.plant;
  $("[data-local]").textContent = r.local_name || "";
  $("[data-local-row]").hidden = !r.local_name;
  $("[data-sci]").textContent = r.scientific_name || "";
  $("[data-sci-row]").hidden = !r.scientific_name;
  $("[data-disease]").textContent = r.disease;
  $("[data-status]").textContent = r.status;
  $("[data-pathogen]").textContent = r.pathogen || "";
  $("[data-pathogen-row]").hidden = !r.pathogen;

  // Confidence ring
  const conf = Number(r.confidence) || 0;
  $("[data-confidence]").textContent = `${conf.toFixed(1)}%`;
  const ring = $("[data-ring]");
  // The model can be very sure that a leaf is NOT one of its plants ("Other" class)
  const unsupported = r.predicted_label === "Other___unsupported_plant" && conf >= r.threshold;
  ring.classList.add(uncertain ? "low" : conf >= 85 ? "high" : conf >= r.threshold ? "mid" : "low");
  const circumference = 2 * Math.PI * 52;
  requestAnimationFrame(() => {
    $("[data-ring-fg]").style.strokeDashoffset = circumference * (1 - conf / 100);
  });
  $("[data-confidence-note]").textContent = unsupported
    ? "Confidence that this leaf is not one of the supported plants."
    : uncertain
    ? `Below the ${r.threshold}% level needed for a reliable answer.`
    : conf >= 85 ? "High confidence prediction." : "Moderate confidence. Consider re-scanning with a clearer photo.";

  // Uncertain message
  if (uncertain) {
    $("[data-uncertain]").hidden = false;
    $("[data-uncertain-text]").textContent = r.description;
    $("[data-section=description]").hidden = true;
    $("[data-prev-title]").textContent = "How to get a better result";
  } else {
    $("[data-desc-title]").textContent = healthy ? "Leaf health" : "About this disease";
    $("[data-description]").textContent = r.description;
    if (healthy) $("[data-prev-title]").textContent = "Care & preventive tips";
  }

  // Lists: sections with no items are hidden (healthy leaves get no treatment)
  ["symptoms", "signs", "causes", "treatment", "prevention"].forEach((key) => {
    const items = r[key] || [];
    const list = $(`[data-list=${key}]`);
    items.forEach((text) => list.appendChild(el("li", "", text)));
    $(`[data-section=${key}]`).hidden = items.length === 0;
  });
  // Hide the two-column block when both of its cards are empty
  $(".info-grid").hidden = !(r.symptoms || []).length && !(r.signs || []).length;

  // Top predictions (only for confident results; never show guesses for uncertain ones)
  const top = $("[data-top]");
  if (!uncertain && r.top_predictions && r.top_predictions.length) {
    r.top_predictions.forEach((p) => {
      const row = el("div", "bar-row");
      const label = el("div", "bar-label");
      label.append(el("span", "", p.label.replace(/_+/g, " ").replace(/\s+/g, " ").trim()), el("span", "", `${p.confidence.toFixed(1)}%`));
      const track = el("div", "bar-track");
      const fill = el("div", "bar-fill");
      track.appendChild(fill);
      row.append(label, track);
      top.appendChild(row);
      requestAnimationFrame(() => { fill.style.width = `${p.confidence}%`; });
    });
  } else {
    $("[data-section=top]").hidden = true;
  }

  result.hidden = false;
}

/* ==========================================================================
   HISTORY PAGE
   ========================================================================== */
async function initHistoryPage() {
  const list = $("[data-history-list]");
  const loading = $("[data-state-loading]");
  const empty = $("[data-state-empty]");
  const errorBox = $("[data-error]");
  const confirmBox = $("[data-confirm]");
  let pendingDelete = null;

  const { ok, data } = await api("/api/history");
  loading.hidden = true;
  if (!ok) {
    $("[data-error-text]").textContent = data.error || "Could not load scan history.";
    errorBox.hidden = false;
    return;
  }

  const scans = data.scans || [];
  if (scans.length === 0) { empty.hidden = false; return; }

  scans.forEach((s) => list.appendChild(renderCard(s)));

  function renderCard(s) {
    const card = el("article", "card history-card");
    const img = el("img", "history-thumb");
    img.src = s.image_url;
    img.alt = `${s.plant} leaf`;
    img.loading = "lazy";

    const body = el("div", "history-body");
    const sub = el("div", "history-sub");
    sub.append(el("span", "history-date", formatDate(s.scan_date)), el("span", `tag ${statusTagClass(s.status)}`, s.status));
    const plantName = s.local_name ? `${s.local_name} (${s.plant})` : s.plant;
    const title = s.status === "Uncertain" ? "Uncertain result" : plantName;
    const disease = s.status === "Uncertain" ? `Confidence ${s.confidence.toFixed(1)}%` : `${s.disease} · ${s.confidence.toFixed(1)}%`;

    const actions = el("div", "history-actions");
    const view = el("a", "btn btn-primary btn-sm", "View Results");
    view.href = `/result/${s.id}`;
    const del = el("button", "btn btn-ghost btn-sm btn-danger-text", "Delete");
    del.type = "button";
    del.addEventListener("click", () => { pendingDelete = { id: s.id, card }; confirmBox.hidden = false; });
    actions.append(view, del);

    body.append(sub, el("h3", "", title), el("p", "muted small", disease), actions);
    card.append(img, body);
    return card;
  }

  $("[data-confirm-cancel]").addEventListener("click", () => { confirmBox.hidden = true; pendingDelete = null; });
  confirmBox.addEventListener("click", (e) => { if (e.target === confirmBox) $("[data-confirm-cancel]").click(); });

  $("[data-confirm-ok]").addEventListener("click", async () => {
    if (!pendingDelete) return;
    const { id, card } = pendingDelete;
    confirmBox.hidden = true;
    pendingDelete = null;
    const res = await api(`/api/history/${id}`, { method: "DELETE" });
    if (!res.ok) { toast(res.data.error || "Could not delete the scan."); return; }
    card.classList.add("removing");
    setTimeout(() => {
      card.remove();
      if (!list.children.length) empty.hidden = false;
    }, 220);
    toast("Scan deleted.");
  });
}

/* ---------- Run the right page ---------- */
const page = document.body.dataset.page;
if (page === "scan") initScanPage();
if (page === "result") initResultPage();
if (page === "history") initHistoryPage();

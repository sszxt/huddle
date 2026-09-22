// Model manager: on-disk list, HF search, download with progress.
// Plain fetch() against Huddle's own /cluster/* API, same origin — no CORS,
// no build step, no framework.

const modelList = document.getElementById("model-list");
const searchBox = document.getElementById("search-box");
const searchBtn = document.getElementById("search-btn");
const searchResults = document.getElementById("search-results");
const repoFilesEl = document.getElementById("repo-files");
const downloadProgressEl = document.getElementById("download-progress");

let pollHandle = null;

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatBytes(n) {
  if (n === null || n === undefined) return "size unknown";
  const gib = n / 1024 ** 3;
  return gib >= 1 ? `${gib.toFixed(1)} GiB` : `${(n / 1024 ** 2).toFixed(0)} MiB`;
}

async function refreshModels() {
  const response = await fetch("/cluster/models");
  const data = await response.json();
  const available = data.available || [];
  const loaded = data.loaded;
  if (available.length === 0) {
    modelList.innerHTML = '<li class="dim">no models on disk yet</li>';
    return;
  }
  modelList.innerHTML = available
    .map((name) => {
      const isLoaded = name === loaded;
      const label = isLoaded
        ? `<span class="loaded">● ${escapeHtml(name)} (loaded)</span>`
        : `<span>○ ${escapeHtml(name)}</span>`;
      const button = isLoaded
        ? ""
        : `<button data-model="${escapeHtml(name)}" class="load-btn">Load</button>`;
      return `<li class="file-row">${label} ${button}</li>`;
    })
    .join("");
  document.querySelectorAll(".load-btn").forEach((btn) => {
    btn.addEventListener("click", () => loadModel(btn.dataset.model));
  });
}

async function loadModel(name) {
  await fetch("/cluster/model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: name }),
  });
  await refreshModels();
}

async function runSearch() {
  const q = searchBox.value.trim();
  if (!q) return;
  searchResults.innerHTML = '<p class="dim">searching…</p>';
  repoFilesEl.innerHTML = "";
  const response = await fetch(`/cluster/models/search?q=${encodeURIComponent(q)}`);
  if (!response.ok) {
    searchResults.innerHTML = `<p class="err">search failed (${response.status})</p>`;
    return;
  }
  const data = await response.json();
  const results = data.results || [];
  if (results.length === 0) {
    searchResults.innerHTML = '<p class="dim">no GGUF repos found</p>';
    return;
  }
  searchResults.innerHTML = results
    .map(
      (r) =>
        `<div class="file-row"><span>${escapeHtml(r.repo_id)}</span>` +
        `<button data-repo="${escapeHtml(r.repo_id)}" class="repo-btn">` +
        `${r.gguf_files.length} file(s)</button></div>`
    )
    .join("");
  document.querySelectorAll(".repo-btn").forEach((btn) => {
    btn.addEventListener("click", () => showRepoFiles(btn.dataset.repo));
  });
}

async function showRepoFiles(repoId) {
  repoFilesEl.innerHTML = '<p class="dim">loading files…</p>';
  const response = await fetch(`/cluster/models/repo-files?repo_id=${encodeURIComponent(repoId)}`);
  if (!response.ok) {
    repoFilesEl.innerHTML = `<p class="err">could not list files (${response.status})</p>`;
    return;
  }
  const data = await response.json();
  const files = data.files || [];
  repoFilesEl.innerHTML =
    `<p class="dim">${escapeHtml(repoId)}</p>` +
    files
      .map(
        (f) =>
          `<div class="file-row"><span>${escapeHtml(f.filename)} ` +
          `<span class="dim">(${formatBytes(f.size)})</span></span>` +
          `<button data-repo="${escapeHtml(repoId)}" data-file="${escapeHtml(f.filename)}" ` +
          `class="download-btn">Download</button></div>`
      )
      .join("");
  document.querySelectorAll(".download-btn").forEach((btn) => {
    btn.addEventListener("click", () => startDownload(btn.dataset.repo, btn.dataset.file));
  });
}

async function startDownload(repoId, filename) {
  const response = await fetch("/cluster/models/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_id: repoId, filename }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    downloadProgressEl.innerHTML = `<p class="err">${escapeHtml(body.detail || response.statusText)}</p>`;
    return;
  }
  pollDownload();
}

async function pollDownload() {
  if (pollHandle) clearInterval(pollHandle);
  pollHandle = setInterval(async () => {
    const response = await fetch("/cluster/models/download");
    const status = await response.json();
    renderDownloadStatus(status);
    if (!status.active) {
      clearInterval(pollHandle);
      pollHandle = null;
      if (status.done) await refreshModels();
    }
  }, 1000);
}

function renderDownloadStatus(status) {
  if (!status.repo_id) {
    downloadProgressEl.innerHTML = "";
    return;
  }
  const pct =
    status.total_bytes && status.total_bytes > 0
      ? Math.round((status.bytes_done / status.total_bytes) * 100)
      : null;
  const bar =
    pct === null
      ? `<progress></progress>`
      : `<progress value="${pct}" max="100"></progress> ${pct}%`;
  let extra = "";
  if (status.error) extra = `<p class="err">failed: ${escapeHtml(status.error)}</p>`;
  else if (status.done) extra = `<p>done</p>`;
  downloadProgressEl.innerHTML =
    `<p>${escapeHtml(status.repo_id)}/${escapeHtml(status.filename)}</p>${bar}${extra}`;
}

searchBtn.addEventListener("click", runSearch);
searchBox.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});

refreshModels();
// Pick up a download already in progress from a previous page load.
fetch("/cluster/models/download")
  .then((r) => r.json())
  .then((status) => {
    if (status.active) {
      renderDownloadStatus(status);
      pollDownload();
    }
  });

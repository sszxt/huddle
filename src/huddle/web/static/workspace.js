// Workspace → Models: the models on this node's disk, plus Hugging Face search
// and download. Layout follows Open WebUI v0.6.30's routes/(app)/workspace
// +layout.svelte and workspace/Models.svelte; the progress bar is its Ollama
// model-pull bar (admin/Settings/Models/Manage/ManageOllama.svelte).
//
// Everything talks to Huddle's own /cluster/* routes, same origin.

import { refreshModels, switchModel } from "./cluster.js";
import { icon } from "./icons.js";
import { escapeHtml } from "./markdown.js";
import { displayName, isMobile, setShowSidebar, state } from "./store.js";
import { api, errorDetail, LOGO_URL } from "./ui.js";

let pollHandle = null;
let modelQuery = "";
let hfQuery = "";
let searchResults = null; // null: no search yet
let searchError = null;
let searching = false;
let repo = null; // { repo_id, files } once a result is opened
let repoLoading = false;
let repoError = null;
let download = null;
let downloadError = null;

function formatBytes(n) {
  if (n === null || n === undefined) return "size unknown";
  const gib = n / 1024 ** 3;
  return gib >= 1 ? `${gib.toFixed(1)} GB` : `${(n / 1024 ** 2).toFixed(0)} MB`;
}

function card({ title, subtitle, footer, action, attrs = "", dim = false }) {
  return (
    `<div class="ws-card" ${attrs}>` +
    `<div class="ws-card-top"><div class="ws-card-image${dim ? " dim" : ""}"><img src="${LOGO_URL}" alt=""></div>` +
    `<div class="ws-card-text"><div class="ws-card-title">${title}</div>` +
    `<div class="ws-card-sub"><div>${subtitle}</div></div></div></div>` +
    `<div class="ws-card-bottom"><div class="ws-card-footer">${footer}</div>` +
    `<div class="ws-card-actions">${action ?? ""}</div></div>` +
    `</div>`
  );
}

function header(title, count) {
  return (
    `<div class="ws-header"><div class="ws-title">${title}` +
    (count === null ? "" : `<div class="ws-title-divider"></div><span class="ws-count">${count}</span>`) +
    `</div></div>`
  );
}

function searchRow(cls, value, placeholder) {
  return (
    `<div class="ws-search-row"><div class="ws-search">` +
    `<div class="ws-search-icon">${icon("search", "size-3.5")}</div>` +
    `<input class="ws-search-input ${cls}" value="${escapeHtml(value)}" placeholder="${escapeHtml(placeholder)}" autocomplete="off">` +
    (value
      ? `<div class="ws-search-clear"><button type="button" data-action="clear-${cls}" aria-label="Clear">${icon("xMark", "size-3", 2)}</button></div>`
      : "") +
    `</div></div>`
  );
}

function modelsSection() {
  const q = modelQuery.trim().toLowerCase();
  const models = state.models.filter((file) => file.toLowerCase().includes(q));
  const cards = models
    .map((file) => {
      const loaded = file === state.loaded;
      const loading = file === state.switching;
      const footer = loading
        ? '<span class="ws-status">Loading…</span>'
        : loaded
          ? '<span class="ws-status"><span class="ws-dot"></span>Loaded</span>'
          : '<span class="ws-status">On disk</span>';
      return card({
        title: escapeHtml(displayName(file)),
        subtitle: escapeHtml(file),
        footer,
        attrs: `data-action="use-model" data-file="${escapeHtml(file)}" role="button" tabindex="0"`,
      });
    })
    .join("");
  const empty =
    state.models.length === 0
      ? '<div class="ws-empty">No models on disk yet. Download one below.</div>'
      : models.length === 0
        ? '<div class="ws-empty">No results found</div>'
        : "";
  return (
    `<div class="ws-section">${header("Models", models.length)}` +
    searchRow("model-query", modelQuery, "Search Models") +
    `</div>${empty}<div class="ws-grid">${cards}</div>`
  );
}

function downloadBox() {
  if (downloadError) return `<div class="ws-error">${escapeHtml(downloadError)}</div>`;
  const status = download;
  if (!status?.repo_id) return "";
  const pct =
    status.total_bytes && status.total_bytes > 0
      ? Math.round((status.bytes_done / status.total_bytes) * 100)
      : null;
  let line = `${formatBytes(status.bytes_done)} of ${formatBytes(status.total_bytes)}`;
  if (status.error) line = `Failed: ${escapeHtml(status.error)}`;
  else if (status.done) line = "Download complete";
  const bar =
    status.done || status.error
      ? ""
      : `<div class="ws-progress-row"><div class="ws-progress-track">` +
        `<div class="ws-progress${pct === null ? " indeterminate" : ""}" style="width: ${Math.max(15, pct ?? 0)}%">${pct ?? 0}%</div>` +
        `</div></div>`;
  return (
    `<div class="ws-download"><div class="ws-download-name">${escapeHtml(status.repo_id)}/${escapeHtml(status.filename)}</div>` +
    `${bar}<div class="ws-download-line${status.error ? " error" : status.done ? " done" : ""}">${line}</div></div>`
  );
}

function hubSection() {
  let results = "";
  if (searching) results = '<div class="ws-empty">Searching…</div>';
  else if (searchError) results = `<div class="ws-error">${escapeHtml(searchError)}</div>`;
  else if (searchResults && searchResults.length === 0) results = '<div class="ws-empty">No GGUF repos found</div>';
  else if (searchResults) {
    results =
      `<div class="ws-grid">` +
      searchResults
        .map((r) =>
          card({
            title: escapeHtml(r.repo_id),
            subtitle: `${r.gguf_files.length} GGUF file(s)`,
            footer: '<span class="ws-status">Hugging Face</span>',
            attrs: `data-action="open-repo" data-repo="${escapeHtml(r.repo_id)}" role="button" tabindex="0"`,
            dim: true,
          })
        )
        .join("") +
      `</div>`;
  }

  let files = "";
  if (repoLoading) files = '<div class="ws-empty">Loading files…</div>';
  else if (repoError) files = `<div class="ws-error">${escapeHtml(repoError)}</div>`;
  else if (repo) {
    const busy = Boolean(download?.active);
    files =
      `<div class="ws-section">${header(escapeHtml(repo.repo_id), repo.files.length)}</div>` +
      `<div class="ws-grid">` +
      repo.files
        .map((f) =>
          card({
            title: escapeHtml(f.filename),
            subtitle: formatBytes(f.size),
            footer: '<span class="ws-status">GGUF</span>',
            action:
              `<button type="button" class="ws-button" data-action="download" data-repo="${escapeHtml(repo.repo_id)}" ` +
              `data-file="${escapeHtml(f.filename)}"${busy ? " disabled" : ""}>Download</button>`,
            dim: true,
          })
        )
        .join("") +
      `</div>`;
  }

  return (
    `<div class="ws-section ws-hub">${header("Download from Hugging Face", null)}` +
    searchRow("hf-query", hfQuery, "Search Hugging Face, e.g. qwen2.5 32b instruct") +
    `</div><div class="ws-download-slot">${downloadBox()}</div>${results}${files}`
  );
}

function pageHtml() {
  const toggle =
    isMobile() && !state.showSidebar
      ? `<div class="ws-sidebar-toggle"><button type="button" class="nav-sidebar-button" data-action="open-sidebar" aria-label="Open Sidebar">` +
        `<div class="nav-sidebar-icon">${icon("sidebar", "size-5")}</div></button></div>`
      : "";
  return (
    `<div class="workspace"><nav class="ws-nav"><div class="ws-nav-row">${toggle}` +
    `<div><div class="ws-tabs scrollbar-none"><a class="ws-tab" href="#/workspace">Models</a></div></div>` +
    `</div></nav>` +
    `<div class="ws-container" id="workspace-container"><div class="ws-models">${modelsSection()}</div>` +
    `<div class="ws-hub-wrap">${hubSection()}</div></div></div>`
  );
}

export function renderWorkspace() {
  const container = document.getElementById("chat-container");
  const scroller = container.querySelector("#workspace-container");
  const scrollTop = scroller?.scrollTop ?? 0;
  const focused = document.activeElement?.classList.contains("ws-search-input")
    ? [...document.activeElement.classList].find((c) => c.endsWith("-query"))
    : null;
  container.innerHTML = pageHtml();
  container.querySelector("#workspace-container").scrollTop = scrollTop;
  if (focused) {
    const input = container.querySelector(`.${focused}`);
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }
}

function rerenderIfShown() {
  if (state.route.name === "workspace") renderWorkspace();
}

async function runSearch() {
  const q = hfQuery.trim();
  if (!q) return;
  searching = true;
  searchError = null;
  repo = null;
  repoError = null;
  rerenderIfShown();
  try {
    const response = await api(`/cluster/models/search?q=${encodeURIComponent(q)}`);
    if (!response.ok) throw new Error(`Search failed: ${await errorDetail(response)}`);
    searchResults = (await response.json()).results ?? [];
  } catch (error) {
    searchResults = null;
    searchError = error.message.startsWith("Search failed") ? error.message : "Search failed: network error";
  } finally {
    searching = false;
    rerenderIfShown();
  }
}

async function showRepoFiles(repoId) {
  repoLoading = true;
  repoError = null;
  rerenderIfShown();
  try {
    const response = await api(`/cluster/models/repo-files?repo_id=${encodeURIComponent(repoId)}`);
    if (!response.ok) throw new Error(`Could not list files: ${await errorDetail(response)}`);
    const data = await response.json();
    repo = { repo_id: repoId, files: data.files ?? [] };
  } catch (error) {
    repo = null;
    repoError = error.message.startsWith("Could not") ? error.message : "Could not list files: network error";
  } finally {
    repoLoading = false;
    rerenderIfShown();
  }
}

async function startDownload(repoId, filename) {
  downloadError = null;
  const response = await api("/cluster/models/download", {
    method: "POST",
    json: { repo_id: repoId, filename },
  });
  if (!response.ok) {
    downloadError = await errorDetail(response);
    rerenderIfShown();
    return;
  }
  download = await response.json();
  rerenderIfShown();
  pollDownload();
}

function pollDownload() {
  if (pollHandle) clearInterval(pollHandle);
  pollHandle = setInterval(async () => {
    try {
      download = await (await api("/cluster/models/download")).json();
    } catch {
      return;
    }
    const slot = document.querySelector("#workspace-container .ws-download-slot");
    if (slot) slot.innerHTML = downloadBox();
    if (!download.active) {
      clearInterval(pollHandle);
      pollHandle = null;
      if (download.done) await refreshModels();
      rerenderIfShown();
    }
  }, 1000);
}

export function initWorkspace() {
  const container = document.getElementById("chat-container");
  container.addEventListener("click", (event) => {
    if (state.route.name !== "workspace") return;
    const target = event.target.closest("[data-action]");
    if (!target) return;
    const action = target.dataset.action;
    if (action === "use-model") {
      // Open WebUI's model card opens a chat with that model; here that means
      // loading it, which restarts the cluster.
      const file = target.dataset.file;
      location.hash = "#/";
      if (file !== state.loaded) switchModel(file);
    } else if (action === "open-repo") {
      showRepoFiles(target.dataset.repo);
    } else if (action === "download") {
      target.disabled = true;
      target.textContent = "Starting…";
      startDownload(target.dataset.repo, target.dataset.file);
    } else if (action === "clear-model-query") {
      modelQuery = "";
      renderWorkspace();
    } else if (action === "clear-hf-query") {
      hfQuery = "";
      renderWorkspace();
    } else if (action === "open-sidebar") {
      setShowSidebar(true);
    }
  });
  container.addEventListener("input", (event) => {
    if (event.target.classList.contains("model-query")) {
      modelQuery = event.target.value;
      renderWorkspace();
    } else if (event.target.classList.contains("hf-query")) {
      hfQuery = event.target.value;
    }
  });
  container.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.classList.contains("hf-query")) runSearch();
    if (event.key === "Enter" && event.target.classList.contains("ws-card")) event.target.click();
  });

  // Pick up a download already in progress from a previous page load.
  api("/cluster/models/download")
    .then((r) => r.json())
    .then((status) => {
      if (status.active) {
        download = status;
        pollDownload();
      }
    })
    .catch(() => {});
}

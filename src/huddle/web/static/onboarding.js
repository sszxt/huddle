// First run: what to do with a cluster that has no model running.
//
// Shown on the home screen in place of the prompt suggestions until a model is
// running somewhere in the cluster: how much the PCs Huddle found can hold,
// the models already on them (one click to start), and a short list to
// download, each labelled with whether it fits. A model downloaded from here
// is started as soon as it arrives, on this PC, borrowing the others' GPUs.

import { refreshModels, switchModel } from "./cluster.js";
import { icon } from "./icons.js";
import { escapeHtml } from "./markdown.js";
import { displayName, state } from "./store.js";
import { api, errorDetail, formatBytes } from "./ui.js";

// The catalog costs the server a hardware probe of every PC, so ask for it
// at most this often.
const CATALOG_SECONDS = 30;

const FIT = {
  gpu: ["Runs on GPUs", "ob-fit-gpu"],
  partial: ["Partly on CPU, slower", "ob-fit-partial"],
  too_large: ["Too large for these PCs", "ob-fit-no"],
};

let catalog = null;
let catalogError = null;
let fetchedAt = 0;
let download = null;
let downloadError = null;
// The file to start once its download finishes, if it was started from here.
let startWhenDone = null;
let poll = null;

/** Nothing is running or loading anywhere, so there is nothing to chat with yet. */
export function wantsOnboarding() {
  return (
    !state.loaded &&
    !state.loading &&
    !state.switching &&
    state.health?.status !== "starting" &&
    state.health?.status !== "unreachable"
  );
}

function gb(mib) {
  const value = mib / 1024;
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} GB`;
}

function section(title, rows) {
  return (
    `<div class="ob-section"><div class="ob-head">${escapeHtml(title)}</div>` +
    `<div class="ob-list">${rows}</div></div>`
  );
}

function row({ title, sub, meta = "", end }) {
  return (
    `<div class="ob-row"><div class="ob-text">` +
    `<div class="ob-title">${title}</div>` +
    `<div class="ob-sub">${sub}</div>` +
    (meta ? `<div class="ob-meta">${meta}</div>` : "") +
    `</div><div class="ob-end">${end}</div></div>`
  );
}

function capacityLine() {
  if (!catalog) return "";
  const { nodes, gpu_mib: gpu } = catalog.capacity;
  const pcs = `${nodes} PC${nodes === 1 ? "" : "s"}`;
  const alone = nodes === 1 ? ". Run Huddle on your other PCs to pool their GPUs" : "";
  return (
    `<div class="ob-capacity">${icon("serverStack", "size-3.5")}` +
    `<span>${pcs} · ${gb(gpu)} of GPU memory free${alone}</span>` +
    `<a href="#/cluster" class="ob-link">See cluster</a></div>`
  );
}

function progressHtml() {
  const total = download?.total_bytes;
  const pct = total ? Math.round((download.bytes_done / total) * 100) : null;
  return (
    `<div class="ob-progress" aria-label="Downloading">` +
    `<div class="ws-progress-track"><div class="ws-progress${pct === null ? " indeterminate" : ""}" ` +
    `style="width: ${Math.max(15, pct ?? 0)}%">${pct ?? 0}%</div></div></div>`
  );
}

function onDiskRows() {
  if (state.models.length === 0) return "";
  const rows = state.models
    .map((m) =>
      row({
        title: escapeHtml(displayName(m.file)),
        sub: escapeHtml(
          [m.node ? `on ${m.node}` : null, m.size ? formatBytes(m.size) : null]
            .filter(Boolean)
            .join(" · ")
        ),
        end:
          `<button type="button" class="ob-button" data-action="ob-start" data-file="${escapeHtml(m.file)}" ` +
          `data-node-id="${m.local ? "" : escapeHtml(m.node_id ?? "")}" data-node="${m.local ? "" : escapeHtml(m.node ?? "")}">Start</button>`,
      })
    )
    .join("");
  return section("On your PCs", rows);
}

function catalogRows() {
  if (catalogError) return `<div class="ob-error">${escapeHtml(catalogError)}</div>`;
  if (!catalog) return `<div class="ob-empty">Checking what these PCs can run…</div>`;
  const present = new Set(state.models.map((m) => m.file));
  const busy = Boolean(download?.active);
  const rows = catalog.models
    .filter((m) => !present.has(m.file))
    .map((m) => {
      const [label, cls] = FIT[m.fit] ?? FIT.too_large;
      const tested = m.verified
        ? ` <span class="ob-tested" data-tooltip="Run on a real two-PC Huddle cluster" data-placement="top">tested</span>`
        : "";
      const end =
        busy && download.filename === m.file
          ? progressHtml()
          : `<button type="button" class="ob-button${m.fit === "too_large" ? " muted" : ""}" data-action="ob-download" ` +
            `data-repo="${escapeHtml(m.repo_id)}" data-file="${escapeHtml(m.file)}"${busy ? " disabled" : ""}>Download</button>`;
      return row({
        title: escapeHtml(m.name) + tested,
        sub: escapeHtml(m.summary),
        meta: `<span class="ob-fit ${cls}"><span class="ob-dot"></span>${label}</span><span>${formatBytes(m.size)}</span>`,
        end,
      });
    })
    .join("");
  if (!rows) return "";
  return section(state.models.length ? "Or download another" : "Download a model to start", rows);
}

function innerHtml() {
  const error = downloadError ? `<div class="ob-error">${escapeHtml(downloadError)}</div>` : "";
  return (
    capacityLine() +
    error +
    onDiskRows() +
    catalogRows() +
    `<div class="ob-more">Looking for something else? <a href="#/workspace" class="ob-link">Search Hugging Face</a></div>`
  );
}

export function onboardingHtml() {
  return `<div class="ob" id="onboarding">${innerHtml()}</div>`;
}

/** Redraw the panel in place, if it is on screen. */
export function updateOnboarding() {
  const panel = document.getElementById("onboarding");
  if (panel) panel.innerHTML = innerHtml();
}

export async function refreshCatalog(force = false) {
  if (!force && Date.now() - fetchedAt < CATALOG_SECONDS * 1000) return;
  fetchedAt = Date.now();
  try {
    const response = await api("/cluster/catalog");
    if (!response.ok) throw new Error(await errorDetail(response));
    catalog = await response.json();
    catalogError = null;
  } catch (error) {
    catalogError = `Could not check what fits: ${error.message}`;
  }
  updateOnboarding();
}

async function startDownload(repoId, filename) {
  downloadError = null;
  try {
    const response = await api("/cluster/models/download", {
      method: "POST",
      json: { repo_id: repoId, filename },
    });
    if (!response.ok) throw new Error(await errorDetail(response));
    download = await response.json();
  } catch (error) {
    downloadError = `Could not download ${filename}: ${error.message}`;
    updateOnboarding();
    return;
  }
  // Downloads land flat in the models folder, under the file's own name.
  startWhenDone = filename.split("/").pop();
  updateOnboarding();
  watchDownload();
}

function watchDownload() {
  if (poll) return;
  poll = setInterval(async () => {
    try {
      download = await (await api("/cluster/models/download")).json();
    } catch {
      return;
    }
    if (!download.active) {
      clearInterval(poll);
      poll = null;
      if (download.error) {
        downloadError = `Download failed: ${download.error}`;
        startWhenDone = null;
      } else if (download.done && startWhenDone) {
        const file = startWhenDone;
        startWhenDone = null;
        await refreshModels();
        switchModel(file);
      }
    }
    updateOnboarding();
  }, 1000);
}

export function initOnboarding(container) {
  container.addEventListener("click", (event) => {
    const target = event.target.closest("#onboarding [data-action]");
    if (!target) return;
    if (target.dataset.action === "ob-download") {
      target.disabled = true;
      target.textContent = "Starting…";
      startDownload(target.dataset.repo, target.dataset.file);
    } else if (target.dataset.action === "ob-start") {
      switchModel(target.dataset.file, target.dataset.nodeId || null, target.dataset.node || null);
    }
  });

  // A download already running (started before this page loaded) shows here.
  api("/cluster/models/download")
    .then((response) => response.json())
    .then((status) => {
      if (status.active) {
        download = status;
        watchDownload();
      }
    })
    .catch(() => {});
}

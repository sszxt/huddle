// The cluster page (#/cluster): every node as a card, the links between them,
// and where the model's layers sit.
//
// Open WebUI has no cluster page, so this follows its design system rather
// than a screen of it: the Workspace page's header and padding
// (routes/(app)/workspace/+layout.svelte), its model cards' borders and type
// (workspace/Models.svelte), the gray palette and Heroicons outline icons.
// Everything shown comes from one coordinator route, /cluster/nodes, which
// gathers each node's own agent report; nothing here is estimated.

import { icon } from "./icons.js";
import { escapeHtml } from "./markdown.js";
import { displayName, isMobile, setShowSidebar, state } from "./store.js";
import { api, errorDetail } from "./ui.js";

const REFRESH_MS = 3000;
// Below this width the topology stacks and the links give way to text.
const STACK_BELOW_PX = 880;
// Neutral fills for layer-bar segments, light to dark, one per device.
const SEGMENT_FILLS = ["#e3e3e3", "#9b9b9b", "#676767", "#4e4e4e", "#cdcdcd", "#b4b4b4"];

let overview = null;
let fetchedAt = 0;
let failure = null;
let timer = null;
let polling = false;
let ticker = null;
let observer = null;

const isOpen = () => state.route.name === "cluster";

// --- Formatting ------------------------------------------------------------

const gb = (mib) => (mib / 1024).toFixed(1);

function formatRtt(ms) {
  if (ms === null || ms === undefined) return null;
  return ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms)} ms`;
}

function formatUptime(seconds) {
  if (!seconds && seconds !== 0) return null;
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  if (days > 0) return `up ${days}d ${hours % 24}h`;
  if (hours > 0) return `up ${hours}h ${minutes % 60}m`;
  return `up ${Math.max(minutes, 1)}m`;
}

function formatLink(mbps) {
  if (!mbps) return null;
  return mbps >= 1000 ? `${+(mbps / 1000).toFixed(1)} Gb/s` : `${mbps} Mb/s`;
}

/** "Intel(R) Core(TM) i7-14700" → "Intel Core i7-14700". */
function cleanCpu(model) {
  if (!model) return null;
  return model
    .replace(/\((R|TM|tm|r)\)/g, "")
    .replace(/\s+CPU\s+@.*$/, "")
    .replace(/\s+\d+-Core Processor$/i, "")
    .replace(/\s+/g, " ")
    .trim();
}

/** "NVIDIA GeForce RTX 5070" → "RTX 5070", for tight labels only. */
function shortGpu(name) {
  return (name ?? "").replace(/^NVIDIA\s+(GeForce\s+)?/i, "").replace(/\(R\)/g, "").trim();
}

/** The commit a llama.cpp version string names, e.g. "4c9233c". */
function buildCommit(version) {
  return version?.match(/commit\s+([0-9a-f]+)/i)?.[1] ?? null;
}

/** Same rule as llamacpp.same_build: short hashes are truncated differently. */
function sameBuild(a, b) {
  if (!a || !b) return true;
  return a.startsWith(b) || b.startsWith(a);
}

// --- Derived facts ------------------------------------------------------------

/** Same state names as the TUI's cluster_state(). */
function clusterState(cluster) {
  if (cluster.starting) return "starting";
  if (cluster.running) return "running";
  if (cluster.desired && !cluster.running) return "degraded";
  return "stopped";
}

function discreteGpus(nodes) {
  return nodes.flatMap((node) =>
    (node.hardware?.devices ?? []).filter((d) => !d.is_rpc && !d.unified_memory && d.total_mib)
  );
}

/** Plan entries for a node's devices, matched by name in enumeration order. */
function devicePlans(node, plan) {
  const entries = (plan?.placement ?? []).filter((p) => p.node === node.name);
  return (node.hardware?.devices ?? [])
    .filter((device) => !device.is_rpc)
    .map((device) => {
      const at = entries.findIndex((p) => p.name === device.name);
      return at < 0 ? null : entries.splice(at, 1)[0];
    });
}

/**
 * Segments of the layer bar, in llama.cpp's entry order: entries 0..n_layers,
 * the last being the output head. llama.cpp's own report wins when present;
 * otherwise the plan says it — CPU takes the first n_layers + 1 - ngl entries,
 * then each device its tensor_split share, in enumeration order.
 */
function layerSegments(data) {
  const plan = data.cluster.plan;
  if (!plan) return null;
  const byId = new Map(plan.placement.map((p) => [p.id, p]));
  const head = data.cluster.head_node;
  const describe = (id) => {
    if (id.toUpperCase() === "CPU") return { node: head, name: "CPU", cpu: true };
    const entry = byId.get(id);
    return entry ? { node: entry.node, name: entry.name, cpu: false } : { node: "", name: id, cpu: false };
  };

  const reported = Object.entries(data.placement ?? {}).filter(([, layers]) => layers.length);
  if (reported.length) {
    const segments = reported.map(([id, layers]) => ({
      id,
      ...describe(id),
      first: Math.min(...layers),
      last: Math.max(...layers),
      count: layers.length,
    }));
    segments.sort((a, b) => a.first - b.first);
    return { source: "from llama.cpp", total: plan.n_layers + 1, nLayers: plan.n_layers, segments };
  }

  const segments = [];
  let next = 0;
  const onCpu = Math.max(0, plan.n_layers + 1 - plan.ngl);
  if (onCpu > 0) {
    segments.push({ id: "CPU", ...describe("CPU"), first: 0, last: onCpu - 1, count: onCpu });
    next = onCpu;
  }
  plan.placement.forEach((p, i) => {
    const count = Math.round(plan.tensor_split[i] ?? 0);
    if (count <= 0) return;
    segments.push({ id: p.id, node: p.node, name: p.name, cpu: false, first: next, last: next + count - 1, count });
    next += count;
  });
  return { source: "planned", total: plan.n_layers + 1, nLayers: plan.n_layers, segments };
}

// --- Markup ---------------------------------------------------------------------

function meter(fraction, tooltip) {
  const width = Math.max(0, Math.min(100, fraction * 100));
  return (
    `<div class="cv-meter" data-tooltip="${escapeHtml(tooltip)}">` +
    `<div class="cv-meter-fill" style="width: ${width.toFixed(1)}%"></div></div>`
  );
}

function statePill(data) {
  const cluster = data.cluster;
  const kind = clusterState(cluster);
  const label = { running: "Serving", starting: "Starting", degraded: "Degraded", stopped: "Stopped" }[kind];
  const detail =
    kind === "running" && data.tokens_per_sec
      ? ` · ${data.tokens_per_sec.toFixed(1)} tok/s`
      : kind === "starting"
        ? " · loading the model"
        : "";
  const tip = kind === "degraded" && cluster.last_failure ? ` data-tooltip="${escapeHtml(cluster.last_failure)}"` : "";
  return `<div class="cv-state cv-state-${kind}"${tip}><span class="cv-state-dot"></span>${label}${escapeHtml(detail)}</div>`;
}

function stat(label, value, sub, numeric = true) {
  return (
    `<div class="cv-stat"><div class="cv-stat-label">${label}</div>` +
    `<div class="cv-stat-value${numeric ? " cv-stat-number" : ""}">${value}</div>` +
    `<div class="cv-stat-sub">${sub}</div></div>`
  );
}

function statsHtml(data) {
  const cluster = data.cluster;
  const plan = cluster.plan;
  const gpus = discreteGpus(data.nodes);
  const total = gpus.reduce((sum, d) => sum + d.total_mib, 0);
  const used = gpus.reduce((sum, d) => sum + (d.total_mib - (d.free_mib ?? d.total_mib)), 0);
  const model = cluster.model ? escapeHtml(displayName(cluster.model)) : "None loaded";
  return (
    `<div class="cv-stats">` +
    stat(
      "Model",
      `<span class="cv-truncate" data-tooltip="${escapeHtml(cluster.model ?? "")}">${model}</span>`,
      cluster.running ? "loaded" : cluster.starting ? "loading" : "not serving",
      false
    ) +
    stat("Speed", data.tokens_per_sec ? `${data.tokens_per_sec.toFixed(1)} tok/s` : "—", "last reply") +
    stat("GPU memory", total ? `${gb(used)} / ${gb(total)} GB` : "—", `${gpus.length} discrete GPU${gpus.length === 1 ? "" : "s"}`) +
    stat("On GPU", plan ? `${plan.n_gpu_layers} / ${plan.n_layers}` : "—", "layers") +
    `</div>`
  );
}

function alertsHtml(data) {
  const cluster = data.cluster;
  const parts = [];
  if (cluster.last_failure) {
    parts.push(
      `<div class="msg-error"><div class="msg-error-icon">${icon("info", "size-5")}</div>` +
        `<div class="msg-error-text">${escapeHtml(cluster.last_failure)}</div></div>`
    );
  }
  for (const warning of cluster.plan?.warnings ?? []) {
    parts.push(`<div class="cv-note">${icon("info", "size-4")}<span>${escapeHtml(warning)}</span></div>`);
  }
  const unreachable = cluster.plan?.unreachable ?? [];
  if (unreachable.length) {
    parts.push(
      `<div class="cv-note">${icon("info", "size-4")}<span>Planned without ${escapeHtml(unreachable.join(", "))}: ` +
        `unreachable when the model was loaded.</span></div>`
    );
  }
  return parts.length ? `<div class="cv-alerts">${parts.join("")}</div>` : "";
}

const ROLE_LABEL = { head: "Head", worker: "Worker", idle: "Idle", offline: "Offline" };

function addressRows(node, headNode, stacked) {
  const rows = (node.system?.addresses ?? []).map(
    (a) =>
      `<div class="cv-addr"><span class="cv-if">${escapeHtml(a.interface)}</span>` +
      `<span class="cv-ip">${escapeHtml(a.address)}</span>` +
      (formatLink(a.link_mbps) ? `<span class="cv-dim">${formatLink(a.link_mbps)}</span>` : "") +
      `</div>`
  );
  if (node.role !== "head" && node.address) {
    const rtt = formatRtt(node.rtt_ms);
    const via = stacked && rtt ? ` · ${rtt} from ${escapeHtml(headNode)}` : "";
    rows.push(
      `<div class="cv-addr cv-dim">dialled at ` +
        `<span class="cv-ip">${escapeHtml(node.address)}:${node.agent_port}</span>${via}</div>`
    );
  }
  return rows.length ? `<div class="cv-addresses">${rows.join("")}</div>` : "";
}

function gpuHtml(device, planned) {
  const title =
    `<div class="cv-row"><span class="cv-gpu-name" data-tooltip="${escapeHtml(device.name)}">${escapeHtml(device.name)}</span>` +
    `<span class="cv-dim cv-mono">${escapeHtml(device.id)}</span></div>`;
  let chip = "";
  if (planned?.skipped) {
    chip = `<span class="cv-chip cv-chip-muted" data-tooltip="${escapeHtml(planned.skipped)}">not used</span>`;
  } else if (planned?.layers) {
    chip = `<span class="cv-chip">${planned.layers} layers</span>`;
  }
  let memory;
  if (device.unified_memory) {
    memory = `<div class="cv-row cv-dim"><span>Shared with system RAM</span>${chip}</div>`;
  } else if (device.total_mib) {
    const used = device.total_mib - (device.free_mib ?? device.total_mib);
    memory =
      `<div class="cv-row"><span class="cv-label">VRAM</span><span class="cv-value cv-num">${gb(used)} / ${gb(device.total_mib)} GB</span></div>` +
      meter(used / device.total_mib, `${used.toLocaleString()} of ${device.total_mib.toLocaleString()} MiB in use`) +
      (chip ? `<div class="cv-row cv-row-end">${chip}</div>` : "");
  } else {
    memory = chip ? `<div class="cv-row cv-row-end">${chip}</div>` : "";
  }
  const live = [];
  if (device.util_pct !== null && device.util_pct !== undefined) live.push(`${device.util_pct}% busy`);
  if (device.temp_c !== null && device.temp_c !== undefined) live.push(`${device.temp_c}°C`);
  if (device.power_w !== null && device.power_w !== undefined) live.push(`${Math.round(device.power_w)} W`);
  const stats = live.length ? `<div class="cv-gpu-stats">${live.join(" · ")}</div>` : "";
  return `<div class="cv-gpu">${title}${memory}${stats}</div>`;
}

function nodeFooter(node, data, headCommit) {
  const items = [];
  if (node.layers) items.push(`<span>${node.layers} layers</span>`);
  else if (node.role !== "offline") items.push(`<span class="cv-dim">no layers</span>`);
  const commit = buildCommit(node.hardware?.llamacpp_version);
  if (commit) {
    const skew = !sameBuild(commit, headCommit);
    items.push(
      skew
        ? `<span class="cv-bad" data-tooltip="Every node must run the same llama.cpp build; the RPC handshake rejects this one.">llama.cpp ${escapeHtml(commit)} ≠ head</span>`
        : `<span>llama.cpp ${escapeHtml(commit)}</span>`
    );
  }
  if (node.role === "head") {
    items.push(`<span>llama-server :${data.backend_port}${data.cluster.running ? "" : " · stopped"}</span>`);
  } else if (node.rpc) {
    const rpc = node.rpc;
    const text = rpc.running
      ? `rpc-server :${rpc.port}${rpc.cache ? " · cache on" : ""}`
      : rpc.foreign
        ? `rpc-server :${rpc.port} · started outside this agent`
        : `rpc-server :${rpc.port} · stopped`;
    items.push(`<span>${text}</span>`);
  }
  return `<div class="cv-node-foot">${items.join('<span class="cv-sep">·</span>')}</div>`;
}

function nodeHtml(node, data, stacked, headCommit) {
  const system = node.system;
  const hardware = node.hardware;
  const subParts = [system?.os, formatUptime(system?.uptime_s)].filter(Boolean);
  const head =
    `<div class="cv-node-head"><div class="cv-node-icon">${icon("computerDesktop", "cv-computer", 1.2)}` +
    `<span class="cv-node-dot cv-dot-${node.role}"></span></div>` +
    `<div class="cv-node-title"><div class="cv-node-name"><span class="cv-truncate">${escapeHtml(node.name)}</span>` +
    `<span class="cv-role cv-role-${node.role}">${ROLE_LABEL[node.role]}</span></div>` +
    `<div class="cv-node-sub">${escapeHtml(subParts.join(" · ") || (node.role === "offline" ? "Not responding" : ""))}</div>` +
    `</div></div>`;

  if (node.role === "offline") {
    return (
      `<div class="cv-node cv-node-offline" data-node="${escapeHtml(node.name)}">${head}` +
      addressRows(node, data.cluster.head_node, stacked) +
      `<div class="cv-offline-reason">${escapeHtml(node.error ?? "No answer from its agent.")}</div></div>`
    );
  }

  const sections = [];
  if (system || hardware) {
    const cpu = cleanCpu(system?.cpu_model);
    const threads = system?.cpu_threads || hardware?.cpu_count;
    const cpuLabel = [cpu, threads ? `${threads} threads` : null].filter(Boolean).join(" · ");
    let cpuRows = `<div class="cv-row"><span class="cv-label">CPU</span><span class="cv-value cv-truncate" data-tooltip="${escapeHtml(system?.cpu_model ?? "")}">${escapeHtml(cpuLabel || "—")}</span></div>`;
    if (system?.cpu_pct !== null && system?.cpu_pct !== undefined) {
      cpuRows +=
        meter(system.cpu_pct / 100, `${system.cpu_pct}% busy${system.load_1m !== null ? ` · load ${system.load_1m}` : ""}`) +
        `<div class="cv-row cv-row-end cv-dim">${Math.round(system.cpu_pct)}% busy</div>`;
    }
    let memRows = "";
    if (hardware?.ram_total_mib) {
      const used = hardware.ram_total_mib - hardware.ram_available_mib;
      memRows =
        `<div class="cv-row"><span class="cv-label">Memory</span><span class="cv-value cv-num">${gb(used)} / ${gb(hardware.ram_total_mib)} GB</span></div>` +
        meter(used / hardware.ram_total_mib, `${used.toLocaleString()} of ${hardware.ram_total_mib.toLocaleString()} MiB in use`);
    }
    sections.push(`<div class="cv-section">${cpuRows}${memRows}</div>`);
  }

  const devices = (hardware?.devices ?? []).filter((d) => !d.is_rpc);
  if (devices.length) {
    const plans = devicePlans(node, data.cluster.plan);
    sections.push(`<div class="cv-section">${devices.map((d, i) => gpuHtml(d, plans[i])).join("")}</div>`);
  }

  const notes = [];
  if (node.error) notes.push(`<div class="cv-bad-note">${escapeHtml(node.error)}</div>`);
  if (!system) notes.push(`<div class="cv-dim cv-small">Update Huddle on this node for OS, CPU and network details.</div>`);

  return (
    `<div class="cv-node" data-node="${escapeHtml(node.name)}">${head}` +
    addressRows(node, data.cluster.head_node, stacked) +
    sections.join("") +
    notes.join("") +
    nodeFooter(node, data, headCommit) +
    `</div>`
  );
}

function layersHtml(data) {
  const layout = layerSegments(data);
  if (!layout) {
    return (
      `<div class="cv-panel"><div class="cv-panel-head"><div class="cv-panel-title">Layers</div></div>` +
      `<div class="cv-dim cv-small">No model is loaded. Pick one from the model menu in a chat to see how it is split.</div></div>`
    );
  }
  const bar = layout.segments
    .map((s, i) => {
      const fill = s.cpu ? null : SEGMENT_FILLS[i % SEGMENT_FILLS.length];
      const dark = fill && ["#e3e3e3", "#cdcdcd", "#b4b4b4", "#9b9b9b"].includes(fill);
      const style = `flex-grow: ${s.count};${fill ? ` background: ${fill};` : ""}`;
      return (
        `<div class="cv-seg${s.cpu ? " cv-seg-cpu" : ""}${dark ? " cv-seg-dark" : ""}" style="${style}" ` +
        `data-tooltip="${escapeHtml(`${s.node} · ${s.name} · ${s.count} entries`)}">` +
        `<span>${escapeHtml(s.cpu ? "CPU" : s.node)}</span></div>`
      );
    })
    .join("");
  const legend = layout.segments
    .map((s, i) => {
      const last = Math.min(s.last, layout.nLayers - 1);
      const range = s.first > last ? "output head" : `${s.first}–${last}${s.last >= layout.nLayers ? " + output" : ""}`;
      const swatch = s.cpu
        ? `<span class="cv-swatch cv-seg-cpu"></span>`
        : `<span class="cv-swatch" style="background: ${SEGMENT_FILLS[i % SEGMENT_FILLS.length]}"></span>`;
      return (
        `<div class="cv-legend-item">${swatch}<span class="cv-legend-name">${escapeHtml(s.node)} · ${escapeHtml(s.cpu ? "CPU" : shortGpu(s.name))}</span>` +
        `<span class="cv-dim">layers ${range}</span></div>`
      );
    })
    .join("");
  return (
    `<div class="cv-panel"><div class="cv-panel-head"><div class="cv-panel-title">Layers</div>` +
    `<div class="cv-dim cv-small">${layout.nLayers} layers + output head · ${layout.source}</div></div>` +
    `<div class="cv-bar">${bar}</div><div class="cv-legend">${legend}</div></div>`
  );
}

function topologyHtml(data, stacked) {
  const [head, ...peers] = data.nodes;
  const headCommit = buildCommit(head.hardware?.llamacpp_version);
  const peerCards = peers.map((p) => nodeHtml(p, data, stacked, headCommit)).join("");
  const layout = peers.length === 0 ? " cv-single" : stacked ? " cv-stacked" : "";
  return (
    `<div class="cv-topology${layout}"><svg class="cv-links" aria-hidden="true"></svg><div class="cv-link-labels"></div>` +
    `<div class="cv-head-col">${nodeHtml(head, data, stacked, headCommit)}</div>` +
    (peers.length ? `<div class="cv-gap"></div><div class="cv-peer-col">${peerCards}</div>` : "") +
    `</div>`
  );
}

function bodyHtml(data, stacked) {
  const count = data.nodes.length;
  return (
    `<div class="cv-header"><div class="ws-title">Cluster<div class="ws-title-divider"></div>` +
    `<span class="ws-count">${count}</span></div>${statePill(data)}</div>` +
    statsHtml(data) +
    alertsHtml(data) +
    topologyHtml(data, stacked) +
    layersHtml(data)
  );
}

function liveHtml() {
  if (failure) {
    return `<span class="cv-live-dot cv-live-bad"></span><span>Connection lost · retrying</span>`;
  }
  if (!fetchedAt) return `<span class="cv-live-dot"></span><span>Connecting…</span>`;
  const seconds = Math.max(0, Math.round((Date.now() - fetchedAt) / 1000));
  return `<span class="cv-live-dot cv-live-ok"></span><span>Live · updated ${seconds}s ago</span>`;
}

// --- Links between cards ------------------------------------------------------------

function drawLinks() {
  const topology = document.querySelector("#chat-container .cv-topology");
  if (!topology || !overview) return;
  const svg = topology.querySelector(".cv-links");
  const labels = topology.querySelector(".cv-link-labels");
  svg.innerHTML = "";
  labels.innerHTML = "";
  if (topology.classList.contains("cv-stacked") || topology.classList.contains("cv-single")) return;

  const box = topology.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
  const headCard = topology.querySelector(".cv-head-col .cv-node");
  if (!headCard) return;
  const h = headCard.getBoundingClientRect();
  const x1 = h.right - box.left;
  const y1 = h.top - box.top + h.height / 2;
  const running = overview.cluster.running;

  const paths = [];
  const pills = [];
  for (const node of overview.nodes.slice(1)) {
    const card = topology.querySelector(`.cv-peer-col .cv-node[data-node="${CSS.escape(node.name)}"]`);
    if (!card) continue;
    const r = card.getBoundingClientRect();
    const x2 = r.left - box.left;
    const y2 = r.top - box.top + Math.min(r.height / 2, 120);
    const bend = (x2 - x1) / 2;
    const kind = node.role === "worker" ? (running ? "cv-link-live" : "cv-link-worker") : node.role === "idle" ? "cv-link-idle" : "cv-link-offline";
    paths.push(`<path class="cv-link ${kind}" d="M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}"/>`);
    const text = node.role === "offline" ? "offline" : formatRtt(node.rtt_ms) ?? "";
    if (text) {
      pills.push(
        `<div class="cv-link-label${node.role === "offline" ? " cv-link-label-bad" : ""}" style="left: ${(x1 + x2) / 2}px; top: ${(y1 + y2) / 2}px" ` +
          `data-tooltip="${node.role === "offline" ? "No answer from its agent" : `Network round trip from ${escapeHtml(overview.cluster.head_node)} (TCP connect to its agent)`}">${text}</div>`
      );
    }
  }
  svg.innerHTML = paths.join("");
  labels.innerHTML = pills.join("");
}

function isStacked() {
  const container = document.querySelector("#chat-container .ws-container");
  return container ? container.clientWidth < STACK_BELOW_PX : isMobile();
}

// --- Rendering and polling ------------------------------------------------------------

function renderBody() {
  const body = document.querySelector("#chat-container .cv-body");
  if (!body) return;
  if (!overview) {
    body.innerHTML = failure
      ? `<div class="msg-error"><div class="msg-error-icon">${icon("info", "size-5")}</div><div class="msg-error-text">${escapeHtml(failure)}</div></div>`
      : `<div class="cv-loading">Loading the cluster…</div>`;
    return;
  }
  body.innerHTML = bodyHtml(overview, isStacked());
  drawLinks();
}

function renderLive() {
  const live = document.querySelector("#chat-container .cv-live");
  if (live) live.innerHTML = liveHtml();
}

export function renderClusterPage() {
  const container = document.getElementById("chat-container");
  const toggle =
    isMobile() && !state.showSidebar
      ? `<div class="ws-sidebar-toggle"><button type="button" class="nav-sidebar-button" data-action="open-sidebar" aria-label="Open Sidebar">` +
        `<div class="nav-sidebar-icon">${icon("sidebar", "size-5")}</div></button></div>`
      : "";
  container.innerHTML =
    `<div class="workspace cluster-page"><nav class="ws-nav"><div class="ws-nav-row">${toggle}` +
    `<div><div class="ws-tabs scrollbar-none"><a class="ws-tab" href="#/cluster">Cluster</a></div></div>` +
    `<div class="cv-live">${liveHtml()}</div></div></nav>` +
    `<div class="ws-container" id="cluster-container"><div class="cv-body"></div></div></div>`;
  renderBody();

  observer?.disconnect();
  let wasStacked = isStacked();
  observer = new ResizeObserver(() => {
    if (!isOpen()) return;
    const stacked = isStacked();
    if (stacked !== wasStacked) {
      wasStacked = stacked;
      renderBody();
    } else {
      drawLinks();
    }
  });
  observer.observe(container);
}

/** One request at a time: the next is scheduled only after this one lands. */
async function poll() {
  timer = null;
  if (polling || !isOpen() || document.hidden) return;
  polling = true;
  try {
    const response = await api("/cluster/nodes");
    if (!response.ok) throw new Error(await errorDetail(response));
    overview = await response.json();
    fetchedAt = Date.now();
    failure = null;
  } catch (error) {
    failure = error.message || "Could not reach the cluster API";
  } finally {
    polling = false;
  }
  if (!isOpen()) return;
  renderBody();
  renderLive();
  if (!timer && !document.hidden) timer = setTimeout(poll, REFRESH_MS);
}

/** Enter the page: draw it and start the refresh loop. */
export function openClusterPage() {
  renderClusterPage();
  if (!timer && !polling) poll();
  if (!ticker) ticker = setInterval(() => (isOpen() ? renderLive() : closeClusterPage()), 1000);
}

/** Leave the page: stop every timer, so nothing polls in the background. */
export function closeClusterPage() {
  clearTimeout(timer);
  timer = null;
  clearInterval(ticker);
  ticker = null;
  observer?.disconnect();
  observer = null;
}

export function initClusterPage() {
  document.addEventListener("visibilitychange", () => {
    if (!isOpen()) return;
    if (document.hidden) {
      clearTimeout(timer);
      timer = null;
    } else if (!timer && !polling) {
      poll();
    }
  });
  document.getElementById("chat-container").addEventListener("click", (event) => {
    if (!isOpen()) return;
    if (event.target.closest("[data-action=open-sidebar]")) setShowSidebar(true);
  });
}

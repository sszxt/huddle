// What the cluster is doing: node name, health, every PC's models, model loads.

import { displayName, emit, state } from "./store.js";
import { api, errorDetail, toast } from "./ui.js";

export async function refreshModels() {
  try {
    const response = await api("/cluster/models");
    const data = await response.json();
    // An older node answers with file names only, all of them its own.
    state.models = data.entries ?? (data.available ?? []).map((file) => ({ file, local: true }));
    state.loaded = data.loaded ?? null;
    state.loading = data.loading ?? null;
    state.head = data.head ?? null;
    state.headId = data.head_id ?? null;
  } catch {
    state.models = [];
  }
  emit("models");
}

export async function refreshHealth() {
  let health = null;
  try {
    health = await (await api("/health")).json();
  } catch {
    health = { status: "unreachable" };
  }
  const previous = state.health;
  const changed =
    health.status !== previous?.status ||
    health.model !== previous?.model ||
    health.head !== previous?.head;
  state.health = health;
  if (health.node && health.node !== state.node) {
    state.node = health.node;
    emit("node");
  }
  if (changed) await refreshModels();
}

/** Whether a model entry is the one the cluster is serving. */
export function isLoaded(entry) {
  if (entry.file !== state.loaded) return false;
  return !entry.node_id || !state.headId || entry.node_id === state.headId;
}

/**
 * Load a model, on the PC that holds it. llama.cpp holds one model per
 * process, so this restarts the cluster, and for a large model that takes
 * minutes. Whichever PC was serving hands over.
 */
export async function switchModel(file, nodeId = null, nodeName = null) {
  if (!file || state.switching) return;
  if (file === state.loaded && (!nodeId || nodeId === state.headId)) return;
  state.switching = file;
  // Loading from this page goes to this PC unless the file is elsewhere.
  state.switchingOn = nodeName ?? state.node;
  emit("models");
  const where = nodeName ? ` on ${nodeName}` : "";
  const dismiss = toast(
    `Loading ${displayName(file)}${where}. Large models take minutes to load.`,
    "info",
    0
  );
  try {
    const body = nodeId ? { model: file, node_id: nodeId } : { model: file };
    const response = await api("/cluster/model", { method: "POST", json: body });
    if (!response.ok) throw new Error(await errorDetail(response));
    toast(`${displayName(file)} is loaded`, "success");
  } catch (error) {
    toast(`Could not load ${displayName(file)}: ${error.message}`, "error");
  } finally {
    dismiss();
    state.switching = null;
    state.switchingOn = null;
    await refreshHealth();
    await refreshModels();
  }
}

export function startPolling() {
  const tick = async () => {
    await refreshHealth();
    // Poll faster while something is changing, so the UI follows along.
    const busy = state.switching || state.health?.status === "starting";
    setTimeout(tick, busy ? 2000 : 5000);
  };
  tick();
}

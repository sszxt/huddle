// What the cluster is doing: node name, health, models on disk, model switches.

import { displayName, emit, state } from "./store.js";
import { api, errorDetail, toast } from "./ui.js";

export async function refreshModels() {
  try {
    const response = await api("/cluster/models");
    const data = await response.json();
    state.models = data.available ?? [];
    state.loaded = data.loaded ?? null;
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
  const changed = health.status !== state.health?.status || health.model !== state.health?.model;
  state.health = health;
  if (health.node && health.node !== state.node) {
    state.node = health.node;
    emit("node");
  }
  if (changed) await refreshModels();
}

/**
 * Load another model. llama.cpp holds one model per process, so this restarts
 * the whole cluster; for a large model that takes minutes.
 */
export async function switchModel(file) {
  if (!file || file === state.loaded || state.switching) return;
  state.switching = file;
  emit("models");
  const dismiss = toast(
    `Loading ${displayName(file)}. Large models take minutes to load.`,
    "info",
    0
  );
  try {
    const response = await api("/cluster/model", { method: "POST", json: { model: file } });
    if (!response.ok) throw new Error(await errorDetail(response));
    toast(`${displayName(file)} is loaded`, "success");
  } catch (error) {
    toast(`Could not load ${displayName(file)}: ${error.message}`, "error");
  } finally {
    dismiss();
    state.switching = null;
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

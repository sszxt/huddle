// Shared UI state, browser persistence, and a tiny event bus.
//
// Chats live only in this browser's localStorage. Every access is wrapped:
// storage can be blocked (private windows, strict settings) and the page must
// still work, just without history.

const CHATS_KEY = "huddle.chats";
const SIDEBAR_KEY = "huddle.showSidebar";
const API_KEY_KEY = "huddle.apiKey";

export const MOBILE_BREAKPOINT = 768;

function read(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function write(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage unavailable or full: the chat still works for this page load.
  }
}

export const isMobile = () => window.innerWidth < MOBILE_BREAKPOINT;

export const state = {
  node: "",
  health: null,
  models: [],
  loaded: null,
  switching: null,
  chats: read(CHATS_KEY, []),
  current: null,
  route: { name: "home" },
  temporary: false,
  prompt: "",
  generating: null,
  showSidebar: isMobile() ? false : read(SIDEBAR_KEY, true),
  chatsOpen: true,
  showControls: false,
  apiKey: read(API_KEY_KEY, null),
};

const listeners = new Map();

export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(fn);
}

export function emit(event, detail) {
  for (const fn of listeners.get(event) ?? []) fn(detail);
}

export const nowSeconds = () => Math.floor(Date.now() / 1000);

export const newId = () =>
  crypto.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

export function saveChats() {
  write(CHATS_KEY, state.chats);
}

/** Persist a chat after it changed. Temporary chats never touch storage. */
export function touchChat(chat) {
  if (chat.temporary) return;
  chat.updated_at = nowSeconds();
  if (!state.chats.includes(chat)) state.chats.unshift(chat);
  state.chats.sort((a, b) => b.updated_at - a.updated_at);
  saveChats();
  emit("chats");
}

export function deleteChat(id) {
  state.chats = state.chats.filter((chat) => chat.id !== id);
  saveChats();
  emit("chats");
}

export function renameChat(id, title) {
  const chat = state.chats.find((c) => c.id === id);
  if (!chat || !title.trim()) return;
  chat.title = title.trim();
  saveChats();
  emit("chats");
}

export function setShowSidebar(show) {
  state.showSidebar = show;
  if (!isMobile()) write(SIDEBAR_KEY, show);
  emit("sidebar");
}

export function setApiKey(key) {
  state.apiKey = key;
  write(API_KEY_KEY, key);
}

/** Open WebUI's getTimeRange: the sidebar and search group chats by these. */
export function timeRange(timestamp) {
  const now = new Date();
  const date = new Date(timestamp * 1000);
  const diffDays = (now.getTime() - date.getTime()) / (1000 * 3600 * 24);
  const sameMonth = now.getFullYear() === date.getFullYear() && now.getMonth() === date.getMonth();
  if (sameMonth && now.getDate() === date.getDate()) return "Today";
  if (sameMonth && now.getDate() - date.getDate() === 1) return "Yesterday";
  if (diffDays <= 7) return "Previous 7 days";
  if (diffDays <= 30) return "Previous 30 days";
  if (now.getFullYear() === date.getFullYear()) {
    return date.toLocaleString("default", { month: "long" });
  }
  return date.getFullYear().toString();
}

/** A model file name as the UI shows it. */
export const displayName = (file) => (file ? file.replace(/\.gguf$/i, "") : "");

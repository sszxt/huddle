// A signed-in user's chats, kept by the PC serving this page instead of the browser.
//
// Every change saves the whole chat. Saves are debounced per chat and never
// overlap for one chat: a change made while one is in flight is sent after it,
// so the server always ends with the latest copy. Deleting waits for any save
// still in flight, which would otherwise bring the chat back.

import { sessionEnded } from "./session.js";
import { clearLocalChats, emit, localChats, state, usePersistence } from "./store.js";
import { api, confirmDialog, errorDetail, toast } from "./ui.js";

const SAVE_DELAY_MS = 400;
const timers = new Map();
const inFlight = new Map();
const changedAgain = new Set();
let warned = false;

/** Load this user's chats and route every later change to the server. */
export async function loadServerHistory() {
  usePersistence({ save: queueSave, remove: removeChat });
  window.addEventListener("pagehide", flushPending);
  try {
    const response = await api("/api/chats");
    if (response.status === 401) return sessionEnded();
    if (!response.ok) throw new Error(await errorDetail(response));
    state.chats = await response.json();
  } catch (error) {
    toast(`Could not load your chats: ${error.message}`, "error", 0);
  }
}

function body(chat) {
  // `editing` is where the page is, not part of the chat.
  return JSON.stringify(chat, (key, value) => (key === "editing" ? undefined : value));
}

function put(chat, options = {}) {
  return api(`/api/chats/${encodeURIComponent(chat.id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: body(chat),
    ...options,
  });
}

function queueSave(chat) {
  clearTimeout(timers.get(chat.id));
  timers.set(
    chat.id,
    setTimeout(() => {
      timers.delete(chat.id);
      send(chat);
    }, SAVE_DELAY_MS)
  );
}

function send(chat) {
  if (inFlight.has(chat.id)) {
    changedAgain.add(chat.id);
    return;
  }
  const saving = save(chat).finally(() => {
    inFlight.delete(chat.id);
    if (changedAgain.delete(chat.id) && state.chats.includes(chat)) send(chat);
  });
  inFlight.set(chat.id, saving);
}

async function save(chat) {
  try {
    const response = await put(chat);
    if (response.status === 401) return sessionEnded();
    if (!response.ok) throw new Error(await errorDetail(response));
    warned = false;
  } catch (error) {
    // Once per run of failures: a toast per keystroke in Controls helps no one.
    if (!warned) toast(`Could not save the chat: ${error.message}`, "error");
    warned = true;
  }
}

async function removeChat(id) {
  clearTimeout(timers.get(id));
  timers.delete(id);
  changedAgain.delete(id);
  await inFlight.get(id);
  try {
    const response = await api(`/api/chats/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (response.status === 401) return sessionEnded();
    if (!response.ok && response.status !== 404) throw new Error(await errorDetail(response));
  } catch (error) {
    toast(`Could not delete the chat: ${error.message}`, "error");
  }
}

/** Leaving the page: send what is still waiting. Browsers cap these at 64 KB. */
function flushPending() {
  for (const [id, timer] of timers) {
    clearTimeout(timer);
    const chat = state.chats.find((c) => c.id === id);
    if (chat) put(chat, { keepalive: true }).catch(() => {});
  }
  timers.clear();
}

function askedKey() {
  return `huddle.importAsked.${state.user.id}`;
}

/** Offer to move the chats this browser kept before sign-in into the account. */
export async function offerImport() {
  const chats = localChats().filter((c) => c?.id && !c.temporary && c.messages?.length);
  if (!chats.length) return;
  try {
    if (localStorage.getItem(askedKey())) return;
    localStorage.setItem(askedKey(), "1");
  } catch {
    // Without storage we cannot remember asking; asking each time is harmless.
  }
  const ok = await confirmDialog({
    title: "Import chats from this browser?",
    message:
      `This browser has ${chats.length} chat${chats.length === 1 ? "" : "s"} from before you signed in. ` +
      "Add them to your account, so they are kept with your history on this PC?",
    confirmLabel: "Import",
    cancelLabel: "Not now",
  });
  if (!ok) return;
  try {
    const response = await api("/api/chats/import", { method: "POST", json: { chats } });
    if (response.status === 401) return sessionEnded();
    if (!response.ok) throw new Error(await errorDetail(response));
    const { imported } = await response.json();
    clearLocalChats();
    const listed = await api("/api/chats");
    if (listed.ok) state.chats = await listed.json();
    emit("chats");
    toast(`Imported ${imported} chat${imported === 1 ? "" : "s"}.`, "success");
  } catch (error) {
    toast(`Could not import the chats: ${error.message}`, "error");
  }
}

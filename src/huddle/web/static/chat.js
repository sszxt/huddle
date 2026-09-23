// The chat view: home screen, message input, conversation and Controls panel.
// Markup follows Open WebUI v0.6.30: chat/Chat.svelte, Placeholder.svelte,
// Suggestions.svelte, MessageInput.svelte, Messages.svelte, Messages/*.svelte,
// ChatControls.svelte, Controls/Controls.svelte and Settings/Advanced/AdvancedParams.svelte.
//
// Buttons Open WebUI hides when a feature is off stay hidden here, since Huddle
// has nothing behind them: attach (+), integrations, dictation, voice mode.

import { icon } from "./icons.js";
import { bindCodeBlockActions, escapeHtml, renderMarkdown } from "./markdown.js";
import { navbarHtml } from "./navbar.js";
import {
  displayName,
  isMobile,
  newId,
  nowSeconds,
  saveChats,
  setApiKey,
  state,
  touchChat,
} from "./store.js";
import { api, confirmDialog, copyText, errorDetail, LOGO_URL, toast } from "./ui.js";

// Open WebUI's default prompt suggestions, from backend/open_webui/config.py.
const SUGGESTIONS = [
  {
    title: ["Help me study", "vocabulary for a college entrance exam"],
    content:
      "Help me study vocabulary: write a sentence for me to fill in the blank, and I'll try to pick the correct option.",
  },
  {
    title: ["Give me ideas", "for what to do with my kids' art"],
    content:
      "What are 5 creative things I could do with my kids' art? I don't want to throw them away, but it's also so much clutter.",
  },
  {
    title: ["Tell me a fun fact", "about the Roman Empire"],
    content: "Tell me a random fun fact about the Roman Empire",
  },
  {
    title: ["Show me a code snippet", "of a website's sticky header"],
    content: "Show me a code snippet of a website's sticky header in CSS and JavaScript.",
  },
  {
    title: ["Explain options trading", "if I'm familiar with buying and selling stocks"],
    content:
      "Explain options trading in simple terms if I'm familiar with buying and selling stocks.",
  },
  {
    title: ["Overcome procrastination", "give me tips"],
    content:
      "Could you start by asking me about instances when I procrastinate the most and then give me some suggestions to overcome it?",
  },
];

const PARAMS = {
  temperature: {
    label: "Temperature",
    tooltip:
      "The temperature of the model. Increasing the temperature will make the model answer more creatively.",
    initial: 0.8,
    min: 0,
    max: 2,
    step: 0.05,
    numberStep: "any",
  },
  max_tokens: {
    label: "max_tokens",
    tooltip:
      "This option sets the maximum number of tokens the model can generate in its response. Increasing this limit allows the model to provide longer answers, but it may also increase the likelihood of unhelpful or irrelevant content being generated.",
    initial: 128,
    min: -2,
    max: 131072,
    step: 1,
    numberStep: 1,
  },
};

// Params of the chat being composed, before it exists; reset by New Chat.
let draftParams = {};
let autoScroll = true;
const renderQueued = new Set();

const currentParams = () => state.current?.params ?? draftParams;
const isGenerating = (chat) => Boolean(chat && state.generating?.chatId === chat.id);

// --- Formatting helpers ---

function formatTime(seconds) {
  return new Date(seconds * 1000).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
}

/** Open WebUI's formatDate: "Today at 2:30 PM", "Yesterday at …", "09/15/2026 at …". */
function formatDate(seconds) {
  const date = new Date(seconds * 1000);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);
  const same = (a, b) => a.toDateString() === b.toDateString();
  if (same(date, today)) return `Today at ${formatTime(seconds)}`;
  if (same(date, yesterday)) return `Yesterday at ${formatTime(seconds)}`;
  const pad = (n) => String(n).padStart(2, "0");
  const day = `${pad(date.getMonth() + 1)}/${pad(date.getDate())}/${date.getFullYear()}`;
  return `${day} at ${formatTime(seconds)}`;
}

function shuffled(items) {
  return [...items].sort(() => Math.random() - 0.5);
}

// --- Message input (MessageInput.svelte) ---

function sendButtonHtml() {
  if (isGenerating(state.current)) {
    return (
      `<div class="mi-button-wrap"><button type="button" class="mi-stop" data-action="stop" data-tooltip="Stop" aria-label="Stop">` +
      `${icon("stop", "size-5")}</button></div>`
    );
  }
  const empty = state.prompt.trim() === "";
  return (
    `<div class="mi-button-wrap"><button type="submit" id="send-message-button" class="mi-send${empty ? " disabled" : ""}"` +
    `${empty ? " disabled" : ""} data-tooltip="Send message" aria-label="Send message">` +
    `${icon("arrowUp", "size-5")}</button></div>`
  );
}

function messageInputHtml(placeholder) {
  const dashed = state.temporary || state.current?.temporary ? " temporary" : "";
  return (
    `<div class="mi">` +
    `<div class="mi-top"><div class="mi-top-inner"><div class="mi-scroll-anchor">` +
    `<div class="mi-scroll-button-wrap" hidden><button type="button" class="mi-scroll-button" data-action="scroll-bottom" aria-label="Scroll to bottom">` +
    `${icon("arrowDown", "size-5")}</button></div></div></div></div>` +
    `<div class="mi-bg"><div class="mi-pad"><form class="mi-form">` +
    `<div id="message-input-container" class="mi-container${dashed}">` +
    `<div class="mi-text-wrap"><div id="chat-input-container" class="mi-text scrollbar-hidden">` +
    `<textarea id="chat-input" class="mi-textarea" rows="1" placeholder="${escapeHtml(placeholder)}" aria-label="${escapeHtml(placeholder)}"></textarea>` +
    `</div></div>` +
    `<div class="mi-bottom" dir="ltr"><div class="mi-left"></div>` +
    `<div class="mi-right">${sendButtonHtml()}</div></div>` +
    `</div><div class="mi-foot"></div></form></div></div></div>`
  );
}

function updateSendButton() {
  const slot = document.querySelector("#chat-content .mi-right");
  if (slot) slot.innerHTML = sendButtonHtml();
}

function resizeInput(textarea) {
  textarea.style.height = "";
  textarea.style.height = `${textarea.scrollHeight}px`;
}

// --- Home screen (Placeholder.svelte + Suggestions.svelte) ---

function suggestionsHtml() {
  const items = shuffled(SUGGESTIONS)
    .map(
      (s, idx) =>
        `<button type="button" role="listitem" class="sg-item waterfall" style="animation-delay: ${idx * 60}ms" data-prompt="${escapeHtml(s.content)}" data-search="${escapeHtml(`${s.title.join(" ")} ${s.content}`.toLowerCase())}">` +
        `<div class="sg-text"><div class="sg-title">${escapeHtml(s.title[0])}</div>` +
        `<div class="sg-sub">${escapeHtml(s.title[1])}</div></div></button>`
    )
    .join("");
  return (
    `<div class="sg-head"><span class="sg-suggested">${icon("bolt", "size-3")}Suggested</span>` +
    `<div class="sg-none" hidden>Huddle</div></div>` +
    `<div class="sg-list-wrap"><div role="list" class="sg-list scrollbar-none">${items}</div></div>`
  );
}

function filterSuggestions() {
  const list = document.querySelector("#chat-content .sg-list");
  if (!list) return;
  const q = state.prompt.trim().toLowerCase();
  let shown = 0;
  list.querySelectorAll(".sg-item").forEach((item) => {
    const match = q.length <= 500 && (!q || item.dataset.search.includes(q));
    item.hidden = !match;
    if (match) shown += 1;
  });
  document.querySelector("#chat-content .sg-suggested").hidden = shown === 0;
  document.querySelector("#chat-content .sg-none").hidden = shown !== 0;
}

function homeTitleHtml() {
  const model = state.switching ?? state.loaded;
  return model
    ? `<span class="ph-title-text">${escapeHtml(displayName(model))}</span>`
    : `Hello, ${escapeHtml(state.node || "there")}`;
}

/** Keep the home heading in step with the loaded model without a full re-render. */
export function updateHomeHeading() {
  const title = document.querySelector("#chat-content .ph-title");
  if (title) title.innerHTML = homeTitleHtml();
}

function homeHtml() {
  const title = homeTitleHtml();
  const temporary = state.temporary
    ? `<div class="ph-temporary-wrap" data-tooltip="This chat won't appear in history and your messages will not be saved." data-placement="top">` +
      `<div class="ph-temporary">${icon("eyeSlash", "size-4", 2.5)}Temporary Chat</div></div>`
    : "";
  return (
    `<div class="ph-wrap"><div class="placeholder">` +
    temporary +
    `<div class="ph-heading"><div class="ph-col">` +
    `<div class="ph-title-row">` +
    `<div class="ph-logo-wrap"><div class="ph-logos"><button type="button" aria-hidden="true" tabindex="-1">` +
    `<img src="${LOGO_URL}" class="ph-logo" alt="" draggable="false"></button></div></div>` +
    `<div class="ph-title">${title}</div>` +
    `</div>` +
    `<div class="ph-desc"><div></div></div>` +
    `<div class="ph-input">${messageInputHtml("How can I help you today?")}</div>` +
    `</div></div>` +
    `<div class="ph-suggestions"><div class="ph-suggestions-inner">${suggestionsHtml()}</div></div>` +
    `</div></div>`
  );
}

// --- Messages (Messages.svelte, Message.svelte, UserMessage.svelte, ResponseMessage.svelte) ---

function actionButton(action, iconName, tooltip, id, visibility = "") {
  return (
    `<button type="button" class="msg-action ${visibility}" data-action="${action}" data-id="${id}" data-tooltip="${tooltip}" data-placement="bottom" aria-label="${tooltip}">` +
    `${icon(iconName, "size-4", 2.3)}</button>`
  );
}

function skeletonHtml() {
  return '<span class="skeleton"><span class="skeleton-ping"></span><span class="skeleton-dot"></span></span>';
}

function errorHtml(error) {
  return (
    `<div class="msg-error"><div class="msg-error-icon">${icon("info", "size-5")}</div>` +
    `<div class="msg-error-text">${escapeHtml(error)}</div></div>`
  );
}

function responseBodyHtml(message) {
  let html = "";
  if (message.content === "" && !message.error) html = skeletonHtml();
  // ContentRenderer's plain block wrapper, so the spacers between blocks collapse.
  else if (message.content) html = `<div>${renderMarkdown(message.content)}</div>`;
  if (message.error) html += errorHtml(message.error);
  return html;
}

function userMessageHtml(message, readOnly) {
  if (message.editing) {
    return (
      `<div class="user-message" id="message-${message.id}"><div class="msg-body">` +
      `<div class="msg-edit"><div class="msg-edit-scroll"><textarea class="msg-edit-input" data-id="${message.id}">${escapeHtml(message.content)}</textarea></div>` +
      `<div class="msg-edit-actions"><div><button type="button" class="msg-edit-save" data-action="edit-save" data-id="${message.id}">Save</button></div>` +
      `<div class="msg-edit-right"><button type="button" class="msg-edit-cancel" data-action="edit-cancel" data-id="${message.id}">Cancel</button>` +
      `<button type="button" class="msg-edit-send" data-action="edit-send" data-id="${message.id}">Send</button></div></div>` +
      `</div></div></div>`
    );
  }
  const actions = readOnly
    ? ""
    : `<div class="user-actions">` +
      actionButton("edit-user", "pencil", "Edit", message.id, "hover-only") +
      actionButton("copy", "clipboard", "Copy", message.id, "hover-only") +
      `</div>`;
  return (
    `<div class="user-message" id="message-${message.id}"><div class="msg-body">` +
    (message.timestamp
      ? `<div class="user-time"><div class="user-time-text"><span>${escapeHtml(formatDate(message.timestamp))}</span></div></div>`
      : "") +
    `<div class="chat-user markdown-prose"><div class="user-bubble-row"><div class="user-bubble-align">` +
    `<div class="user-bubble">${renderMarkdown(message.content)}</div></div></div>` +
    actions +
    `</div></div></div>`
  );
}

function responseMessageHtml(message, { readOnly, isLast }) {
  const visibility = isLast ? "" : "hover-only";
  let actions = "";
  if (!readOnly && message.done) {
    actions =
      `<div class="response-actions">` +
      actionButton("copy", "clipboard", "Copy", message.id, visibility) +
      (isLast ? actionButton("regenerate", "arrowPath", "Regenerate", message.id) : "") +
      `</div>`;
  }
  return (
    `<div class="response-message" id="message-${message.id}">` +
    `<div class="response-avatar"><img src="${LOGO_URL}" class="size-8" alt="" draggable="false"></div>` +
    `<div class="response-main">` +
    `<div class="msg-name"><span class="msg-name-text">${escapeHtml(displayName(message.model) || "Assistant")}</span>` +
    (message.timestamp
      ? `<div class="msg-name-time"><span>${escapeHtml(formatDate(message.timestamp))}</span></div>`
      : "") +
    `</div>` +
    `<div><div class="chat-assistant markdown-prose"><div>` +
    `<div class="response-content" data-id="${message.id}">${responseBodyHtml(message)}</div>` +
    `</div></div>${actions}</div>` +
    `</div></div>`
  );
}

function messageHtml(chat, message, idx, readOnly) {
  const isLast = idx === chat.messages.length - 1;
  const inner =
    message.role === "user"
      ? userMessageHtml(message, readOnly)
      : responseMessageHtml(message, { readOnly, isLast });
  return `<li class="message" data-id="${message.id}">${inner}</li>`;
}

function messagesListHtml(chat, readOnly) {
  return chat.messages.map((m, idx) => messageHtml(chat, m, idx, readOnly)).join("");
}

/** Read-only conversation, for the search modal's preview pane. */
export function renderReadOnlyMessages(chat) {
  return (
    `<div class="messages messages-preview"><div class="messages-list"><section>` +
    `<ul role="log">${messagesListHtml(chat, true)}</ul></section></div></div>`
  );
}

function conversationHtml(chat) {
  return (
    `<div id="messages-container" class="messages-container scrollbar-hidden"><div class="messages-inner">` +
    `<div class="messages"><div class="messages-list"><section aria-labelledby="chat-conversation">` +
    `<h2 class="sr-only" id="chat-conversation">Chat Conversation</h2>` +
    `<ul role="log" aria-live="polite" aria-relevant="additions" aria-atomic="false">${messagesListHtml(chat, false)}</ul>` +
    `</section><div class="messages-end"></div></div></div>` +
    `</div></div>` +
    `<div class="input-dock">${messageInputHtml("Send a Message")}</div>`
  );
}

// --- Controls panel (ChatControls.svelte, Controls.svelte, AdvancedParams.svelte) ---

function paramHtml(key) {
  const def = PARAMS[key];
  const value = currentParams()[key];
  const custom = value !== undefined && value !== null;
  const controls = custom
    ? `<div class="param-controls"><div class="param-range-wrap">` +
      `<input type="range" class="param-range" data-param="${key}" min="${def.min}" max="${def.max}" step="${def.step}" value="${value}"></div>` +
      `<div><input type="number" class="param-number" data-param="${key}" min="${def.min}" step="${def.numberStep}" value="${value}"></div></div>`
    : "";
  return (
    `<div class="param"><div class="param-head" data-tooltip="${escapeHtml(def.tooltip)}" data-placement="top-start">` +
    `<div class="param-name">${def.label}</div>` +
    `<button type="button" class="param-toggle" data-action="toggle-param" data-param="${key}"><span>${custom ? "Custom" : "Default"}</span></button>` +
    `</div>${controls}</div>`
  );
}

function collapsibleHtml(key, title, content, open) {
  return (
    `<div class="collapsible${open ? " open" : ""}" data-collapsible="${key}">` +
    `<div class="collapsible-button" data-action="toggle-collapsible"><div class="collapsible-head">` +
    `<div>${title}</div><div class="collapsible-chevron">${icon(open ? "chevronUp" : "chevronDown", "size-3.5", 3.5)}</div>` +
    `</div></div><div class="collapsible-content"${open ? "" : " hidden"}>${content}</div></div>`
  );
}

const openSections = { system: true, params: true };

function controlsHtml() {
  const system = escapeHtml(currentParams().system ?? "");
  return (
    `<div class="controls">` +
    `<div class="controls-header"><div class="controls-title">Chat Controls</div>` +
    `<button type="button" class="controls-close" data-action="close-controls" aria-label="Close">${icon("xMark", "size-3.5", 2)}</button></div>` +
    `<div class="controls-body">` +
    collapsibleHtml(
      "system",
      "System Prompt",
      `<textarea class="controls-system" rows="4" placeholder="Enter system prompt">${system}</textarea>`,
      openSections.system
    ) +
    `<hr class="controls-hr">` +
    collapsibleHtml(
      "params",
      "Advanced Params",
      `<div class="params-wrap"><div class="params">${Object.keys(PARAMS).map(paramHtml).join("")}</div></div>`,
      openSections.params
    ) +
    `</div></div>`
  );
}

export function renderControls() {
  const pane = document.querySelector("#chat-container .controls-pane");
  if (!pane) return;
  const resizer = document.querySelector("#chat-container .controls-resizer");
  pane.hidden = !state.showControls;
  if (resizer) resizer.hidden = !state.showControls || isMobile();
  document.querySelector("#chat-container .controls-backdrop").hidden = !(state.showControls && isMobile());
  if (state.showControls) pane.querySelector(".controls-container").innerHTML = controlsHtml();
}

/** Persist a Controls change without bumping the chat to the top of the list. */
function saveParams() {
  if (state.current && !state.current.temporary && state.chats.includes(state.current)) saveChats();
}

// --- View rendering ---

export function renderChatView() {
  const container = document.getElementById("chat-container");
  container.innerHTML =
    `<div class="chat-view"><div class="pane-group">` +
    `<div class="pane-main"><nav id="navbar" class="navbar"></nav><div id="chat-content" class="chat-content"></div></div>` +
    `<div class="controls-resizer" hidden></div>` +
    `<div class="controls-backdrop" hidden data-action="close-controls"></div>` +
    `<div class="controls-pane" hidden><div class="controls-container scrollbar-hidden"></div></div>` +
    `</div></div>`;
  renderNavbar();
  renderContent();
  renderControls();
}

export function renderNavbar() {
  const nav = document.getElementById("navbar");
  if (nav) nav.innerHTML = navbarHtml();
}

export function renderContent() {
  const content = document.getElementById("chat-content");
  if (!content) return;
  const chat = state.current;
  const home = !chat || chat.messages.length === 0;
  content.innerHTML = home ? homeHtml() : conversationHtml(chat);
  const textarea = content.querySelector("#chat-input");
  textarea.value = state.prompt;
  resizeInput(textarea);
  if (home) filterSuggestions();
  else {
    autoScroll = true;
    scrollToBottom();
  }
  if (!isMobile()) textarea.focus();
}

function scrollToBottom() {
  const container = document.getElementById("messages-container");
  if (container) container.scrollTop = container.scrollHeight;
}

function updateScrollButton() {
  const wrap = document.querySelector("#chat-content .mi-scroll-button-wrap");
  if (wrap) wrap.hidden = autoScroll || !state.current?.messages.length;
}

/** Re-render one message, e.g. when a reply finishes and gains its buttons. */
function rerenderMessage(chat, messageId) {
  if (state.current !== chat) return;
  const el = document.querySelector(`#chat-content li.message[data-id="${messageId}"]`);
  const idx = chat.messages.findIndex((m) => m.id === messageId);
  if (!el || idx < 0) return;
  el.outerHTML = messageHtml(chat, chat.messages[idx], idx, false);
}

function rerenderMessages(chat) {
  if (state.current !== chat) return;
  const list = document.querySelector("#chat-content ul[role=log]");
  if (!list) return renderContent();
  list.innerHTML = messagesListHtml(chat, false);
}

/** Streaming updates, at most once per animation frame per message. */
function queueContentUpdate(chat, message) {
  const key = message.id;
  if (renderQueued.has(key)) return;
  renderQueued.add(key);
  requestAnimationFrame(() => {
    renderQueued.delete(key);
    if (state.current !== chat) return;
    const el = document.querySelector(`#chat-content .response-content[data-id="${message.id}"]`);
    if (el) el.innerHTML = responseBodyHtml(message);
    if (autoScroll) scrollToBottom();
  });
}

// --- Sending and streaming ---

function buildMessages(chat, upto) {
  const system = chat.params?.system?.trim();
  const history = chat.messages
    .slice(0, chat.messages.indexOf(upto))
    .filter((m) => !m.error || m.content)
    .map((m) => ({ role: m.role, content: m.content }));
  return system ? [{ role: "system", content: system }, ...history] : history;
}

async function askApiKey() {
  const key = await confirmDialog({
    title: "API key required",
    message: "This Huddle server requires an API key. It is kept in this browser only.",
    input: true,
    inputType: "password",
    inputPlaceholder: "API key",
    confirmLabel: "Save",
  });
  if (key && key.trim()) {
    setApiKey(key.trim());
    return true;
  }
  return false;
}

async function readStream(response, chat, message, signal) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done || signal.aborted) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const data = trimmed.slice(5).trim();
      if (data === "[DONE]") return;
      let chunk;
      try {
        chunk = JSON.parse(data);
      } catch {
        continue;
      }
      if (chunk.error) {
        message.error = chunk.error.message ?? JSON.stringify(chunk.error);
        continue;
      }
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) {
        message.content += delta;
        queueContentUpdate(chat, message);
      }
    }
  }
}

async function generate(chat, message) {
  const controller = new AbortController();
  state.generating = { chatId: chat.id, messageId: message.id, controller };
  updateSendButton();
  const params = chat.params ?? {};
  const body = { model: message.model, stream: true, messages: buildMessages(chat, message) };
  if (params.temperature !== undefined && params.temperature !== null) body.temperature = Number(params.temperature);
  if (params.max_tokens !== undefined && params.max_tokens !== null) body.max_tokens = Number(params.max_tokens);

  try {
    const send = () =>
      api("/v1/chat/completions", { method: "POST", json: body, signal: controller.signal });
    let response = await send();
    if (response.status === 401 && (await askApiKey())) response = await send();
    if (!response.ok) {
      message.error = await errorDetail(response);
    } else {
      await readStream(response, chat, message, controller.signal);
    }
  } catch (error) {
    if (error.name !== "AbortError") message.error = error.message || "Network error";
  } finally {
    message.done = true;
    if (state.generating?.messageId === message.id) state.generating = null;
    touchChat(chat);
    rerenderMessage(chat, message.id);
    if (state.current === chat) updateSendButton();
    if (autoScroll) scrollToBottom();
  }
}

function assistantFor() {
  return {
    id: newId(),
    role: "assistant",
    content: "",
    model: state.loaded,
    timestamp: nowSeconds(),
    done: false,
  };
}

function canSend() {
  if (state.switching) {
    toast("Please wait until the model finishes loading.", "error");
    return false;
  }
  if (!state.loaded) {
    toast("Model not selected", "error");
    return false;
  }
  if (isGenerating(state.current)) return false;
  return true;
}

async function submitPrompt(text) {
  const prompt = text.trim();
  if (!prompt || !canSend()) return;

  let chat = state.current;
  const fresh = !chat;
  if (fresh) {
    chat = {
      id: newId(),
      title: prompt.slice(0, 100),
      created_at: nowSeconds(),
      updated_at: nowSeconds(),
      params: draftParams,
      messages: [],
      temporary: state.temporary,
    };
    draftParams = {};
    state.current = chat;
  }
  chat.messages.push({ id: newId(), role: "user", content: prompt, timestamp: nowSeconds() });
  const reply = assistantFor();
  chat.messages.push(reply);
  state.prompt = "";
  touchChat(chat);

  if (fresh && !chat.temporary) {
    // Open WebUI's replaceState to /c/<id>; no hashchange fires, so the view stays.
    history.replaceState(null, "", `#/c/${chat.id}`);
  }
  if (chat.messages.length === 2) renderChatView();
  else {
    rerenderMessages(chat);
    const textarea = document.getElementById("chat-input");
    textarea.value = "";
    resizeInput(textarea);
  }
  autoScroll = true;
  scrollToBottom();
  await generate(chat, reply);
}

function regenerate(chat) {
  if (!canSend()) return;
  const last = chat.messages.at(-1);
  if (last?.role === "assistant") chat.messages.pop();
  const reply = assistantFor();
  chat.messages.push(reply);
  rerenderMessages(chat);
  autoScroll = true;
  scrollToBottom();
  generate(chat, reply);
}

// --- Events ---

function onClick(event) {
  const target = event.target.closest("[data-action]");
  const suggestion = event.target.closest(".sg-item");
  if (suggestion) {
    const textarea = document.getElementById("chat-input");
    textarea.value = suggestion.dataset.prompt;
    state.prompt = textarea.value;
    resizeInput(textarea);
    textarea.focus();
    updateSendButton();
    filterSuggestions();
    return;
  }
  if (!target) return;
  const chat = state.current;
  const action = target.dataset.action;
  const message = chat?.messages.find((m) => m.id === target.dataset.id);

  if (action === "stop") {
    state.generating?.controller.abort();
  } else if (action === "scroll-bottom") {
    autoScroll = true;
    scrollToBottom();
    updateScrollButton();
  } else if (action === "copy" && message) {
    copyText(message.content);
  } else if (action === "regenerate" && chat) {
    regenerate(chat);
  } else if (action === "edit-user" && message) {
    message.editing = true;
    rerenderMessage(chat, message.id);
    const input = document.querySelector(`.msg-edit-input[data-id="${message.id}"]`);
    if (input) {
      resizeInput(input);
      input.focus();
    }
  } else if (action === "edit-cancel" && message) {
    message.editing = false;
    rerenderMessage(chat, message.id);
  } else if ((action === "edit-save" || action === "edit-send") && message) {
    const input = document.querySelector(`.msg-edit-input[data-id="${message.id}"]`);
    const text = input?.value.trim() ?? "";
    if (!text) return;
    message.content = text;
    message.editing = false;
    if (action === "edit-save") {
      touchChat(chat);
      rerenderMessage(chat, message.id);
      return;
    }
    if (!canSend()) return rerenderMessage(chat, message.id);
    // Sending an edit replaces everything after it with a fresh reply.
    chat.messages.splice(chat.messages.indexOf(message) + 1);
    const reply = assistantFor();
    chat.messages.push(reply);
    touchChat(chat);
    rerenderMessages(chat);
    generate(chat, reply);
  } else if (action === "close-controls") {
    state.showControls = false;
    renderControls();
  } else if (action === "toggle-collapsible") {
    const section = target.closest(".collapsible");
    openSections[section.dataset.collapsible] = !section.classList.contains("open");
    renderControls();
  } else if (action === "toggle-param") {
    const key = target.dataset.param;
    const params = state.current ? (state.current.params ??= {}) : draftParams;
    params[key] = params[key] === undefined || params[key] === null ? PARAMS[key].initial : null;
    saveParams();
    renderControls();
  }
}

function onInput(event) {
  const el = event.target;
  if (el.id === "chat-input") {
    state.prompt = el.value;
    resizeInput(el);
    updateSendButton();
    filterSuggestions();
  } else if (el.classList.contains("msg-edit-input")) {
    resizeInput(el);
  } else if (el.classList.contains("controls-system")) {
    const params = state.current ? (state.current.params ??= {}) : draftParams;
    params.system = el.value;
    saveParams();
  } else if (el.classList.contains("param-range") || el.classList.contains("param-number")) {
    const params = state.current ? (state.current.params ??= {}) : draftParams;
    const key = el.dataset.param;
    params[key] = el.value === "" ? PARAMS[key].initial : Number(el.value);
    const other = el.closest(".param-controls").querySelector(
      el.classList.contains("param-range") ? ".param-number" : ".param-range"
    );
    other.value = params[key];
    saveParams();
  }
}

function onKeydown(event) {
  const el = event.target;
  if (el.id === "chat-input" && event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    // Enter sends on desktop; on touch devices it inserts a newline, as in Open WebUI.
    if (isMobile() && ("ontouchstart" in window || navigator.maxTouchPoints > 0)) return;
    event.preventDefault();
    submitPrompt(el.value);
  } else if (el.classList.contains("msg-edit-input")) {
    if (event.key === "Escape") {
      el.closest(".msg-edit").querySelector("[data-action=edit-cancel]").click();
    } else if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      el.closest(".msg-edit").querySelector("[data-action=edit-send]").click();
    }
  }
}

function onSubmit(event) {
  if (!event.target.classList.contains("mi-form")) return;
  event.preventDefault();
  submitPrompt(document.getElementById("chat-input").value);
}

function onScroll(event) {
  const el = event.target;
  if (el.id !== "messages-container") return;
  autoScroll = el.scrollHeight - el.scrollTop <= el.clientHeight + 5;
  updateScrollButton();
}

/** Start a new chat: Open WebUI's initNewChat. Keeps the temporary-chat toggle. */
export function newChat() {
  state.current = null;
  state.prompt = "";
  draftParams = {};
  // Pushed, like Open WebUI's goto('/'), so Back returns to the previous chat.
  if (location.hash !== "#/" && location.hash !== "") history.pushState(null, "", "#/");
}

export function initChat() {
  const container = document.getElementById("chat-container");
  container.addEventListener("click", onClick);
  container.addEventListener("input", onInput);
  container.addEventListener("keydown", onKeydown);
  container.addEventListener("submit", onSubmit);
  container.addEventListener("scroll", onScroll, true);
  bindCodeBlockActions(container, (text) => copyText(text, false));
}

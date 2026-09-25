// Sidebar (expanded and collapsed rail), chat list, chat menu and search modal.
// Markup follows Open WebUI v0.6.30: layout/Sidebar.svelte, common/Folder.svelte,
// layout/Sidebar/ChatItem.svelte, ChatMenu.svelte and layout/SearchModal.svelte.

import { renderReadOnlyMessages } from "./chat.js";
import { icon } from "./icons.js";
import { escapeHtml } from "./markdown.js";
import {
  deleteChat,
  emit,
  isMobile,
  renameChat,
  setShowSidebar,
  state,
  timeRange,
} from "./store.js";
import { closeMenu, confirmDialog, initialsImage, LOGO_URL, openMenu } from "./ui.js";

let renamingId = null;

const NAV_ITEMS = [
  { action: "new-chat", href: "#/", label: "New Chat", icon: () => icon("pencilSquare", "size-4.5", 2) },
  { action: "search", label: "Search", icon: () => icon("search", "size-4.5", 2) },
  { action: "workspace", href: "#/workspace", label: "Workspace", icon: () => icon("workspace", "size-4.5", 2) },
  { action: "cluster", href: "#/cluster", label: "Cluster", icon: () => icon("serverStack", "size-4.5", 2) },
];

function chatItem(chat) {
  const active = state.current?.id === chat.id;
  const title = escapeHtml(chat.title || "New Chat");
  const body =
    renamingId === chat.id
      ? `<div class="chat-item-link active"><input class="chat-title-input" value="${title}" data-id="${chat.id}"></div>`
      : `<a class="chat-item-link${active ? " active" : ""}" href="#/c/${chat.id}" draggable="false">` +
        `<div class="chat-item-title-wrap"><div dir="auto" class="chat-item-title">${title}</div></div></a>`;
  const menu =
    renamingId === chat.id
      ? ""
      : `<div class="chat-item-menu${active ? " active" : ""}"><div class="chat-item-menu-inner">` +
        `<button type="button" class="chat-menu-button" aria-label="Chat Menu" data-action="chat-menu" data-id="${chat.id}">` +
        `${icon("ellipsisDots", "size-4")}</button></div></div>`;
  return `<div class="chat-item" data-chat-id="${chat.id}">${body}${menu}</div>`;
}

function chatList() {
  let previous = null;
  return state.chats
    .map((chat, idx) => {
      const range = timeRange(chat.updated_at);
      const label =
        range !== previous
          ? `<div class="time-range${idx === 0 ? "" : " time-range-gap"}">${escapeHtml(range)}</div>`
          : "";
      previous = range;
      return label + chatItem(chat);
    })
    .join("");
}

function sidebarHtml() {
  const nav = NAV_ITEMS.map(
    (item) =>
      `<div class="sb-nav-wrap">` +
      `<a class="sb-nav-item" ${item.href ? `href="${item.href}"` : 'href="#" role="button"'} data-action="${item.action}" draggable="false" aria-label="${item.label}">` +
      `<div class="sb-nav-icon">${item.icon()}</div>` +
      `<div class="sb-nav-label"><div>${item.label}</div></div></a></div>`
  ).join("");
  const avatar = initialsImage(state.node || "Huddle");
  return (
    `<div class="sb-inner scrollbar-hidden">` +
    `<div class="sb-header">` +
    `<a class="sb-logo" href="#/" data-action="new-chat" draggable="false"><img src="${LOGO_URL}" class="sb-logo-img" alt=""></a>` +
    `<a class="sb-name" href="#/" data-action="new-chat" draggable="false"><div>Huddle</div></a>` +
    `<button type="button" class="sb-toggle" data-action="toggle-sidebar" data-tooltip="Close Sidebar" data-placement="bottom" aria-label="Close Sidebar">` +
    `<div class="sb-toggle-icon">${icon("sidebar", "size-5")}</div></button>` +
    `</div>` +
    `<div class="sb-nav">${nav}</div>` +
    `<div class="sb-lists">` +
    `<div class="sb-folder">` +
    `<div class="sb-folder-button"><button type="button" class="sb-folder-toggle" data-action="toggle-chats">` +
    `<div class="sb-folder-name">Chats</div></button></div>` +
    `<div class="sb-folder-content"${state.chatsOpen ? "" : " hidden"}>` +
    `<div class="sb-chat-scroll scrollbar-hidden"><div class="sb-chat-list">${chatList()}</div></div>` +
    `</div></div></div>` +
    `<div class="sb-footer"><div class="sb-footer-inner">` +
    `<div class="sb-user"><div class="sb-user-avatar"><img src="${avatar}" alt="" draggable="false"></div>` +
    `<div class="sb-user-name">${escapeHtml(state.node || "")}</div></div>` +
    `</div></div>` +
    `</div>`
  );
}

function railHtml() {
  const cell = (inner, attrs, tip) =>
    `<div><a class="rail-button" ${attrs} data-tooltip="${tip}" data-placement="right" draggable="false" aria-label="${tip}">` +
    `<div class="rail-cell">${inner}</div></a></div>`;
  return (
    `<div class="rail-top" data-action="toggle-sidebar">` +
    `<div class="rail-logo-wrap"><button type="button" class="rail-button rail-open" data-action="toggle-sidebar" data-tooltip="Open Sidebar" data-placement="right" aria-label="Open Sidebar">` +
    `<div class="rail-cell"><img src="${LOGO_URL}" class="rail-logo" alt="">${icon("sidebar", "size-5 rail-sidebar-icon")}</div></button></div>` +
    `<div>` +
    cell(icon("pencilSquare", "size-4.5"), 'href="#/" data-action="new-chat"', "New Chat") +
    cell(icon("search", "size-4.5"), 'href="#" data-action="search"', "Search") +
    cell(icon("workspace", "size-4.5"), 'href="#/workspace" data-action="workspace"', "Workspace") +
    cell(icon("serverStack", "size-4.5"), 'href="#/cluster" data-action="cluster"', "Cluster") +
    `</div></div>` +
    `<div><div class="rail-user"><div class="rail-button"><div class="rail-cell">` +
    `<img src="${initialsImage(state.node || "Huddle")}" class="rail-avatar" alt=""></div></div></div></div>`
  );
}

export function renderSidebar() {
  const sidebar = document.getElementById("sidebar");
  const rail = document.getElementById("sidebar-rail");
  const backdrop = document.getElementById("sidebar-backdrop");
  const mobile = isMobile();
  sidebar.hidden = !state.showSidebar;
  rail.hidden = state.showSidebar || mobile;
  backdrop.hidden = !(state.showSidebar && mobile);
  document.getElementById("chat-container").classList.toggle("with-sidebar", state.showSidebar);
  if (state.showSidebar) {
    const scroller = sidebar.querySelector(".sb-chat-scroll");
    const scrollTop = scroller?.scrollTop ?? 0;
    sidebar.innerHTML = sidebarHtml();
    const next = sidebar.querySelector(".sb-chat-scroll");
    if (next) next.scrollTop = scrollTop;
    const input = sidebar.querySelector(".chat-title-input");
    if (input) {
      input.focus();
      input.select();
    }
  } else {
    rail.innerHTML = railHtml();
  }
}

function chatMenu(button) {
  const id = button.dataset.id;
  const menu = document.createElement("div");
  menu.className = "dropdown chat-dropdown";
  menu.innerHTML =
    `<button type="button" class="dropdown-item" data-menu="rename">${icon("pencil", "size-4")}<div>Rename</div></button>` +
    `<hr class="dropdown-hr">` +
    `<button type="button" class="dropdown-item" data-menu="delete">${icon("garbageBin", "size-4")}<div>Delete</div></button>`;
  menu.addEventListener("click", async (event) => {
    const item = event.target.closest("[data-menu]");
    if (!item) return;
    closeMenu();
    if (item.dataset.menu === "rename") {
      renamingId = id;
      renderSidebar();
    } else {
      const chat = state.chats.find((c) => c.id === id);
      const ok = await confirmDialog({
        title: "Delete chat?",
        html: `This will delete <span class="font-semibold">${escapeHtml(chat?.title ?? "")}</span>.`,
      });
      if (!ok) return;
      deleteChat(id);
      if (state.current?.id === id) emit("new-chat");
    }
  });
  button.closest(".chat-item").classList.add("menu-open");
  openMenu(button, menu, {
    align: "start",
    sideOffset: -2,
    onClose: () => button.closest(".chat-item")?.classList.remove("menu-open"),
  });
}

function finishRename(input, save) {
  if (renamingId === null) return;
  const id = renamingId;
  renamingId = null;
  if (save) renameChat(id, input.value);
  renderSidebar();
}

function onSidebarClick(event) {
  const target = event.target.closest("[data-action]");
  if (!target) {
    // Following a chat link on mobile closes the overlay, as Open WebUI does.
    if (event.target.closest(".chat-item-link") && isMobile()) setShowSidebar(false);
    return;
  }
  const action = target.dataset.action;
  if (action === "toggle-sidebar") {
    event.preventDefault();
    event.stopPropagation();
    setShowSidebar(!state.showSidebar);
  } else if (action === "new-chat") {
    event.preventDefault();
    emit("new-chat");
    if (isMobile()) setShowSidebar(false);
  } else if (action === "search") {
    event.preventDefault();
    event.stopPropagation();
    openSearch();
  } else if (action === "workspace" || action === "cluster") {
    if (isMobile()) setShowSidebar(false);
  } else if (action === "toggle-chats") {
    state.chatsOpen = !state.chatsOpen;
    renderSidebar();
  } else if (action === "chat-menu") {
    event.preventDefault();
    event.stopPropagation();
    chatMenu(target);
  }
}

// --- SearchModal.svelte ---

function calendar(seconds) {
  // dayjs(...).calendar() with its default English formats.
  const date = new Date(seconds * 1000);
  const time = date.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  const startOf = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((startOf(date) - startOf(new Date())) / 86400000);
  const weekday = date.toLocaleDateString("en-US", { weekday: "long" });
  if (days === 0) return `Today at ${time}`;
  if (days === -1) return `Yesterday at ${time}`;
  if (days === 1) return `Tomorrow at ${time}`;
  if (days < -1 && days >= -6) return `Last ${weekday} at ${time}`;
  if (days > 1 && days < 7) return `${weekday} at ${time}`;
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(date.getMonth() + 1)}/${pad(date.getDate())}/${date.getFullYear()}`;
}

function matches(chat, query) {
  if (!query) return true;
  const q = query.toLowerCase();
  return (
    (chat.title ?? "").toLowerCase().includes(q) ||
    chat.messages.some((m) => (m.content ?? "").toLowerCase().includes(q))
  );
}

export function openSearch() {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.innerHTML =
    `<div class="modal-panel modal-xl fly-in" role="dialog" aria-modal="true"><div class="search-modal">` +
    `<div class="search-head"><div class="search-container"><div class="search-field">` +
    `<div class="search-icon">${icon("search", "size-4")}</div>` +
    `<input class="search-input" placeholder="Search" autocomplete="off">` +
    `<div class="search-clear" hidden><button type="button" aria-label="Clear">${icon("xMark", "size-3", 2)}</button></div>` +
    `</div></div></div>` +
    `<div class="search-body"><div class="search-results scrollbar-hidden"></div>` +
    `<div class="search-preview scrollbar-hidden"><div class="search-preview-empty">Select a conversation to preview</div></div>` +
    `</div></div></div>`;
  const input = overlay.querySelector(".search-input");
  const results = overlay.querySelector(".search-results");
  const preview = overlay.querySelector(".search-preview");
  const clear = overlay.querySelector(".search-clear");
  let list = [];
  let selected = null;

  const close = () => {
    overlay.remove();
    document.removeEventListener("keydown", onKey, true);
  };
  const open = (chat) => {
    close();
    location.hash = `#/c/${chat.id}`;
    if (isMobile()) setShowSidebar(false);
  };
  const select = (idx) => {
    selected = idx;
    results.querySelectorAll(".search-item").forEach((el, i) => el.classList.toggle("selected", i === idx));
    const chat = list[idx];
    preview.innerHTML = chat
      ? `<div class="search-preview-messages">${renderReadOnlyMessages(chat)}</div>`
      : `<div class="search-preview-empty">Select a conversation to preview</div>`;
  };
  const render = () => {
    const query = input.value.trim();
    clear.hidden = !input.value;
    list = state.chats.filter((chat) => matches(chat, query));
    let previous = null;
    results.innerHTML =
      (list.length === 0 ? '<div class="search-empty">No results found</div>' : "") +
      list
        .map((chat, idx) => {
          const range = timeRange(chat.updated_at);
          const label =
            range !== previous
              ? `<div class="search-range${idx === 0 ? "" : " search-range-gap"}">${escapeHtml(range)}</div>`
              : "";
          previous = range;
          return (
            label +
            `<a class="search-item" href="#/c/${chat.id}" data-idx="${idx}" draggable="false">` +
            `<div class="search-item-title"><div>${escapeHtml(chat.title || "New Chat")}</div></div>` +
            `<div class="search-item-date">${escapeHtml(calendar(chat.updated_at))}</div></a>`
          );
        })
        .join("");
    selected = null;
    preview.innerHTML = `<div class="search-preview-empty">Select a conversation to preview</div>`;
  };
  const onKey = (event) => {
    if (event.key === "Escape") return close();
    if (event.key === "ArrowDown") {
      event.preventDefault();
      select(Math.min((selected ?? -1) + 1, list.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      select(Math.max((selected ?? 1) - 1, 0));
    } else if (event.key === "Enter" && list.length > 0) {
      event.preventDefault();
      open(list[selected ?? 0]);
    }
    results.querySelector(".search-item.selected")?.scrollIntoView({ block: "center" });
  };

  input.addEventListener("input", render);
  input.addEventListener("focus", () => select(null));
  clear.addEventListener("click", () => {
    input.value = "";
    render();
    input.focus();
  });
  results.addEventListener("mouseover", (event) => {
    const item = event.target.closest(".search-item");
    if (item && Number(item.dataset.idx) !== selected) select(Number(item.dataset.idx));
  });
  results.addEventListener("click", (event) => {
    const item = event.target.closest(".search-item");
    if (!item) return;
    event.preventDefault();
    open(list[Number(item.dataset.idx)]);
  });
  overlay.addEventListener("mousedown", (event) => {
    if (event.target === overlay) close();
  });
  document.addEventListener("keydown", onKey, true);
  document.body.append(overlay);
  render();
  input.focus();
}

export function initSidebar() {
  const sidebar = document.getElementById("sidebar");
  const rail = document.getElementById("sidebar-rail");
  sidebar.addEventListener("click", onSidebarClick);
  rail.addEventListener("click", onSidebarClick);
  document.getElementById("sidebar-backdrop").addEventListener("mousedown", () => setShowSidebar(false));

  sidebar.addEventListener("keydown", (event) => {
    if (!event.target.classList.contains("chat-title-input")) return;
    if (event.key === "Enter") finishRename(event.target, true);
    if (event.key === "Escape") finishRename(event.target, false);
  });
  sidebar.addEventListener("focusout", (event) => {
    if (event.target.classList.contains("chat-title-input")) finishRename(event.target, true);
  });
  // Double-click a chat title to rename it, as in Open WebUI.
  sidebar.addEventListener("dblclick", (event) => {
    const link = event.target.closest(".chat-item-link");
    if (!link) return;
    event.preventDefault();
    renamingId = link.closest(".chat-item").dataset.chatId;
    renderSidebar();
  });
}

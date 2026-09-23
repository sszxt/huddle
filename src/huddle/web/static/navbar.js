// Chat navbar: model selector, temporary chat, controls, avatar.
// Markup follows Open WebUI v0.6.30 chat/Navbar.svelte, chat/ModelSelector.svelte,
// ModelSelector/Selector.svelte and ModelSelector/ModelItem.svelte. The "+ add
// model" and "Set as default" links are left out: llama.cpp holds one model per
// process, and the browser cannot change the cluster's startup default.

import { switchModel } from "./cluster.js";
import { icon } from "./icons.js";
import { escapeHtml } from "./markdown.js";
import { displayName, emit, isMobile, setShowSidebar, state, touchChat } from "./store.js";
import { closeMenu, initialsImage, LOGO_URL, openMenu } from "./ui.js";

function selectedModel() {
  return state.switching ?? state.loaded;
}

export function navbarHtml() {
  const chat = state.current;
  const mobileToggle =
    isMobile() && !state.showSidebar
      ? `<div class="nav-sidebar-toggle"><button type="button" class="nav-sidebar-button" data-action="open-sidebar" aria-label="Open Sidebar">` +
        `<div class="nav-sidebar-icon">${icon("sidebar", "size-5")}</div></button></div>`
      : "";
  const label = selectedModel() ? escapeHtml(displayName(selectedModel())) : "Select a model";

  let temporaryButton = "";
  if (!chat || chat.messages.length === 0) {
    const on = state.temporary;
    temporaryButton =
      `<button type="button" class="nav-button" data-action="toggle-temporary" data-tooltip="Temporary Chat" aria-label="Temporary Chat">` +
      `<div class="nav-button-icon">${icon(on ? "chatBubbleDottedChecked" : "chatBubbleDotted", "size-4.5", 1.5)}</div></button>`;
  } else if (chat.temporary) {
    temporaryButton =
      `<button type="button" class="nav-button" data-action="save-temporary" data-tooltip="Save Chat" aria-label="Save Chat">` +
      `<div class="nav-button-icon">${icon("chatCheck", "size-4.5", 1.5)}</div></button>`;
  }

  const temporaryLabel = chat?.temporary
    ? `<div class="nav-temporary"><div>Temporary Chat</div></div>`
    : "";

  return (
    `<div class="nav-row"><div class="nav-gradient"></div>` +
    `<div class="nav-inner"><div class="nav-content">` +
    mobileToggle +
    `<div class="nav-left${state.showSidebar ? " with-sidebar" : ""}">` +
    `<div class="model-selector"><div class="model-selector-row"><div class="model-selector-clip"><div class="model-selector-pad">` +
    `<button type="button" class="model-trigger" id="model-selector-0-button" aria-label="Select a model">` +
    `<div class="model-trigger-inner"><span class="model-trigger-label">${label}</span>` +
    `${icon("chevronDown", "model-trigger-chevron", 2.5)}</div></button>` +
    `</div></div></div></div>` +
    `</div>` +
    `<div class="nav-right">` +
    temporaryButton +
    `<button type="button" class="nav-button" data-action="toggle-controls" data-tooltip="Controls" aria-label="Controls">` +
    `<div class="nav-button-icon">${icon("adjustmentsHorizontal", "size-5", 1)}</div></button>` +
    `<div class="nav-user"><div class="nav-user-inner"><span class="sr-only">User menu</span>` +
    `<img src="${initialsImage(state.node || "Huddle")}" class="nav-avatar" alt="" draggable="false"></div></div>` +
    `</div></div></div></div>` +
    temporaryLabel
  );
}

function modelMenu(trigger) {
  const menu = document.createElement("div");
  menu.className = "dropdown model-dropdown";
  menu.innerHTML =
    `<div class="model-search">${icon("search", "size-4", 2.5)}` +
    `<input id="model-search-input" class="model-search-input" placeholder="Search a model" autocomplete="off" aria-label="Search In Models"></div>` +
    `<div class="model-list scrollbar-hidden"></div><div class="model-list-end"></div>`;
  const input = menu.querySelector("input");
  const list = menu.querySelector(".model-list");
  let items = [];
  let highlighted = 0;

  const render = () => {
    const q = input.value.trim().toLowerCase();
    items = state.models.filter((file) => displayName(file).toLowerCase().includes(q));
    const current = selectedModel();
    list.innerHTML =
      items
        .map(
          (file, idx) =>
            `<button type="button" class="model-item${idx === highlighted ? " highlighted" : ""}" data-idx="${idx}" aria-label="${escapeHtml(displayName(file))}">` +
            `<div class="model-item-main"><div class="model-item-row">` +
            `<div class="model-item-image"><img src="${LOGO_URL}" alt="Model"></div>` +
            `<div class="model-item-label"><div>${escapeHtml(displayName(file))}</div></div>` +
            `</div></div>` +
            `<div class="model-item-end">${file === current ? `<div>${icon("check", "size-3")}</div>` : ""}</div>` +
            `</button>`
        )
        .join("") || `<div><div class="model-empty">No results found</div></div>`;
  };
  const choose = (idx) => {
    const file = items[idx];
    closeMenu();
    if (file) switchModel(file);
  };

  input.addEventListener("input", () => {
    highlighted = 0;
    render();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && items.length > 0) {
      event.preventDefault();
      choose(highlighted);
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      highlighted = Math.min(highlighted + 1, items.length - 1);
      render();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      highlighted = Math.max(highlighted - 1, 0);
      render();
    }
    list.querySelector(".highlighted")?.scrollIntoView({ block: "center" });
  });
  list.addEventListener("click", (event) => {
    const item = event.target.closest(".model-item");
    if (item) choose(Number(item.dataset.idx));
  });

  highlighted = Math.max(0, state.models.indexOf(selectedModel()));
  render();
  openMenu(trigger, menu, { align: "start", sideOffset: 2, alignOffset: -1 });
  setTimeout(() => input.focus(), 0);
}

/** Delegated from the chat container, which outlives each navbar render. */
export function bindNavbar(container) {
  container.addEventListener("click", (event) => {
    if (!event.target.closest("#navbar")) return;
    const trigger = event.target.closest(".model-trigger");
    if (trigger) {
      modelMenu(trigger);
      return;
    }
    const target = event.target.closest("[data-action]");
    if (!target) return;
    const action = target.dataset.action;
    if (action === "open-sidebar") {
      setShowSidebar(true);
    } else if (action === "toggle-temporary") {
      state.temporary = !state.temporary;
      emit("new-chat");
    } else if (action === "save-temporary") {
      const chat = state.current;
      if (!chat?.temporary) return;
      chat.temporary = false;
      state.temporary = false;
      touchChat(chat);
      location.hash = `#/c/${chat.id}`;
      emit("chat-view");
    } else if (action === "toggle-controls") {
      state.showControls = !state.showControls;
      emit("controls");
    }
  });
}

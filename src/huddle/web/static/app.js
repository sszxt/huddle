// Huddle web UI: a chat front end over Huddle's own OpenAI-compatible API.
//
// Layout and styling recreate Open WebUI v0.6.30's chat screen
// (github.com/open-webui/open-webui) in plain HTML, CSS and JS, with Huddle's
// own name and logo. No Open WebUI source or branding is included. Plain ES
// modules against same-origin routes: no build step, no framework.

import {
  initChat,
  newChat,
  renderChatView,
  renderControls,
  renderNavbar,
  updateHomeHeading,
} from "./chat.js";
import { startPolling } from "./cluster.js";
import { closeClusterPage, initClusterPage, openClusterPage, renderClusterPage } from "./clusterview.js";
import { bindNavbar } from "./navbar.js";
import { initSidebar, renderSidebar } from "./sidebar.js";
import { isMobile, on, state } from "./store.js";
import { initTooltips } from "./ui.js";
import { initWorkspace, renderWorkspace } from "./workspace.js";

function parseRoute() {
  const path = location.hash.replace(/^#/, "") || "/";
  const chat = path.match(/^\/c\/([^/?#]+)/);
  if (chat) return { name: "chat", id: decodeURIComponent(chat[1]) };
  if (path.startsWith("/workspace")) return { name: "workspace" };
  if (path.startsWith("/cluster")) return { name: "cluster" };
  return { name: "home" };
}

// Pages that replace the chat view, and how to redraw each in place.
const PAGES = { workspace: renderWorkspace, cluster: renderClusterPage };

/** Open WebUI's page title: the chat title (cut at 30 characters) • name. */
function updateTitle() {
  if (state.route.name === "workspace" || state.route.name === "cluster") {
    document.title = `${state.route.name === "workspace" ? "Workspace" : "Cluster"} • Huddle`;
    return;
  }
  const title = state.current && !state.current.temporary ? state.current.title : "";
  document.title = title ? `${title.length > 30 ? `${title.slice(0, 30)}...` : title} • Huddle` : "Huddle";
}

function route() {
  const next = parseRoute();
  const previous = state.route.name;
  state.route = next;
  if (next.name !== "cluster") closeClusterPage();

  if (next.name === "workspace") {
    renderWorkspace();
  } else if (next.name === "cluster") {
    openClusterPage();
  } else if (next.name === "chat") {
    const chat = state.chats.find((c) => c.id === next.id);
    if (!chat) {
      state.route = { name: "home" };
      newChat();
      renderChatView();
    } else if (chat !== state.current || previous in PAGES) {
      state.current = chat;
      state.prompt = "";
      renderChatView();
    }
  } else {
    newChat();
    renderChatView();
  }
  renderSidebar();
  updateTitle();
}

function showNewChat() {
  newChat();
  state.route = { name: "home" };
  closeClusterPage();
  renderChatView();
  renderSidebar();
  updateTitle();
}

function init() {
  initTooltips();
  initSidebar();
  initChat();
  initWorkspace();
  initClusterPage();
  bindNavbar(document.getElementById("chat-container"));

  on("new-chat", showNewChat);
  on("chat-view", () => {
    renderChatView();
    renderSidebar();
    updateTitle();
  });
  on("chats", () => {
    renderSidebar();
    updateTitle();
  });
  on("sidebar", () => {
    renderSidebar();
    if (state.route.name in PAGES) PAGES[state.route.name]();
    else renderNavbar();
  });
  on("models", () => {
    if (state.route.name === "workspace") renderWorkspace();
    else if (!(state.route.name in PAGES)) {
      renderNavbar();
      updateHomeHeading();
    }
  });
  on("node", () => {
    renderSidebar();
    if (!(state.route.name in PAGES)) {
      renderNavbar();
      updateHomeHeading();
    }
  });
  on("controls", renderControls);

  let wasMobile = isMobile();
  window.addEventListener("resize", () => {
    if (isMobile() === wasMobile) return;
    wasMobile = isMobile();
    if (wasMobile) state.showSidebar = false;
    renderSidebar();
    if (state.route.name in PAGES) PAGES[state.route.name]();
    else {
      renderNavbar();
      renderControls();
    }
  });

  window.addEventListener("hashchange", route);
  route();
  startPolling();
}

init();

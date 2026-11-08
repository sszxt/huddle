// Huddle web UI: a chat front end over Huddle's own OpenAI-compatible API.
//
// Layout and styling recreate Open WebUI v0.6.30's chat screen
// (github.com/open-webui/open-webui) in plain HTML, CSS and JS, with Huddle's
// own name and logo. No Open WebUI source or branding is included. Plain ES
// modules against same-origin routes: no build step, no framework.

import { initAdminPage, renderAdminPage, resetAdminPage } from "./admin.js";
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
import { loadServerHistory, offerImport } from "./history.js";
import { bindNavbar } from "./navbar.js";
import { loadSession, renderAuthPage, renderPendingPage, renderUnavailablePage } from "./session.js";
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
  const admin = path.match(/^\/admin(?:\/(\w+))?/);
  if (admin) return { name: "admin", tab: admin[1] ?? "users" };
  return { name: "home" };
}

// Pages that replace the chat view, and how to redraw each in place.
const PAGES = { workspace: renderWorkspace, cluster: renderClusterPage, admin: renderAdminPage };
const PAGE_TITLES = { workspace: "Workspace", cluster: "Cluster", admin: "Admin Panel" };

/** Open WebUI's page title: the chat title (cut at 30 characters) • name. */
function updateTitle() {
  if (state.route.name in PAGE_TITLES) {
    document.title = `${PAGE_TITLES[state.route.name]} • Huddle`;
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

  if (next.name === "admin" && state.user?.role !== "admin") {
    location.hash = "#/";
    return;
  }

  if (next.name === "workspace") {
    renderWorkspace();
  } else if (next.name === "admin") {
    // Fresh each visit: accounts change while the page is elsewhere.
    if (previous !== "admin") resetAdminPage();
    renderAdminPage();
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
  initAdminPage();
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

/** Sign-in first: the chat app starts only for someone allowed to use it. */
async function boot() {
  const session = await loadSession();
  if (session === "signin") return renderAuthPage();
  if (session === "pending") return renderPendingPage();
  if (session === "unavailable") return renderUnavailablePage();
  if (session === "active") await loadServerHistory();
  init();
  if (session === "active") offerImport();
}

boot();

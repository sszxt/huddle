// Admin Panel: this PC's accounts and its sign-up settings.
// After Open WebUI v0.6.30's admin/users (the user list, role, add, edit,
// delete) and admin/settings General ("Enable New Sign Ups", "Default User Role").

import { icon } from "./icons.js";
import { escapeHtml } from "./markdown.js";
import { isMobile, setShowSidebar, state } from "./store.js";
import { api, confirmDialog, errorDetail, formDialog, initialsImage, toast } from "./ui.js";

let users = null;
let settings = null;
let loadError = null;
let loading = false;
let query = "";

const ROLES = [
  { value: "pending", label: "Pending" },
  { value: "user", label: "User" },
  { value: "admin", label: "Admin" },
];

function tab() {
  return state.route.tab === "settings" ? "settings" : "users";
}

// --- Dates: dayjs fromNow() and "LL", as Open WebUI shows them ---

function fromNow(seconds) {
  if (!seconds) return "-";
  const diff = Date.now() / 1000 - seconds;
  const units = [
    [60, "second"],
    [60, "minute"],
    [24, "hour"],
    [30, "day"],
    [12, "month"],
    [Infinity, "year"],
  ];
  let value = diff;
  for (const [size, unit] of units) {
    if (value < size) {
      const n = Math.max(1, Math.round(value));
      return unit === "second" ? "a few seconds ago" : `${n} ${unit}${n === 1 ? "" : "s"} ago`;
    }
    value /= size;
  }
  return "-";
}

function longDate(seconds) {
  return new Date(seconds * 1000).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

// --- Rendering ---

function usersHtml() {
  if (loadError) return `<div class="ws-error">${escapeHtml(loadError)}</div>`;
  if (users === null) return `<div class="ws-empty">Loading…</div>`;
  const q = query.trim().toLowerCase();
  const shown = users.filter(
    (u) => !q || u.name.toLowerCase().includes(q) || u.email.toLowerCase().includes(q)
  );
  const rows = shown
    .map((u) => {
      const me = u.id === state.user?.id;
      const roles = ROLES.map(
        (r) => `<option value="${r.value}"${r.value === u.role ? " selected" : ""}>${r.label}</option>`
      ).join("");
      return (
        `<tr class="ad-row">` +
        `<td><select class="ad-role ad-role-${u.role}" data-id="${u.id}" aria-label="Role">${roles}</select></td>` +
        `<td><div class="ad-user"><img src="${initialsImage(u.name)}" alt="" class="ad-avatar">` +
        `<span class="ad-name">${escapeHtml(u.name)}</span>${me ? '<span class="ad-you">You</span>' : ""}</div></td>` +
        `<td class="ad-muted">${escapeHtml(u.email)}</td>` +
        `<td class="ad-muted">${escapeHtml(fromNow(u.last_active_at))}</td>` +
        `<td class="ad-muted">${escapeHtml(longDate(u.created_at))}</td>` +
        `<td><div class="ad-actions">` +
        `<button type="button" class="ad-icon-button" data-action="edit-user" data-id="${u.id}" data-tooltip="Edit User">${icon("pencil", "size-4")}</button>` +
        `<button type="button" class="ad-icon-button" data-action="delete-user" data-id="${u.id}" data-tooltip="Delete User">${icon("garbageBin", "size-4")}</button>` +
        `</div></td></tr>`
      );
    })
    .join("");
  const pending = users.filter((u) => u.role === "pending").length;
  return (
    `<div class="ad-header"><div class="ws-title">Users<div class="ws-title-divider"></div>` +
    `<span class="ws-count">${users.length}</span></div>` +
    `<div class="ad-tools"><div class="ad-search">${icon("search", "size-4")}` +
    `<input class="ad-search-input" placeholder="Search" value="${escapeHtml(query)}"></div>` +
    `<button type="button" class="ad-add" data-action="add-user" data-tooltip="Add User">${icon("plus", "size-4", 2)}</button></div></div>` +
    (pending
      ? `<div class="ad-banner">${pending} account${pending === 1 ? " is" : "s are"} waiting for approval. ` +
        `Set the role to User to let ${pending === 1 ? "it" : "them"} in.</div>`
      : "") +
    `<div class="ad-table-wrap"><table class="ad-table"><thead><tr>` +
    `<th>Role</th><th>Name</th><th>Email</th><th>Last Active</th><th>Created at</th><th></th>` +
    `</tr></thead><tbody>${rows || '<tr><td colspan="6" class="ws-empty">No users found</td></tr>'}</tbody></table></div>`
  );
}

function settingsHtml() {
  if (loadError) return `<div class="ws-error">${escapeHtml(loadError)}</div>`;
  if (settings === null) return `<div class="ws-empty">Loading…</div>`;
  const on = settings.signup_enabled;
  const roles = ROLES.filter((r) => r.value !== "admin")
    .map((r) => `<option value="${r.value}"${r.value === settings.default_role ? " selected" : ""}>${r.label}</option>`)
    .join("");
  return (
    `<div class="ad-settings"><div class="ad-section-title">Authentication</div>` +
    `<div class="ad-setting"><div><div class="ad-setting-name">Default User Role</div>` +
    `<div class="ad-setting-sub">What a new sign-up becomes. Pending accounts wait for an admin.</div></div>` +
    `<select class="ad-select" data-setting="default_role">${roles}</select></div>` +
    `<div class="ad-setting"><div><div class="ad-setting-name">Enable New Sign Ups</div>` +
    `<div class="ad-setting-sub">Let anyone who can open this page create an account.</div></div>` +
    `<button type="button" class="ad-switch${on ? " on" : ""}" role="switch" aria-checked="${on}" ` +
    `data-setting="signup_enabled" aria-label="Enable New Sign Ups"><span class="ad-switch-thumb"></span></button></div>` +
    `</div>`
  );
}

function pageHtml() {
  const toggle =
    isMobile() && !state.showSidebar
      ? `<div class="ws-sidebar-toggle"><button type="button" class="nav-sidebar-button" data-action="open-sidebar" aria-label="Open Sidebar">` +
        `<div class="nav-sidebar-icon">${icon("sidebar", "size-5")}</div></button></div>`
      : "";
  const current = tab();
  const link = (name, label) =>
    `<a class="ws-tab${current === name ? " ad-tab-active" : " ad-tab"}" href="#/admin/${name}">${label}</a>`;
  return (
    `<div class="workspace admin"><nav class="ws-nav"><div class="ws-nav-row">${toggle}` +
    `<div><div class="ws-tabs scrollbar-none">${link("users", "Users")}${link("settings", "Settings")}</div></div>` +
    `</div></nav>` +
    `<div class="ws-container" id="admin-container">${current === "settings" ? settingsHtml() : usersHtml()}</div></div>`
  );
}

export function renderAdminPage() {
  const container = document.getElementById("chat-container");
  const focused = document.activeElement?.classList.contains("ad-search-input");
  container.innerHTML = pageHtml();
  if (focused) {
    const input = container.querySelector(".ad-search-input");
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }
  if (!loading && (users === null || settings === null) && !loadError) load();
}

function rerenderIfShown() {
  if (state.route.name === "admin") renderAdminPage();
}

async function load() {
  loading = true;
  try {
    const [u, s] = await Promise.all([api("/api/users"), api("/api/settings/auth")]);
    if (!u.ok) throw new Error(await errorDetail(u));
    if (!s.ok) throw new Error(await errorDetail(s));
    users = await u.json();
    settings = await s.json();
    loadError = null;
  } catch (error) {
    loadError = `Could not load the Admin Panel: ${error.message}`;
  } finally {
    loading = false;
    rerenderIfShown();
  }
}

/** Forget what was loaded, so the page asks again next time it opens. */
export function resetAdminPage() {
  users = null;
  settings = null;
  loadError = null;
}

// --- Actions ---

async function send(path, method, json) {
  const response = await api(path, { method, json });
  if (!response.ok) throw new Error(await errorDetail(response));
  return response.json();
}

async function addUser() {
  const created = await formDialog({
    title: "Add User",
    fields: [
      { name: "role", label: "Role", value: "user", options: ROLES },
      { name: "name", label: "Name", placeholder: "Enter Full Name" },
      { name: "email", label: "Email", type: "email", placeholder: "Enter Email" },
      { name: "password", label: "Password", type: "password", placeholder: "At least 8 characters" },
    ],
    confirmLabel: "Save",
    submit: (values) => send("/api/users", "POST", values),
  });
  if (created) {
    users.push(created);
    toast(`${created.name} can now sign in.`, "success");
    rerenderIfShown();
  }
}

async function editUser(id) {
  const user = users.find((u) => u.id === id);
  if (!user) return;
  const updated = await formDialog({
    title: "Edit User",
    message: "Leave the password empty to keep it. A new password signs the user out everywhere.",
    fields: [
      { name: "name", label: "Name", value: user.name },
      { name: "email", label: "Email", type: "email", value: user.email },
      { name: "password", label: "New Password", type: "password", placeholder: "Unchanged" },
    ],
    submit: (values) => {
      const changes = { name: values.name, email: values.email };
      if (values.password) changes.password = values.password;
      return send(`/api/users/${encodeURIComponent(id)}`, "PATCH", changes);
    },
  });
  if (updated) {
    Object.assign(user, updated);
    toast("User updated.", "success");
    rerenderIfShown();
  }
}

async function deleteUser(id) {
  const user = users.find((u) => u.id === id);
  if (!user) return;
  const ok = await confirmDialog({
    title: "Delete user?",
    html: `This deletes <span class="font-semibold">${escapeHtml(user.name)}</span> and all of their chats.`,
  });
  if (!ok) return;
  try {
    await send(`/api/users/${encodeURIComponent(id)}`, "DELETE");
    users = users.filter((u) => u.id !== id);
    if (id === state.user?.id) location.reload();
    toast("User deleted.", "success");
  } catch (error) {
    toast(error.message, "error");
  }
  rerenderIfShown();
}

async function changeRole(select) {
  const user = users.find((u) => u.id === select.dataset.id);
  if (!user) return;
  try {
    const updated = await send(`/api/users/${encodeURIComponent(user.id)}`, "PATCH", { role: select.value });
    Object.assign(user, updated);
    toast(`${user.name} is now ${ROLES.find((r) => r.value === user.role)?.label ?? user.role}.`, "success");
    // Demoting yourself ends your admin view.
    if (user.id === state.user?.id && user.role !== "admin") location.reload();
  } catch (error) {
    toast(error.message, "error");
  }
  rerenderIfShown();
}

async function saveSetting(changes) {
  try {
    settings = await send("/api/settings/auth", "PUT", { ...settings, ...changes });
    toast("Settings saved.", "success");
  } catch (error) {
    toast(error.message, "error");
  }
  rerenderIfShown();
}

export function initAdminPage() {
  const container = document.getElementById("chat-container");
  container.addEventListener("click", (event) => {
    if (state.route.name !== "admin") return;
    const target = event.target.closest("[data-action], [data-setting]");
    if (!target) return;
    const action = target.dataset.action;
    if (action === "add-user") addUser();
    else if (action === "edit-user") editUser(target.dataset.id);
    else if (action === "delete-user") deleteUser(target.dataset.id);
    else if (action === "open-sidebar") setShowSidebar(true);
    else if (target.dataset.setting === "signup_enabled") saveSetting({ signup_enabled: !settings.signup_enabled });
  });
  container.addEventListener("change", (event) => {
    if (state.route.name !== "admin") return;
    if (event.target.classList.contains("ad-role")) changeRole(event.target);
    else if (event.target.dataset.setting === "default_role") saveSetting({ default_role: event.target.value });
  });
  container.addEventListener("input", (event) => {
    if (state.route.name !== "admin" || !event.target.classList.contains("ad-search-input")) return;
    query = event.target.value;
    renderAdminPage();
  });
}

// Signing in: the sign-in and sign-up page, the wait for an admin's approval,
// and the user menu. Layout follows Open WebUI v0.6.30's routes/auth/+page.svelte,
// layout/Overlay/AccountPending.svelte and layout/Sidebar/UserMenu.svelte.
//
// Accounts belong to the PC serving this page. With them off (or on an older
// Huddle), none of this shows and the page works as before.

import { icon } from "./icons.js";
import { escapeHtml } from "./markdown.js";
import { state } from "./store.js";
import { api, closeMenu, errorDetail, formDialog, LOGO_URL, openMenu, toast } from "./ui.js";

/**
 * Where this page stands: "local" (no accounts), "signin", "pending",
 * "active", or "unavailable" (accounts on, but the database would not open).
 */
export async function loadSession() {
  let config = { enabled: false };
  try {
    const response = await api("/api/auth/config");
    if (response.ok) config = await response.json();
  } catch {
    // Unreachable, or a Huddle from before accounts: behave as before.
  }
  state.accounts = config;
  if (!config.enabled) return "local";
  if (config.available === false) return "unavailable";
  try {
    const response = await api("/api/auth/me");
    if (!response.ok) return "signin";
    state.user = (await response.json()).user;
  } catch {
    return "signin";
  }
  return state.user.role === "pending" ? "pending" : "active";
}

function root() {
  return document.getElementById("app-root");
}

// --- Sign in / sign up ---

function field(name, label, type, placeholder, autocomplete) {
  return (
    `<div class="auth-field"><div class="auth-label">${label}</div>` +
    `<input class="auth-input" name="${name}" type="${type}" placeholder="${placeholder}" ` +
    `autocomplete="${autocomplete}" required></div>`
  );
}

function authHtml(mode) {
  const config = state.accounts;
  const signup = mode === "signup";
  const note = config.first_user
    ? `<div class="auth-note">ⓘ This first account becomes the admin, who approves everyone who signs up after. ` +
      `Accounts and chats stay on this PC.</div>`
    : "";
  let switcher = "";
  if (!config.first_user && config.signup_enabled) {
    switcher = signup
      ? `Already have an account? <button type="button" class="auth-switch" data-mode="signin">Sign in</button>`
      : `Don't have an account? <button type="button" class="auth-switch" data-mode="signup">Sign up</button>`;
  }
  return (
    `<div class="auth"><div class="auth-logo"><img src="${LOGO_URL}" alt="" draggable="false"></div>` +
    `<div class="auth-center"><form class="auth-form" data-mode="${mode}">` +
    `<div class="auth-title">${signup ? "Sign up to Huddle" : "Sign in to Huddle"}</div>${note}` +
    `<div class="auth-fields">` +
    (signup ? field("name", "Name", "text", "Enter Your Full Name", "name") : "") +
    field("email", "Email", "email", "Enter Your Email", "email") +
    field("password", "Password", "password", "Enter Your Password", signup ? "new-password" : "current-password") +
    `</div>` +
    `<button type="submit" class="auth-submit">${signup ? "Create Account" : "Sign in"}</button>` +
    (switcher ? `<div class="auth-alt">${switcher}</div>` : "") +
    `</form></div></div>`
  );
}

/** The page for someone not signed in; it replaces the whole app. */
export function renderAuthPage() {
  const show = (mode) => {
    root().innerHTML = authHtml(mode);
    root().querySelector(".auth-input")?.focus();
  };
  show(state.accounts.first_user ? "signup" : "signin");

  root().addEventListener("click", (event) => {
    const switcher = event.target.closest(".auth-switch");
    if (switcher) show(switcher.dataset.mode);
  });
  root().addEventListener("submit", async (event) => {
    const form = event.target.closest(".auth-form");
    if (!form) return;
    event.preventDefault();
    const values = Object.fromEntries(new FormData(form).entries());
    const button = form.querySelector(".auth-submit");
    button.disabled = true;
    try {
      const response = await api(`/api/auth/${form.dataset.mode}`, { method: "POST", json: values });
      if (!response.ok) throw new Error(await errorDetail(response));
      location.reload();
    } catch (error) {
      toast(error.message || "Could not reach Huddle", "error");
      button.disabled = false;
    }
  });
}

// --- Waiting for approval, and a database that would not open ---

function noticeHtml(title, subtitle, text, actions) {
  return (
    `<div class="pending"><div class="pending-inner">` +
    `<div class="pending-title">${title}<br>${subtitle}</div>` +
    `<div class="pending-text">${text}</div>` +
    `<div class="pending-actions">${actions}</div>` +
    `</div></div>`
  );
}

export function renderPendingPage() {
  root().innerHTML = noticeHtml(
    "Account Activation Pending",
    "Contact the admin for access",
    "Your account is waiting for an admin to approve it. Once they have, sign in again or check " +
      "again here. Admins approve accounts from the Admin Panel.",
    `<button type="button" class="pending-button" data-action="check">Check Again</button>` +
      `<button type="button" class="pending-link" data-action="signout">Sign Out</button>`
  );
  root().addEventListener("click", (event) => {
    const action = event.target.closest("[data-action]")?.dataset.action;
    if (action === "check") location.reload();
    if (action === "signout") signOut();
  });
}

export function renderUnavailablePage() {
  root().innerHTML = noticeHtml(
    "Sign-in is unavailable",
    "on this PC",
    "Huddle could not open its accounts database, so no one can sign in here. The service log " +
      "says why: <code>journalctl -u huddle</code>.",
    `<button type="button" class="pending-button" data-action="check">Try Again</button>`
  );
  root().addEventListener("click", (event) => {
    if (event.target.closest("[data-action=check]")) location.reload();
  });
}

// --- Signed in ---

export async function signOut() {
  try {
    await api("/api/auth/signout", { method: "POST" });
  } finally {
    location.hash = "#/";
    location.reload();
  }
}

let ending = false;

/** The server no longer accepts this sign-in: say so, then show the sign-in page. */
export function sessionEnded() {
  if (ending) return;
  ending = true;
  toast("Your sign-in has ended. Please sign in again.", "error", 0);
  setTimeout(() => location.reload(), 1500);
}

async function changePassword() {
  await formDialog({
    title: "Change Password",
    fields: [
      { name: "current_password", label: "Current Password", type: "password" },
      { name: "new_password", label: "New Password", type: "password", placeholder: "At least 8 characters" },
      { name: "confirm", label: "Confirm New Password", type: "password" },
    ],
    confirmLabel: "Update password",
    submit: async (values) => {
      if (values.new_password !== values.confirm) throw new Error("The new passwords do not match.");
      const response = await api("/api/auth/password", {
        method: "POST",
        json: { current_password: values.current_password, new_password: values.new_password },
      });
      if (!response.ok) throw new Error(await errorDetail(response));
      toast("Password updated. Other browsers were signed out.", "success");
      return true;
    },
  });
}

/** UserMenu.svelte: who is signed in, the Admin Panel for admins, and Sign Out. */
export function openUserMenu(anchor) {
  const user = state.user;
  if (!user) return;
  const menu = document.createElement("div");
  menu.className = "dropdown user-dropdown";
  menu.innerHTML =
    `<div class="user-menu-head"><div class="user-menu-name">${escapeHtml(user.name)}</div>` +
    `<div class="user-menu-email">${escapeHtml(user.email)}</div></div>` +
    (user.role === "admin"
      ? `<button type="button" class="dropdown-item" data-menu="admin">${icon("userGroup", "size-5")}<div>Admin Panel</div></button>`
      : "") +
    `<button type="button" class="dropdown-item" data-menu="password">${icon("key", "size-5")}<div>Change Password</div></button>` +
    `<hr class="dropdown-hr">` +
    `<button type="button" class="dropdown-item" data-menu="signout">${icon("signOut", "size-5")}<div>Sign Out</div></button>`;
  menu.addEventListener("click", (event) => {
    const item = event.target.closest("[data-menu]");
    if (!item) return;
    closeMenu();
    if (item.dataset.menu === "admin") location.hash = "#/admin";
    else if (item.dataset.menu === "password") changePassword();
    else signOut();
  });
  openMenu(anchor, menu, { align: "start", sideOffset: 4 });
}

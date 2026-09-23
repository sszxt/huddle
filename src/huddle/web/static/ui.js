// Shared UI pieces: API fetch, toasts, confirm dialog, dropdown menus, avatars.

import { escapeHtml } from "./markdown.js";
import { state } from "./store.js";

/** fetch() against Huddle's own API. `/v1/*` carries the API key when one is set. */
export function api(path, { json, headers = {}, ...options } = {}) {
  const all = { ...headers };
  if (json !== undefined) {
    all["Content-Type"] = "application/json";
    options.body = JSON.stringify(json);
  }
  if (state.apiKey && path.startsWith("/v1/")) all.Authorization = `Bearer ${state.apiKey}`;
  return fetch(path, { ...options, headers: all });
}

/** The most useful error text a failed response carries. */
export async function errorDetail(response) {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (body?.error?.message) return body.error.message;
    if (typeof body?.error === "string") return body.error;
    return JSON.stringify(body);
  } catch {
    return `${response.status} ${response.statusText}`.trim();
  }
}

/** Copy text; `notify` shows Open WebUI's toast (its code blocks say "Copied" instead). */
export async function copyText(text, notify = true) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // navigator.clipboard needs a secure context; a LAN http:// page is not one.
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  if (notify) toast("Copying to clipboard was successful!", "success");
}

// --- Toasts: svelte-sonner, `richColors position="top-right" closeButton` ---

const TOAST_ICONS = {
  success:
    '<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clip-rule="evenodd"/>',
  error:
    '<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 0110 5zm0 10a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd"/>',
  info: '<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clip-rule="evenodd"/>',
};

export function toast(message, type = "info", duration = 4000) {
  const host = document.getElementById("toaster");
  const el = document.createElement("li");
  el.className = `toast toast-${type}`;
  el.innerHTML =
    `<button type="button" class="toast-close" aria-label="Close toast">` +
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>` +
    `<div class="toast-icon"><svg viewBox="0 0 20 20" fill="currentColor">${TOAST_ICONS[type] ?? TOAST_ICONS.info}</svg></div>` +
    `<div class="toast-content">${escapeHtml(message)}</div>`;
  const remove = () => {
    el.classList.add("toast-leaving");
    setTimeout(() => el.remove(), 200);
  };
  el.querySelector(".toast-close").addEventListener("click", remove);
  host.prepend(el);
  if (duration) setTimeout(remove, duration);
  return remove;
}

// --- ConfirmDialog.svelte ---

/** Resolves true/false, or the entered text for `input` dialogs (null on cancel). */
export function confirmDialog({
  title = "Confirm your action",
  message = "This action cannot be undone. Do you wish to continue?",
  html = null,
  input = false,
  inputType = "text",
  inputPlaceholder = "",
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
} = {}) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "dialog-overlay";
    const inputHtml = input
      ? `<input class="dialog-input" type="${inputType}" placeholder="${escapeHtml(inputPlaceholder)}" autocomplete="off">`
      : "";
    overlay.innerHTML =
      `<div class="dialog-panel fly-in" role="dialog" aria-modal="true"><div class="dialog-body">` +
      `<div class="dialog-title">${escapeHtml(title)}</div>` +
      `<div class="dialog-message">${html ?? escapeHtml(message)}${inputHtml}</div>` +
      `<div class="dialog-actions">` +
      `<button type="button" class="dialog-cancel">${escapeHtml(cancelLabel)}</button>` +
      `<button type="button" class="dialog-confirm">${escapeHtml(confirmLabel)}</button>` +
      `</div></div></div>`;
    const inputEl = overlay.querySelector(".dialog-input");
    const close = (value) => {
      overlay.remove();
      document.removeEventListener("keydown", onKey, true);
      resolve(value);
    };
    const confirm = () => close(input ? inputEl.value : true);
    const cancel = () => close(input ? null : false);
    const onKey = (event) => {
      if (event.key === "Escape") cancel();
      if (event.key === "Enter") {
        event.preventDefault();
        confirm();
      }
    };
    overlay.addEventListener("mousedown", (event) => {
      if (event.target === overlay) cancel();
    });
    overlay.querySelector(".dialog-cancel").addEventListener("click", cancel);
    overlay.querySelector(".dialog-confirm").addEventListener("click", confirm);
    document.addEventListener("keydown", onKey, true);
    document.body.append(overlay);
    (inputEl ?? overlay.querySelector(".dialog-confirm")).focus();
  });
}

// --- Dropdown menus (bits-ui DropdownMenu as Open WebUI uses it) ---

let openMenuState = null;

export function closeMenu() {
  if (!openMenuState) return;
  const { el, onClose, cleanup } = openMenuState;
  openMenuState = null;
  cleanup();
  el.remove();
  onClose?.();
}

/**
 * Show `content` (an element) below `anchor`. `align` is "start" or "end";
 * offsets follow bits-ui's sideOffset/alignOffset.
 */
export function openMenu(anchor, content, { align = "start", sideOffset = 2, alignOffset = 0, onClose } = {}) {
  closeMenu();
  const el = document.createElement("div");
  el.className = "menu-layer fly-in";
  el.append(content);
  document.body.append(el);

  const place = () => {
    const rect = anchor.getBoundingClientRect();
    const width = el.offsetWidth;
    let left = align === "end" ? rect.right - width - alignOffset : rect.left + alignOffset;
    left = Math.max(8, Math.min(left, window.innerWidth - width - 8));
    let top = rect.bottom + sideOffset;
    if (top + el.offsetHeight > window.innerHeight - 8) {
      top = Math.max(8, rect.top - sideOffset - el.offsetHeight);
    }
    el.style.left = `${left}px`;
    el.style.top = `${top}px`;
  };
  place();

  const onDown = (event) => {
    if (!el.contains(event.target) && !anchor.contains(event.target)) closeMenu();
  };
  const onKey = (event) => {
    if (event.key === "Escape") closeMenu();
  };
  document.addEventListener("mousedown", onDown, true);
  document.addEventListener("keydown", onKey, true);
  window.addEventListener("resize", closeMenu);
  openMenuState = {
    el,
    onClose,
    cleanup: () => {
      document.removeEventListener("mousedown", onDown, true);
      document.removeEventListener("keydown", onKey, true);
      window.removeEventListener("resize", closeMenu);
    },
  };
  return el;
}

// --- Tooltips: common/Tooltip.svelte (tippy.js, theme "dark", no arrow, offset 4) ---

export function initTooltips() {
  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.hidden = true;
  document.body.append(tip);
  let owner = null;

  const hide = () => {
    owner = null;
    tip.hidden = true;
    tip.classList.remove("visible");
  };
  const show = (el) => {
    owner = el;
    tip.textContent = el.dataset.tooltip;
    tip.hidden = false;
    const rect = el.getBoundingClientRect();
    const placement = el.dataset.placement ?? "top";
    const gap = 4;
    let left;
    let top;
    if (placement === "right") {
      left = rect.right + gap;
      top = rect.top + rect.height / 2 - tip.offsetHeight / 2;
    } else if (placement === "bottom") {
      left = rect.left + rect.width / 2 - tip.offsetWidth / 2;
      top = rect.bottom + gap;
    } else {
      left = placement === "top-start" ? rect.left : rect.left + rect.width / 2 - tip.offsetWidth / 2;
      top = rect.top - gap - tip.offsetHeight;
    }
    tip.style.left = `${Math.max(4, Math.min(left, window.innerWidth - tip.offsetWidth - 4))}px`;
    tip.style.top = `${Math.max(4, top)}px`;
    requestAnimationFrame(() => tip.classList.add("visible"));
  };

  document.addEventListener("mouseover", (event) => {
    const el = event.target.closest?.("[data-tooltip]");
    if (el === owner) return;
    if (!el || !el.dataset.tooltip) return hide();
    show(el);
  });
  document.addEventListener("mousedown", hide, true);
  document.addEventListener("scroll", hide, true);
  window.addEventListener("blur", hide);
}

// --- Avatars: Open WebUI's generateInitialsImage ---

const avatarCache = new Map();

export function initialsImage(name) {
  const key = name ?? "";
  if (avatarCache.has(key)) return avatarCache.get(key);
  const canvas = document.createElement("canvas");
  canvas.width = 100;
  canvas.height = 100;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#F39C12";
  ctx.fillRect(0, 0, 100, 100);
  ctx.fillStyle = "#FFFFFF";
  ctx.font = "40px Helvetica";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const trimmed = key.trim();
  const parts = trimmed.split(" ");
  const initials =
    trimmed.length > 0 ? trimmed[0] + (parts.length > 1 ? trimmed[trimmed.lastIndexOf(" ") + 1] : "") : "";
  ctx.fillText(initials.toUpperCase(), 50, 50);
  const url = canvas.toDataURL();
  avatarCache.set(key, url);
  return url;
}

export const LOGO_URL = "favicon.svg";

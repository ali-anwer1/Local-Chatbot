/* Formatting, clipboard, feedback, and shared UI state helpers */

import { KNOWN_MODEL_SIZE_ESTIMATES, SVG_ICONS } from "./constants.js";
import { el } from "./dom.js";
import { state } from "./state.js";

export function normalizeTag(tag) {
  if (!tag) return "";
  const trimmed = tag.trim().toLowerCase();
  return trimmed.includes(":") ? trimmed : `${trimmed}:latest`;
}

export function tagsMatch(a, b) {
  if (!a || !b) return false;
  return normalizeTag(a) === normalizeTag(b);
}

export function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

export function formatBytes(bytes) {
  const n = Number(bytes || 0);
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export function formatTimestamp(isoStr) {
  if (!isoStr) return "—";
  const date = new Date(isoStr);
  if (Number.isNaN(date.getTime())) return isoStr;
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  const timeStr = date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  if (sameDay) {
    return `Today at ${timeStr}`;
  }
  const dateStr = date.toLocaleDateString([], { month: "short", day: "numeric" });
  return `${dateStr} at ${timeStr}`;
}

export function formatShortDate(isoStr) {
  if (!isoStr) return "";
  const date = new Date(isoStr);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

export function getFileExtensionBadge(filename) {
  const parts = String(filename || "").split(".");
  const ext = parts.length > 1 ? parts.pop().toLowerCase() : "doc";
  return ext.slice(0, 4).toUpperCase();
}

export function estimateModelSize(tag) {
  if (!tag) return "size determined by Ollama registry";
  const cleaned = tag.trim().toLowerCase();
  if (KNOWN_MODEL_SIZE_ESTIMATES[cleaned]) {
    return KNOWN_MODEL_SIZE_ESTIMATES[cleaned];
  }
  const base = cleaned.split(":")[0];
  if (KNOWN_MODEL_SIZE_ESTIMATES[base]) {
    return KNOWN_MODEL_SIZE_ESTIMATES[base];
  }
  return "size determined by Ollama registry";
}

export async function copyToClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fallback below
    }
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
    return true;
  } catch {
    return false;
  } finally {
    ta.remove();
  }
}

export function attachCopyFeedback(buttonEl, getTextFn, labelText = "Copy") {
  buttonEl.innerHTML = `${SVG_ICONS.copy}<span>${escapeHtml(labelText)}</span>`;
  let timer = null;
  buttonEl.addEventListener("click", async () => {
    const ok = await copyToClipboard(getTextFn());
    if (!ok) return;
    if (timer) clearTimeout(timer);
    buttonEl.classList.add("copied");
    buttonEl.innerHTML = `${SVG_ICONS.check}<span>Copied</span>`;
    timer = setTimeout(() => {
      buttonEl.classList.remove("copied");
      buttonEl.innerHTML = `${SVG_ICONS.copy}<span>${escapeHtml(labelText)}</span>`;
      timer = null;
    }, 1600);
  });
}

export function showFeedback(targetEl, text, variant = "success") {
  targetEl.textContent = text;
  targetEl.className = `status-feedback ${variant}`;
  targetEl.hidden = false;
}

// Auto-dismissing Settings Save Toast (#8)
export function showSettingsFeedback(text, variant = "success", autoDismissMs = 3200) {
  if (state.settingsFeedbackTimer) {
    clearTimeout(state.settingsFeedbackTimer);
    state.settingsFeedbackTimer = null;
  }
  showFeedback(el.settingsSaveFeedback, text, variant);
  if (autoDismissMs > 0) {
    state.settingsFeedbackTimer = setTimeout(() => {
      el.settingsSaveFeedback.hidden = true;
      state.settingsFeedbackTimer = null;
    }, autoDismissMs);
  }
}

// --- Panel Collapse / Drawer Controls (#6, #9) ---

export function setLeftSidebarCollapsed(collapsed) {
  state.leftSidebarCollapsed = Boolean(collapsed);
  el.appLayout.classList.toggle("left-collapsed", state.leftSidebarCollapsed);
  el.btnCloseLeftSidebar.setAttribute("aria-expanded", String(!state.leftSidebarCollapsed));
  syncSidebarToggleVisibility();
}

export function syncSidebarToggleVisibility() {
  const collapsed = state.leftSidebarCollapsed;
  el.btnOpenLeftSidebar.hidden = !collapsed;
  el.btnHeaderNewChat.hidden = !collapsed;
  el.emptyStageTopbar.hidden = !(collapsed && el.chatWorkspace.hidden);
}

export function setSettingsPanelOpen(open) {
  state.settingsPanelOpen = Boolean(open);
  el.appLayout.classList.toggle("settings-collapsed", !state.settingsPanelOpen);
  el.btnToggleSettings.setAttribute("aria-expanded", String(state.settingsPanelOpen));
  el.btnToggleSettings.title = state.settingsPanelOpen
    ? "Close Settings & Knowledge Base"
    : "Open Settings & Knowledge Base";
}

// --- Thinking Mode Helpers (#3) ---

export function isThinkingEnabled(profileId) {
  if (!profileId) return false;
  if (state.thinkingByProfile.has(profileId)) {
    return Boolean(state.thinkingByProfile.get(profileId));
  }
  return Boolean(state.activeProfile && state.activeProfile.enable_thinking);
}

export function setThinkingEnabled(profileId, enabled) {
  const val = Boolean(enabled);
  if (profileId) {
    state.thinkingByProfile.set(profileId, val);
  }
  el.btnThinkingToggle.setAttribute("aria-pressed", String(val));
  el.thinkingPillLabel.textContent = val ? "Thinking: On" : "Thinking: Off";
  el.settingThinking.checked = val;
}

// --- Model Dropdown Builder (§4.5) ---

export function populateModelSelect(selectEl, models, selectedValue, fallbackDefault) {
  selectEl.innerHTML = "";
  let matched = false;

  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.name;
    const unverifiedTag = m.capabilities_verified === false ? " · unverified" : "";
    opt.textContent = `${m.name} (${m.size_human}${unverifiedTag})`;
    if (selectedValue && tagsMatch(m.name, selectedValue)) {
      opt.selected = true;
      matched = true;
    }
    selectEl.appendChild(opt);
  }

  const desired = selectedValue || fallbackDefault;
  if (!matched && desired) {
    for (const opt of selectEl.options) {
      if (tagsMatch(opt.value, desired)) {
        opt.selected = true;
        matched = true;
        break;
      }
    }
  }

  if (!matched && selectedValue) {
    const missingOpt = document.createElement("option");
    missingOpt.value = selectedValue;
    missingOpt.textContent = `${selectedValue} (not installed)`;
    missingOpt.selected = true;
    missingOpt.disabled = true;
    selectEl.insertBefore(missingOpt, selectEl.firstChild);
  }

  if (selectEl.options.length === 0 && fallbackDefault) {
    const placeholder = document.createElement("option");
    placeholder.value = fallbackDefault;
    placeholder.textContent = `${fallbackDefault} (not installed)`;
    selectEl.appendChild(placeholder);
  }
}

export function updateInteractiveControlsLock(avail) {
  const ollamaDown = state.health && !state.health.ollama_reachable;
  const chatDisabled = ollamaDown || Boolean(avail.chat_blocked);
  const uploadDisabled = ollamaDown || Boolean(avail.upload_blocked);

  el.chatInput.disabled = chatDisabled;
  el.btnSendChat.disabled = chatDisabled || Boolean(state.chatAbortController);
  el.fileUploadInput.disabled = uploadDisabled;
  el.fileUploadLabel.classList.toggle("disabled", uploadDisabled);
  el.modalFileUploadInput.disabled = uploadDisabled;
  el.modalFileUploadLabel.classList.toggle("disabled", uploadDisabled);
}

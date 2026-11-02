/* Profile Settings Form, Save History & Embedding Re-index Confirmation (§4.5, §7.10) */

import { apiJson } from "./api.js";
import { el } from "./dom.js";
import { state } from "./state.js";
import {
  escapeHtml,
  formatTimestamp,
  isThinkingEnabled,
  populateModelSelect,
  showFeedback,
  showSettingsFeedback,
  tagsMatch,
} from "./utils.js";

let renderProfileSidebarCb = () => {};
let renderMainViewCb = () => {};
let selectProfileCb = async () => {};

export function configureSettings({ renderProfileSidebar, renderMainView, selectProfile }) {
  renderProfileSidebarCb = renderProfileSidebar;
  renderMainViewCb = renderMainView;
  selectProfileCb = selectProfile;
}

export function renderSettingsForm(prof, opts = {}) {
  const h = state.health || {};
  el.settingName.value = prof.name || "";
  if (el.settingDefaultLanguage) {
    el.settingDefaultLanguage.value = prof.default_language || "en";
  }
  el.settingPreamble.value = prof.preamble || "";
  populateModelSelect(
    el.settingChatModel,
    h.completion_models || [],
    prof.chat_model,
    h.default_chat_model
  );
  populateModelSelect(
    el.settingEmbedModel,
    h.embedding_models || [],
    prof.embedding_model,
    h.default_embedding_model
  );

  el.settingThinking.checked = isThinkingEnabled(prof.id);

  const thresh = Number(prof.similarity_threshold ?? 0.5);
  el.settingThreshold.value = thresh.toFixed(2);
  el.settingThresholdOut.textContent = thresh.toFixed(2);

  el.settingTemperature.value = prof.temperature ?? 0.2;
  el.settingTopK.value = prof.top_k ?? 5;
  el.settingChunkSize.value = prof.chunk_size ?? 1000;
  el.settingChunkOverlap.value = prof.chunk_overlap ?? 150;
  el.settingHistoryTurns.value = prof.history_turns ?? 4;
  const refusalMsgs = prof.refusal_messages || {};
  el.settingRefusalMsg.value = refusalMsgs.en || prof.refusal_message || "";
  document.querySelectorAll(".refusal-lang-fields input[data-lang]").forEach((input) => {
    const lang = input.dataset.lang;
    input.value = refusalMsgs[lang] || "";
  });

  if (!opts.preserveFeedback) {
    if (state.settingsFeedbackTimer) {
      clearTimeout(state.settingsFeedbackTimer);
      state.settingsFeedbackTimer = null;
    }
    el.settingsSaveFeedback.hidden = true;
  }
}

export function renderSettingsHistory(prof) {
  el.settingsHistoryBox.innerHTML = "";
  const history = Array.isArray(prof.settings_history) ? prof.settings_history : [];

  if (history.length === 0) {
    const p = document.createElement("div");
    p.textContent = `No settings edits saved yet · Created ${formatTimestamp(prof.created_at)}`;
    el.settingsHistoryBox.appendChild(p);
    return;
  }

  const latest = history[0];
  const latestLine = document.createElement("div");
  latestLine.className = "history-latest-line";
  latestLine.innerHTML = `<strong>Last saved:</strong> ${escapeHtml(formatTimestamp(latest.saved_at))}`;
  el.settingsHistoryBox.appendChild(latestLine);

  const latestChanges = document.createElement("div");
  latestChanges.className = "history-latest-changes";
  const summaryText = (latest.changes || []).join(", ") || "Saved";
  latestChanges.textContent = `Latest change: ${summaryText}`;
  el.settingsHistoryBox.appendChild(latestChanges);

  if (history.length > 1) {
    const det = document.createElement("details");
    det.className = "history-details";
    const sum = document.createElement("summary");
    sum.textContent = `View recent save history (${history.length})`;
    det.appendChild(sum);

    const ul = document.createElement("ul");
    ul.className = "history-log-list";
    for (const entry of history) {
      const li = document.createElement("li");
      li.className = "history-log-item";
      const changesStr = (entry.changes || []).join(", ");
      li.innerHTML = `<strong>${escapeHtml(formatTimestamp(entry.saved_at))}</strong> — ${escapeHtml(changesStr)}`;
      ul.appendChild(li);
    }
    det.appendChild(ul);
    el.settingsHistoryBox.appendChild(det);
  }
}

export function collectSettingsPayload() {
  const refusalMessages = {
    en: el.settingRefusalMsg.value.trim(),
  };
  document.querySelectorAll(".refusal-lang-fields input[data-lang]").forEach((input) => {
    const lang = input.dataset.lang;
    const val = input.value.trim();
    if (val) {
      refusalMessages[lang] = val;
    }
  });

  return {
    name: el.settingName.value.trim(),
    preamble: el.settingPreamble.value,
    default_language: el.settingDefaultLanguage ? el.settingDefaultLanguage.value : "en",
    chat_model: el.settingChatModel.value,
    embedding_model: el.settingEmbedModel.value,
    enable_thinking: Boolean(el.settingThinking.checked),
    similarity_threshold: parseFloat(el.settingThreshold.value),
    temperature: parseFloat(el.settingTemperature.value),
    top_k: parseInt(el.settingTopK.value, 10),
    chunk_size: parseInt(el.settingChunkSize.value, 10),
    chunk_overlap: parseInt(el.settingChunkOverlap.value, 10),
    history_turns: parseInt(el.settingHistoryTurns.value, 10),
    refusal_message: el.settingRefusalMsg.value.trim(),
    refusal_messages: refusalMessages,
  };
}

export function openReindexConfirmDialog({ mode, payload, message }) {
  state.pendingReindexPayload = { mode, payload };
  el.reindexConfirmMessage.textContent = message;
  el.dialogConfirmReindex.showModal();
}

export function requestEmbeddingModelSwitch(newEmbedModel) {
  if (!state.activeProfile) return;
  const prof = state.activeProfile;
  if (tagsMatch(prof.embedding_model, newEmbedModel)) return;

  const docCount = prof.document_count || 0;
  const chunkCount = prof.total_chunks || 0;
  openReindexConfirmDialog({
    mode: "update_profile",
    payload: { ...collectSettingsPayload(), embedding_model: newEmbedModel },
    message: `Changing the embedding model from '${prof.embedding_model}' to '${newEmbedModel}' will re-index the entire knowledge base (~${chunkCount} chunk${chunkCount === 1 ? "" : "s"} across ${docCount} document${docCount === 1 ? "" : "s"}). Nothing will change unless you confirm.`,
  });
}

export async function saveProfilePatch(patch, confirmReindex = false) {
  if (!state.activeProfile) return;
  try {
    const updated = await apiJson(
      `/api/profiles/${encodeURIComponent(state.activeProfile.id)}`,
      {
        method: "PUT",
        body: JSON.stringify({ ...patch, confirm_reindex: confirmReindex }),
      }
    );
    state.activeProfile = updated;
    state.thinkingByProfile.set(updated.id, Boolean(updated.enable_thinking));
    state.profiles = state.profiles.map((p) => (p.id === updated.id ? updated : p));
    const docsData = await apiJson(
      `/api/profiles/${encodeURIComponent(updated.id)}/documents`
    );
    state.documents = docsData.documents || [];
    renderProfileSidebarCb();
    renderMainViewCb({ preserveFeedback: true });

    const msg = updated.reindexed
      ? `✓ Settings saved and re-indexed ${updated.reindexed.total_chunks} chunk${updated.reindexed.total_chunks === 1 ? "" : "s"}.`
      : "✓ Settings saved.";
    showSettingsFeedback(msg, "success", 3200);
  } catch (err) {
    if (err.code === "reindex_confirmation_required" && err.payload) {
      openReindexConfirmDialog({
        mode: "update_profile",
        payload: patch,
        message: err.message,
      });
      return;
    }
    showSettingsFeedback(err.message, "error", 0);
  }
}

export function cancelPendingReindex() {
  state.pendingReindexPayload = null;
  if (state.activeProfile) {
    renderSettingsForm(state.activeProfile);
  }
  el.dialogConfirmReindex.close();
}

export async function confirmPendingReindex() {
  const pending = state.pendingReindexPayload;
  state.pendingReindexPayload = null;
  el.dialogConfirmReindex.close();
  if (!pending || !state.activeProfile) return;

  if (pending.mode === "explicit_reindex") {
    try {
      showSettingsFeedback("Re-indexing knowledge base…", "warning", 0);
      const res = await apiJson(
        `/api/profiles/${encodeURIComponent(state.activeProfile.id)}/reindex`,
        { method: "POST" }
      );
      await selectProfileCb(state.activeProfile.id, { preserveFeedback: true });
      showSettingsFeedback(
        `✓ Re-indexed ${res.reindexed?.total_chunks ?? 0} chunk${res.reindexed?.total_chunks === 1 ? "" : "s"}.`,
        "success",
        3500
      );
    } catch (err) {
      showSettingsFeedback(err.message, "error", 0);
    }
  } else if (pending.mode === "update_profile") {
    showSettingsFeedback("Re-indexing with new embedding model…", "warning", 0);
    await saveProfilePatch(pending.payload, true);
  }
}

export function renderAppSettingsForm() {
  if (!el.settingEmbeddingDevice) return;
  const device = state.appSettings?.embedding_device || "cpu";
  el.settingEmbeddingDevice.value = device === "auto" ? "auto" : "cpu";
}

function showAppSettingsFeedback(text, variant = "success", autoDismissMs = 3200) {
  if (!el.appSettingsFeedback) return;
  if (state.appSettingsFeedbackTimer) {
    clearTimeout(state.appSettingsFeedbackTimer);
    state.appSettingsFeedbackTimer = null;
  }
  showFeedback(el.appSettingsFeedback, text, variant);
  if (autoDismissMs > 0) {
    state.appSettingsFeedbackTimer = setTimeout(() => {
      el.appSettingsFeedback.hidden = true;
      state.appSettingsFeedbackTimer = null;
    }, autoDismissMs);
  }
}

export async function saveAppSettings(embeddingDevice) {
  try {
    const updated = await apiJson("/api/app-settings", {
      method: "PUT",
      body: JSON.stringify({ embedding_device: embeddingDevice }),
    });
    state.appSettings = updated;
    renderAppSettingsForm();
    showAppSettingsFeedback("✓ Embedding device saved.", "success", 3200);
  } catch (err) {
    renderAppSettingsForm();
    showAppSettingsFeedback(err.message || "Failed to save app settings.", "error", 0);
  }
}

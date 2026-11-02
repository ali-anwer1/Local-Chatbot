/* Local RAG Chatbot — Offline Browser Client Entry Point (§4.5, §4.7 + UX Redesign) */

import { apiJson } from "./api.js";
import { renderProfileBanners } from "./banners.js";
import {
  buildTurnSourcesMap,
  configureChat,
  exportActiveChatAsMarkdown,
  getTranscript,
  handleSendChat,
  openCitationModal,
  openSaveChatDialog,
  persistSavedChat,
  renderConversationTranscript,
  renderSavedChatsSidebar,
  startNewChat,
  updateChatActionButtons,
} from "./chat.js";
import {
  configureDocuments,
  handleFileUpload,
  openDocsManagerModal,
  renderDocsModalList,
  renderDocumentSummaryAndModal,
} from "./documents.js";
import { el } from "./dom.js";
import { initInfoTips } from "./infoTips.js";
import {
  closeOrCancelPull,
  configurePullModal,
  executeModelPull,
  openPullModal,
  reviewPullConfirmation,
} from "./pullModal.js";
import {
  cancelPendingReindex,
  collectSettingsPayload,
  confirmPendingReindex,
  configureSettings,
  renderAppSettingsForm,
  renderSettingsForm,
  renderSettingsHistory,
  requestEmbeddingModelSwitch,
  saveAppSettings,
  saveProfilePatch,
} from "./settings.js";
import { state } from "./state.js";
import {
  copyToClipboard,
  escapeHtml,
  getFileExtensionBadge,
  isThinkingEnabled,
  populateModelSelect,
  setLeftSidebarCollapsed,
  setSettingsPanelOpen,
  setThinkingEnabled,
  syncSidebarToggleVisibility,
  tagsMatch,
  updateInteractiveControlsLock,
} from "./utils.js";

// --- State Refresh & Top-Level Rendering ---

export async function refreshAll(forceRefresh = false) {
  const query = forceRefresh ? "?refresh=true" : "";
  try {
    const [health, profilesData, appSettings] = await Promise.all([
      apiJson(`/api/health${query}`),
      apiJson("/api/profiles"),
      apiJson("/api/app-settings"),
    ]);
    state.health = health;
    state.profiles = profilesData.profiles || [];
    state.appSettings = appSettings;

    if (
      state.activeProfileId &&
      !state.profiles.some((p) => p.id === state.activeProfileId)
    ) {
      state.activeProfileId = null;
      state.activeProfile = null;
    }

    if (!state.activeProfileId && state.profiles.length > 0) {
      state.activeProfileId = state.profiles[0].id;
    }

    renderGlobalStatus();
    renderProfileSidebar();

    if (state.activeProfileId) {
      await selectProfile(state.activeProfileId, { forceRefresh });
    } else {
      renderSavedChatsSidebar();
      renderMainView();
    }
  } catch (err) {
    el.ollamaStatusPill.className = "status-pill offline";
    el.ollamaStatusText.textContent = "Server error";
    el.globalBannerText.textContent = err.message || "Failed to connect to local server.";
    el.globalBanner.hidden = false;
  }
}

function renderGlobalStatus() {
  const h = state.health;
  if (!h) return;

  if (!h.ollama_reachable) {
    el.globalBannerText.textContent =
      h.banner_message ||
      `Ollama is not reachable at ${h.ollama_url}. Start Ollama and click Retry.`;
    el.globalBanner.hidden = false;
    el.ollamaStatusPill.className = "status-pill offline";
    el.ollamaStatusText.textContent = "Ollama unreachable";
  } else {
    el.globalBanner.hidden = true;
    el.ollamaStatusPill.className = "status-pill online";
    const modelCount = (h.models || []).length;
    el.ollamaStatusText.textContent =
      modelCount > 0
        ? `Ollama online · ${modelCount} model${modelCount === 1 ? "" : "s"}`
        : "Ollama online · 0 models";
  }
}

export function renderProfileSidebar() {
  el.profileList.innerHTML = "";
  if (state.profiles.length === 0) {
    const li = document.createElement("li");
    li.className = "sidebar-empty-hint";
    li.textContent = "No profiles created yet.";
    el.profileList.appendChild(li);
    return;
  }

  for (const prof of state.profiles) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `profile-item-btn${prof.id === state.activeProfileId ? " active" : ""}`;

    const top = document.createElement("div");
    top.className = "profile-item-top";

    const nameSpan = document.createElement("span");
    nameSpan.className = "profile-item-name";
    nameSpan.textContent = prof.name;
    top.appendChild(nameSpan);

    if (prof.availability && prof.availability.has_warning) {
      const badge = document.createElement("span");
      badge.className = "warn-badge";
      badge.textContent = "!";
      badge.title = "Model or index configuration needs attention";
      top.appendChild(badge);
    }

    const meta = document.createElement("span");
    meta.className = "profile-item-meta";
    const docCount = prof.document_count || 0;
    const chunkCount = prof.total_chunks || 0;
    meta.textContent = `${docCount} doc${docCount === 1 ? "" : "s"} · ${chunkCount} chunks`;

    btn.appendChild(top);
    btn.appendChild(meta);
    btn.addEventListener("click", () => selectProfile(prof.id));
    li.appendChild(btn);
    el.profileList.appendChild(li);
  }
}

function renderFirstRunCommands() {
  const cmds = (state.health && state.health.recommended_commands) || [];
  el.recommendedCommandsList.innerHTML = "";
  for (const cmd of cmds) {
    const row = document.createElement("div");
    row.className = "command-row";

    const codeEl = document.createElement("code");
    codeEl.textContent = cmd;
    row.appendChild(codeEl);

    const actions = document.createElement("div");
    actions.className = "banner-actions";

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "btn subtle small";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", async () => {
      const ok = await copyToClipboard(cmd);
      if (ok) {
        copyBtn.textContent = "Copied";
        setTimeout(() => {
          copyBtn.textContent = "Copy";
        }, 1500);
      }
    });
    actions.appendChild(copyBtn);

    const modelTag = cmd.replace(/^ollama\s+pull\s+/i, "").trim();
    if (modelTag) {
      const pullBtn = document.createElement("button");
      pullBtn.type = "button";
      pullBtn.className = "btn small";
      pullBtn.textContent = "Download…";
      pullBtn.addEventListener("click", () => openPullModal(modelTag));
      actions.appendChild(pullBtn);
    }

    row.appendChild(actions);
    el.recommendedCommandsList.appendChild(row);
  }
}

function renderEmptyState(prof) {
  if (!el.welcomeHeading) return;
  el.welcomeHeading.textContent = prof.name;

  const firstLine = (prof.preamble || "").trim().split("\n")[0].trim();
  if (firstLine && el.welcomePreamble) {
    el.welcomePreamble.textContent = firstLine;
    el.welcomePreamble.hidden = false;
  } else if (el.welcomePreamble) {
    el.welcomePreamble.hidden = true;
  }

  if (el.welcomeDocList) {
    el.welcomeDocList.innerHTML = "";
    const docs = state.documents || [];
    if (docs.length === 0) {
      const li = document.createElement("li");
      li.className = "welcome-doc-empty";
      li.textContent = "No documents uploaded yet.";
      el.welcomeDocList.appendChild(li);
    } else {
      const slice = docs.slice(0, 8);
      for (const doc of slice) {
        const li = document.createElement("li");
        li.className = "welcome-doc-row";
        const ext = getFileExtensionBadge(doc.filename);
        li.innerHTML = `<span class="welcome-doc-type">${escapeHtml(ext)}</span><span class="welcome-doc-name">${escapeHtml(doc.filename)}</span>`;
        el.welcomeDocList.appendChild(li);
      }
      if (docs.length > 8) {
        const moreLi = document.createElement("li");
        const moreBtn = document.createElement("button");
        moreBtn.type = "button";
        moreBtn.className = "welcome-doc-more-link";
        moreBtn.textContent = `+${docs.length - 8} more…`;
        moreBtn.addEventListener("click", openDocsManagerModal);
        moreLi.appendChild(moreBtn);
        el.welcomeDocList.appendChild(moreLi);
      }
    }
  }
}

export function renderMainView(opts = {}) {
  const h = state.health;
  if (h && h.ollama_reachable && h.no_models_installed) {
    renderFirstRunCommands();
    el.firstRunScreen.hidden = false;
    el.noProfileScreen.hidden = true;
    el.chatWorkspace.hidden = true;
    el.settingsContent.hidden = true;
    syncSidebarToggleVisibility();
    return;
  }

  el.firstRunScreen.hidden = true;

  if (!state.activeProfile) {
    el.noProfileScreen.hidden = false;
    el.chatWorkspace.hidden = true;
    el.settingsContent.hidden = true;
    syncSidebarToggleVisibility();
    return;
  }

  el.noProfileScreen.hidden = true;
  el.chatWorkspace.hidden = false;
  el.settingsContent.hidden = false;
  syncSidebarToggleVisibility();

  const prof = state.activeProfile;
  const avail = prof.availability || {};

  el.activeProfileTitle.textContent = prof.name;
  const activeModelLabel = avail.fallback_active
    ? `${avail.effective_chat_model} (fallback)`
    : avail.effective_chat_model || prof.chat_model;
  el.activeModelBadge.textContent = activeModelLabel;

  const docCount = prof.document_count || 0;
  const chunkCount = prof.total_chunks || 0;
  el.activeKbBadge.textContent = `${docCount} doc${docCount === 1 ? "" : "s"} · ${chunkCount} chunks`;

  setThinkingEnabled(prof.id, isThinkingEnabled(prof.id));
  renderProfileBanners(prof, avail);
  updateInteractiveControlsLock(avail);
  renderSettingsForm(prof, { preserveFeedback: Boolean(opts.preserveFeedback) });
  renderAppSettingsForm();
  renderSettingsHistory(prof);
  renderDocumentSummaryAndModal();
  renderEmptyState(prof);
  renderConversationTranscript(prof.id);
  updateChatActionButtons(prof.id);
}

export async function selectProfile(profileId, opts = {}) {
  const isSwitching = state.activeProfileId !== profileId;
  state.activeProfileId = profileId;
  if (isSwitching) {
    state.dismissedChatBlockForProfile.delete(profileId);
    el.uploadStatusMsg.hidden = true;
  }

  const refreshParam = opts.forceRefresh ? "?refresh=true" : "";
  const [prof, docsData, chatsData] = await Promise.all([
    apiJson(`/api/profiles/${encodeURIComponent(profileId)}${refreshParam}`),
    apiJson(`/api/profiles/${encodeURIComponent(profileId)}/documents`),
    apiJson(`/api/profiles/${encodeURIComponent(profileId)}/chats`),
  ]);

  state.activeProfile = prof;
  state.documents = docsData.documents || [];
  state.savedChatsByProfile.set(profileId, chatsData.chats || []);
  if (!state.thinkingByProfile.has(profileId)) {
    state.thinkingByProfile.set(profileId, Boolean(prof.enable_thinking));
  }
  state.profiles = state.profiles.map((p) => (p.id === prof.id ? prof : p));

  renderProfileSidebar();
  renderSavedChatsSidebar();
  renderMainView({ preserveFeedback: Boolean(opts.preserveFeedback) });
}

// --- Event Listeners ---

function bindEvents() {
  el.btnRetryOllama.addEventListener("click", () => refreshAll(true));
  el.btnFirstRunRetry.addEventListener("click", () => refreshAll(true));

  // Left Sidebar Collapse / Expand (#9)
  el.btnCloseLeftSidebar.addEventListener("click", () => setLeftSidebarCollapsed(true));
  if (el.btnOpenLeftSidebar) {
    el.btnOpenLeftSidebar.addEventListener("click", () => setLeftSidebarCollapsed(false));
  }
  if (el.btnOpenLeftSidebarAlt) {
    el.btnOpenLeftSidebarAlt.addEventListener("click", () => setLeftSidebarCollapsed(false));
  }

  // Right Settings Panel Open / Close (#6)
  el.btnToggleSettings.addEventListener("click", () => {
    setSettingsPanelOpen(!state.settingsPanelOpen);
  });
  el.btnCloseSettings.addEventListener("click", () => {
    setSettingsPanelOpen(false);
  });

  // Thinking Mode Toggle in Composer & Settings (#3)
  el.btnThinkingToggle.addEventListener("click", () => {
    if (!state.activeProfileId) return;
    const next = !isThinkingEnabled(state.activeProfileId);
    setThinkingEnabled(state.activeProfileId, next);
  });
  el.settingThinking.addEventListener("change", () => {
    if (!state.activeProfileId) return;
    setThinkingEnabled(state.activeProfileId, el.settingThinking.checked);
  });

  // Delegated click handler for inline superscript [n] citation chips
  el.chatMessages.addEventListener("click", (event) => {
    const chip = event.target.closest(".citation-chip");
    if (!chip || !state.activeProfileId) return;
    const rowEl = chip.closest(".msg-row.assistant");
    if (!rowEl) return;
    const turnIdx = Number(rowEl.dataset.turnIdx);
    const turns = getTranscript(state.activeProfileId);
    const turn = turns[turnIdx];
    if (!turn) return;
    const citeIdx = Number(chip.dataset.citeIdx);
    const src = buildTurnSourcesMap(turn).get(citeIdx);
    if (src) {
      openCitationModal(src);
    }
  });

  // Starter / Example prompt links: fill composer and focus when clicked
  el.chatWelcome.querySelectorAll(".starter-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (el.chatInput.disabled || state.chatAbortController) return;
      const promptText = chip.dataset.prompt || chip.textContent.trim();
      el.chatInput.value = promptText;
      el.chatInput.style.height = "auto";
      el.chatInput.focus();
    });
  });

  // Overflow menu: Toggle Debug & Keyboard Navigation
  if (el.toggleDebug) {
    el.toggleDebug.addEventListener("click", () => {
      state.debugEnabled = !state.debugEnabled;
      el.toggleDebug.setAttribute("aria-checked", String(state.debugEnabled));
      el.toggleDebug.classList.toggle("checked", state.debugEnabled);
      document.querySelectorAll(".debug-panel").forEach((node) => {
        node.hidden = !state.debugEnabled;
      });
      if (el.chatOverflowMenu && el.chatOverflowMenu.matches(":popover-open")) {
        el.chatOverflowMenu.hidePopover();
      }
    });
  }

  if (el.chatOverflowMenu && el.btnOverflowMenu) {
    el.chatOverflowMenu.addEventListener("toggle", (event) => {
      const isOpen = event.newState === "open";
      el.btnOverflowMenu.setAttribute("aria-expanded", String(isOpen));
      if (isOpen && el.toggleDebug) {
        el.toggleDebug.setAttribute("aria-checked", String(state.debugEnabled));
        el.toggleDebug.classList.toggle("checked", state.debugEnabled);
      }
    });

    el.chatOverflowMenu.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        el.chatOverflowMenu.hidePopover();
        el.btnOverflowMenu.focus();
        return;
      }
      if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
        const items = [...el.chatOverflowMenu.querySelectorAll(".menu-item:not(:disabled)")];
        if (items.length === 0) return;
        const current = items.indexOf(document.activeElement);
        let next = 0;
        if (event.key === "Home") next = 0;
        else if (event.key === "End") next = items.length - 1;
        else if (event.key === "ArrowUp") next = (current - 1 + items.length) % items.length;
        else if (event.key === "ArrowDown") next = (current + 1) % items.length;
        event.preventDefault();
        items[next].focus();
      }
    });
  }

  // New Chat / Clear Chat
  el.btnClearChat.addEventListener("click", startNewChat);
  if (el.btnSidebarNewChat) {
    el.btnSidebarNewChat.addEventListener("click", startNewChat);
  }
  if (el.btnHeaderNewChat) {
    el.btnHeaderNewChat.addEventListener("click", startNewChat);
  }

  // Save Chat & Export .md (#10)
  el.btnSaveChat.addEventListener("click", () => {
    if (el.chatOverflowMenu && el.chatOverflowMenu.matches(":popover-open")) {
      el.chatOverflowMenu.hidePopover();
    }
    openSaveChatDialog();
  });
  el.btnCancelSaveChat.addEventListener("click", () => el.dialogSaveChat.close());
  el.btnCloseSaveChatX.addEventListener("click", () => el.dialogSaveChat.close());
  el.formSaveChat.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.activeProfile) return;
    const title = el.saveChatTitle.value.trim();
    if (!title) return;
    try {
      await persistSavedChat(state.activeProfile.id, title);
      el.dialogSaveChat.close();
    } catch (err) {
      alert(err.message);
    }
  });

  el.btnExportChat.addEventListener("click", () => {
    if (el.chatOverflowMenu && el.chatOverflowMenu.matches(":popover-open")) {
      el.chatOverflowMenu.hidePopover();
    }
    exportActiveChatAsMarkdown();
  });

  // Knowledge Base Document Manager Modal (#7)
  el.btnOpenDocsModal.addEventListener("click", openDocsManagerModal);
  el.btnKbPill.addEventListener("click", openDocsManagerModal);
  el.btnCloseDocsModal.addEventListener("click", () => el.dialogManageDocs.close());
  el.docsSearchInput.addEventListener("input", () => {
    state.docSearchQuery = el.docsSearchInput.value;
    renderDocsModalList();
  });
  document.querySelectorAll(".filter-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      state.docFilter = pill.dataset.docFilter || "all";
      document.querySelectorAll(".filter-pill").forEach((p) => {
        p.classList.toggle("active", p === pill);
      });
      renderDocsModalList();
    });
  });

  // IME-Safe Enter-to-Submit in Chat Textarea
  el.chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (event.isComposing || event.keyCode === 229) {
        return;
      }
      el.chatForm.requestSubmit();
    }
  });

  el.chatForm.addEventListener("submit", handleSendChat);

  el.btnStopStream.addEventListener("click", () => {
    if (state.chatAbortController) {
      state.chatAbortController.abort();
    }
  });

  // New Profile Dialog
  const openNewProfileDialog = () => {
    const h = state.health || {};
    el.newProfileName.value = "";
    el.newProfilePreamble.value = "";
    populateModelSelect(
      el.newProfileChatModel,
      h.completion_models || [],
      null,
      h.default_chat_model || "qwen3.5:4b"
    );
    populateModelSelect(
      el.newProfileEmbedModel,
      h.embedding_models || [],
      null,
      h.default_embedding_model || "nomic-embed-text"
    );
    el.dialogNewProfile.showModal();
  };

  el.btnNewProfile.addEventListener("click", openNewProfileDialog);
  el.btnEmptyNewProfile.addEventListener("click", openNewProfileDialog);
  el.btnCancelNewProfile.addEventListener("click", () => el.dialogNewProfile.close());
  el.btnCloseNewProfileX.addEventListener("click", () => el.dialogNewProfile.close());

  el.formNewProfile.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const created = await apiJson("/api/profiles", {
        method: "POST",
        body: JSON.stringify({
          name: el.newProfileName.value.trim(),
          preamble: el.newProfilePreamble.value,
          chat_model: el.newProfileChatModel.value,
          embedding_model: el.newProfileEmbedModel.value,
        }),
      });
      el.dialogNewProfile.close();
      state.profiles.push(created);
      await selectProfile(created.id);
    } catch (err) {
      alert(err.message);
    }
  });

  // Threshold slider live output
  el.settingThreshold.addEventListener("input", () => {
    el.settingThresholdOut.textContent = Number(el.settingThreshold.value).toFixed(2);
  });

  // Immediate confirmation dialog when changing embedding model dropdown (§4.5)
  el.settingEmbedModel.addEventListener("change", () => {
    if (!state.activeProfile) return;
    const chosen = el.settingEmbedModel.value;
    if (!tagsMatch(chosen, state.activeProfile.embedding_model)) {
      requestEmbeddingModelSwitch(chosen);
    }
  });

  el.btnCancelReindex.addEventListener("click", cancelPendingReindex);
  el.btnConfirmReindex.addEventListener("click", confirmPendingReindex);

  // Save Machine-Wide App Settings (Embedding Device)
  el.settingEmbeddingDevice.addEventListener("change", () => {
    saveAppSettings(el.settingEmbeddingDevice.value);
  });

  // Save Profile Settings
  el.settingsForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.activeProfile) return;
    const payload = collectSettingsPayload();
    if (!tagsMatch(payload.embedding_model, state.activeProfile.embedding_model)) {
      requestEmbeddingModelSwitch(payload.embedding_model);
      return;
    }
    await saveProfilePatch(payload, false);
  });

  // Delete Profile
  el.btnDeleteProfile.addEventListener("click", async () => {
    if (!state.activeProfile) return;
    const prof = state.activeProfile;
    const ok = window.confirm(
      `Delete profile '${prof.name}' and all of its uploaded documents, saved chats, and vector embeddings?`
    );
    if (!ok) return;
    await apiJson(`/api/profiles/${encodeURIComponent(prof.id)}`, {
      method: "DELETE",
    });
    state.transcripts.delete(prof.id);
    state.thinkingByProfile.delete(prof.id);
    state.savedChatsByProfile.delete(prof.id);
    state.activeSavedChatByProfile.delete(prof.id);
    state.activeProfileId = null;
    state.activeProfile = null;
    await refreshAll(false);
  });

  // Document Uploads (both right drawer & manager modal)
  el.fileUploadInput.addEventListener("change", () => {
    handleFileUpload(el.fileUploadInput.files, { fromModal: false });
  });
  el.modalFileUploadInput.addEventListener("change", () => {
    handleFileUpload(el.modalFileUploadInput.files, { fromModal: true });
  });

  // Pull Model Modal
  el.btnOpenPullModal.addEventListener("click", () => openPullModal(""));
  el.pullModelName.addEventListener("input", () => {
    el.pullConfirmBox.hidden = true;
    el.btnStepConfirmPull.hidden = false;
    el.btnStartPull.hidden = true;
  });
  el.btnStepConfirmPull.addEventListener("click", reviewPullConfirmation);
  el.btnStartPull.addEventListener("click", executeModelPull);
  el.btnCancelPull.addEventListener("click", closeOrCancelPull);
  el.btnClosePullX.addEventListener("click", closeOrCancelPull);
}

// Wire cross-module callbacks and initialize
configurePullModal({ refreshAll });
configureDocuments({ selectProfile });
configureSettings({ renderProfileSidebar, renderMainView, selectProfile });
configureChat({ refreshAll });

setSettingsPanelOpen(false);
bindEvents();
initInfoTips();
refreshAll(false);

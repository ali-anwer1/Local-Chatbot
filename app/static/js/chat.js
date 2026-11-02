/* Chat Transcript Rendering, Streaming, Copy/Retry, Citations, Saved Chats & Export (§4.3, §4.5, #10) */

import { apiJson, streamSSE } from "./api.js";
import { SUPPORTED_LANG_MAP, SUPPORTED_LANGUAGES, SVG_ICONS } from "./constants.js";
import { el } from "./dom.js";
import { formatAssistantMarkdownAndCitations } from "./markdown.js";
import { state } from "./state.js";
import {
  attachCopyFeedback,
  escapeHtml,
  formatShortDate,
  isThinkingEnabled,
  updateInteractiveControlsLock,
} from "./utils.js";

let refreshAllCb = async () => {};

export function configureChat({ refreshAll }) {
  refreshAllCb = refreshAll;
}

export function renderSavedChatsSidebar() {
  el.savedChatList.innerHTML = "";
  const pid = state.activeProfileId;
  if (!pid) {
    el.savedChatsSection.hidden = true;
    return;
  }
  el.savedChatsSection.hidden = false;

  const chats = state.savedChatsByProfile.get(pid) || [];
  const activeSaved = state.activeSavedChatByProfile.get(pid);
  el.savedChatsCount.textContent = String(chats.length);

  if (chats.length === 0) {
    const li = document.createElement("li");
    li.className = "sidebar-empty-hint";
    li.textContent = "No saved chats for this profile.";
    el.savedChatList.appendChild(li);
    return;
  }

  for (const chat of chats) {
    const li = document.createElement("li");
    li.className = `saved-chat-item${activeSaved && activeSaved.id === chat.id ? " active" : ""}`;

    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "saved-chat-open-btn";

    const titleSpan = document.createElement("span");
    titleSpan.className = "saved-chat-title";
    titleSpan.textContent = chat.title;

    const metaSpan = document.createElement("span");
    metaSpan.className = "saved-chat-meta";
    const datePart = formatShortDate(chat.updated_at);
    const turnsCount = chat.turn_count || 0;
    metaSpan.textContent = `${turnsCount} turn${turnsCount === 1 ? "" : "s"}${datePart ? ` · ${datePart}` : ""}`;

    openBtn.appendChild(titleSpan);
    openBtn.appendChild(metaSpan);
    openBtn.addEventListener("click", () => loadSavedChat(pid, chat.id));

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "saved-chat-del-btn";
    delBtn.title = `Delete '${chat.title}'`;
    delBtn.setAttribute("aria-label", `Delete saved chat ${chat.title}`);
    delBtn.innerHTML = `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>`;
    delBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      handleDeleteSavedChat(pid, chat.id, chat.title);
    });

    li.appendChild(openBtn);
    li.appendChild(delBtn);
    el.savedChatList.appendChild(li);
  }
}

export function getTranscript(profileId) {
  if (!state.transcripts.has(profileId)) {
    state.transcripts.set(profileId, []);
  }
  return state.transcripts.get(profileId);
}

export function suggestChatTitle(turns) {
  const firstUser = turns.find((t) => t.role === "user" && t.text);
  if (!firstUser) return "Saved conversation";
  const cleaned = firstUser.text.trim().replace(/\s+/g, " ");
  return cleaned.length > 56 ? `${cleaned.slice(0, 56).trim()}…` : cleaned;
}

export function serializeTurnsForSave(turns) {
  return turns
    .filter((t) => !t.streaming)
    .map((t) => {
      if (t.role === "user") {
        return { role: "user", text: t.text };
      }
      return {
        role: "assistant",
        text: t.text || "",
        refused: Boolean(t.refused),
        conversational: Boolean(t.conversational),
        sources: t.sources || [],
        debugChunks: t.debugChunks || null,
        timings: t.timings || null,
        bestScore: t.bestScore ?? 0,
        threshold: t.threshold ?? 0.5,
        retrievalQuery: t.retrievalQuery || "",
        language: t.language || null,
        translations: t.translations || {},
        activeTranslationLang: t.activeTranslationLang || null,
        error: Boolean(t.error),
        errorDetails: t.errorDetails || null,
      };
    });
}

export function buildHistoryForRequest(turns) {
  const history = [];
  for (const t of turns) {
    if (t.streaming || t.error) continue;
    if (t.role !== "user" && t.role !== "assistant") continue;
    const content = String(t.text || t.content || "").trim();
    if (content) {
      history.push({ role: t.role, content });
    }
  }
  return history;
}

export function updateChatActionButtons(profileId) {
  if (!profileId) return;
  const turns = getTranscript(profileId);
  const hasCompletedTurns = turns.some((t) => !t.streaming);
  const busy = Boolean(state.chatAbortController);

  el.btnSaveChat.disabled = !hasCompletedTurns || busy;
  el.btnExportChat.disabled = !hasCompletedTurns;

  const activeSaved = state.activeSavedChatByProfile.get(profileId);
  if (activeSaved) {
    el.activeChatBadge.hidden = false;
    el.activeChatBadge.textContent = "Saved";
    el.activeChatBadge.title = `Saved as: ${activeSaved.title}`;
    el.btnSaveChatLabel.textContent = "Saved ✓";
  } else {
    el.activeChatBadge.hidden = true;
    el.btnSaveChatLabel.textContent = "Save chat";
  }
}

export function openSaveChatDialog() {
  if (!state.activeProfile) return;
  const pid = state.activeProfile.id;
  const turns = getTranscript(pid);
  if (turns.length === 0) return;

  const activeSaved = state.activeSavedChatByProfile.get(pid);
  el.saveChatTitle.value = activeSaved ? activeSaved.title : suggestChatTitle(turns);
  el.dialogSaveChat.showModal();
  el.saveChatTitle.select();
}

export async function persistSavedChat(profileId, titleOverride = null) {
  const turns = getTranscript(profileId);
  const serialized = serializeTurnsForSave(turns);
  if (serialized.length === 0) return null;

  const activeSaved = state.activeSavedChatByProfile.get(profileId);
  const saved = await apiJson(
    `/api/profiles/${encodeURIComponent(profileId)}/chats`,
    {
      method: "POST",
      body: JSON.stringify({
        id: activeSaved ? activeSaved.id : null,
        title: titleOverride ?? (activeSaved ? activeSaved.title : suggestChatTitle(turns)),
        turns: serialized,
      }),
    }
  );

  state.activeSavedChatByProfile.set(profileId, {
    id: saved.id,
    title: saved.title,
  });

  const chatsData = await apiJson(
    `/api/profiles/${encodeURIComponent(profileId)}/chats`
  );
  state.savedChatsByProfile.set(profileId, chatsData.chats || []);
  renderSavedChatsSidebar();
  updateChatActionButtons(profileId);
  return saved;
}

export async function loadSavedChat(profileId, chatId) {
  if (state.chatAbortController) {
    state.chatAbortController.abort();
  }
  try {
    const saved = await apiJson(
      `/api/profiles/${encodeURIComponent(profileId)}/chats/${encodeURIComponent(chatId)}`
    );
    const restoredTurns = (saved.turns || []).map((t) => ({
      ...t,
      candidateSources: t.sources || [],
      streaming: false,
    }));
    state.transcripts.set(profileId, restoredTurns);
    state.activeSavedChatByProfile.set(profileId, {
      id: saved.id,
      title: saved.title,
    });
    renderSavedChatsSidebar();
    renderConversationTranscript(profileId);
    updateChatActionButtons(profileId);
  } catch (err) {
    alert(err.message);
  }
}

export async function handleDeleteSavedChat(profileId, chatId, title) {
  const ok = window.confirm(`Delete saved chat '${title}'?`);
  if (!ok) return;
  try {
    await apiJson(
      `/api/profiles/${encodeURIComponent(profileId)}/chats/${encodeURIComponent(chatId)}`,
      { method: "DELETE" }
    );
    const activeSaved = state.activeSavedChatByProfile.get(profileId);
    if (activeSaved && activeSaved.id === chatId) {
      state.activeSavedChatByProfile.set(profileId, null);
    }
    const chatsData = await apiJson(
      `/api/profiles/${encodeURIComponent(profileId)}/chats`
    );
    state.savedChatsByProfile.set(profileId, chatsData.chats || []);
    renderSavedChatsSidebar();
    updateChatActionButtons(profileId);
  } catch (err) {
    alert(err.message);
  }
}

export function startNewChat() {
  if (!state.activeProfile) return;
  if (state.chatAbortController) {
    state.chatAbortController.abort();
  }
  const pid = state.activeProfile.id;
  state.transcripts.set(pid, []);
  state.activeSavedChatByProfile.set(pid, null);
  if (el.selectAnswerLanguage) {
    el.selectAnswerLanguage.value = "auto";
  }
  renderSavedChatsSidebar();
  renderConversationTranscript(pid);
  updateChatActionButtons(pid);
  if (!el.chatInput.disabled) {
    el.chatInput.focus();
  }
}

export function exportActiveChatAsMarkdown() {
  if (!state.activeProfile) return;
  const prof = state.activeProfile;
  const turns = getTranscript(prof.id).filter((t) => !t.streaming);
  if (turns.length === 0) return;

  const activeSaved = state.activeSavedChatByProfile.get(prof.id);
  const title = activeSaved ? activeSaved.title : suggestChatTitle(turns);
  const lines = [
    `# ${title}`,
    "",
    `- **Profile:** ${prof.name}`,
    `- **Model:** ${prof.chat_model}`,
    `- **Exported:** ${new Date().toLocaleString()}`,
    "",
    "---",
    "",
  ];

  for (const turn of turns) {
    if (turn.role === "user") {
      lines.push("### User", "", turn.text, "");
    } else {
      lines.push("### Assistant", "", turn.text || "_[No response]_", "");
      if (!turn.refused && Array.isArray(turn.sources) && turn.sources.length > 0) {
        lines.push("**Cited sources:**");
        for (const src of turn.sources) {
          const loc = src.location_label ? `, ${src.location_label}` : "";
          lines.push(`- \`[${src.index}]\` ${src.source_file}${loc}`);
        }
        lines.push("");
      }
      if (turn.translations && typeof turn.translations === "object") {
        const transEntries = Object.entries(turn.translations);
        if (transEntries.length > 0) {
          lines.push("**Translations:**", "");
          for (const [code, transText] of transEntries) {
            const langMeta = SUPPORTED_LANG_MAP[code];
            const langLabel = langMeta ? `${langMeta.nativeName} (${langMeta.name})` : code;
            lines.push(`#### ${langLabel}`, "", transText, "");
          }
        }
      }
      lines.push("---", "");
    }
  }

  const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const slug =
    title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 48) || "chat-export";
  const link = document.createElement("a");
  link.href = url;
  link.download = `${slug}.md`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function openCitationModal(source) {
  const idx = source.index ? `[${source.index}] ` : "";
  el.citationModalTitle.textContent = `${idx}${source.source_file}`;
  el.citationModalScore.textContent = `Score: ${Number(source.score).toFixed(2)}`;
  const loc = source.location_label ? ` · ${source.location_label}` : "";
  el.citationModalMeta.textContent = `${source.source_file}${loc} (chunk #${source.chunk_index})`;
  el.citationModalText.textContent = source.text || "";
  el.dialogCitation.showModal();
}

function formatDurationMs(ms) {
  if (ms === null || ms === undefined || Number.isNaN(Number(ms))) {
    return "—";
  }
  const val = Number(ms);
  return val < 100 ? `${val.toFixed(1)} ms` : `${Math.round(val)} ms`;
}

function buildDebugTimingsDOM(timings) {
  const box = document.createElement("div");
  box.className = "debug-timings";

  const pipelineRow = document.createElement("div");
  pipelineRow.className = "debug-timing-row";
  const pipelineItems = [
    ["Embed", formatDurationMs(timings.embed_ms)],
    ["Retrieval", formatDurationMs(timings.retrieval_ms)],
    ["First token", formatDurationMs(timings.time_to_first_token_ms)],
    ["Total", formatDurationMs(timings.total_ms)],
  ];
  for (const [label, value] of pipelineItems) {
    const item = document.createElement("span");
    item.className = "debug-timing-item";
    item.innerHTML = `<span class="debug-timing-label">${escapeHtml(label)}:</span> <code>${escapeHtml(value)}</code>`;
    pipelineRow.appendChild(item);
  }
  box.appendChild(pipelineRow);

  const hasPromptCount =
    timings.prompt_eval_count !== undefined && timings.prompt_eval_count !== null;
  const hasPromptDur =
    timings.prompt_eval_duration !== undefined && timings.prompt_eval_duration !== null;
  const hasEvalCount =
    timings.eval_count !== undefined && timings.eval_count !== null;
  const hasEvalDur =
    timings.eval_duration !== undefined && timings.eval_duration !== null;

  if (hasPromptCount || hasPromptDur || hasEvalCount || hasEvalDur) {
    const llmRow = document.createElement("div");
    llmRow.className = "debug-timing-row";

    const promptMs =
      timings.prompt_eval_ms ??
      (hasPromptDur ? Number(timings.prompt_eval_duration) / 1e6 : null);
    const tps =
      timings.tokens_per_second ??
      (hasEvalCount && hasEvalDur && Number(timings.eval_duration) > 0
        ? Number(timings.eval_count) / (Number(timings.eval_duration) / 1e9)
        : null);

    const llmItems = [];
    if (hasPromptCount) {
      llmItems.push(["Prompt tokens", String(timings.prompt_eval_count)]);
    }
    if (promptMs !== null) {
      llmItems.push(["Prompt eval", formatDurationMs(promptMs)]);
    }
    if (hasEvalCount) {
      llmItems.push(["Generated tokens", String(timings.eval_count)]);
    }
    if (tps !== null) {
      llmItems.push(["Speed", `${Number(tps).toFixed(1)} tok/s`]);
    }

    for (const [label, value] of llmItems) {
      const item = document.createElement("span");
      item.className = "debug-timing-item";
      item.innerHTML = `<span class="debug-timing-label">${escapeHtml(label)}:</span> <code>${escapeHtml(value)}</code>`;
      llmRow.appendChild(item);
    }
    box.appendChild(llmRow);
  }

  return box;
}

function createDebugElement(turn) {
  const wrap = document.createElement("div");
  wrap.className = "debug-panel";
  wrap.hidden = !state.debugEnabled;

  const header = document.createElement("div");
  header.className = "debug-header";
  const threshold = Number(turn.threshold ?? 0.5).toFixed(2);
  const best = Number(turn.bestScore ?? 0).toFixed(2);
  const modeNote = turn.conversational ? " · Conversational intent" : "";
  header.innerHTML = `<span>Retrieval candidates (${(turn.debugChunks || []).length}${modeNote})</span><span class="mono-num">Best: ${best} / Cutoff: ${threshold}</span>`;
  wrap.appendChild(header);

  if (turn.timings && typeof turn.timings === "object") {
    wrap.appendChild(buildDebugTimingsDOM(turn.timings));
  }

  if (turn.retrievalQuery) {
    const qLine = document.createElement("div");
    qLine.className = "helper-text";
    qLine.textContent = `Query: "${turn.retrievalQuery}"`;
    wrap.appendChild(qLine);
  }

  if (turn.language && typeof turn.language === "object") {
    const langLine = document.createElement("div");
    langLine.className = "helper-text debug-language-line";
    const det = turn.language.detected_language || "none";
    const conf =
      turn.language.language_confidence !== undefined &&
      turn.language.language_confidence !== null
        ? Number(turn.language.language_confidence).toFixed(2)
        : "—";
    const target = turn.language.target_language || "en";
    const name = turn.language.language_name || target;
    const rule = turn.language.language_rule || "fallback";
    langLine.innerHTML = `<strong>Language:</strong> ${escapeHtml(target)} (${escapeHtml(name)}) · <strong>Detected:</strong> ${escapeHtml(det)} (${escapeHtml(conf)}) · <strong>Rule:</strong> <code>${escapeHtml(rule)}</code>`;
    wrap.appendChild(langLine);
  }

  const chunks = turn.debugChunks || [];
  if (chunks.length === 0) {
    const empty = document.createElement("div");
    empty.className = "helper-text";
    empty.textContent = turn.conversational
      ? "Conversational query answered using profile overview."
      : "No chunks retrieved from knowledge base.";
    wrap.appendChild(empty);
    return wrap;
  }

  for (const c of chunks) {
    const det = document.createElement("details");
    det.className = "debug-chunk-item";

    const sum = document.createElement("summary");
    const loc = c.location_label ? ` (${c.location_label})` : "";
    const labelSpan = document.createElement("span");
    labelSpan.textContent = `[${c.index}] ${c.source_file}${loc}`;

    const badgeGroup = document.createElement("span");
    badgeGroup.className = "debug-badges";

    if (c.cited) {
      const citedBadge = document.createElement("span");
      citedBadge.className = "score-tag cited";
      citedBadge.textContent = "CITED";
      badgeGroup.appendChild(citedBadge);
    }

    const scoreBadge = document.createElement("span");
    const passed = Number(c.score) >= Number(turn.threshold ?? 0.5);
    scoreBadge.className = `score-tag ${passed ? "pass" : "fail"}`;
    scoreBadge.textContent = `${Number(c.score).toFixed(3)} ${passed ? "PASS" : "BELOW"}`;
    badgeGroup.appendChild(scoreBadge);

    sum.appendChild(labelSpan);
    sum.appendChild(badgeGroup);
    det.appendChild(sum);

    const pre = document.createElement("pre");
    pre.tabIndex = 0;
    pre.textContent = c.text;
    det.appendChild(pre);

    wrap.appendChild(det);
  }
  return wrap;
}

function buildUserTurnDOM(turn) {
  const row = document.createElement("article");
  row.className = "msg-row user";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  const sender = document.createElement("div");
  sender.className = "msg-sender-label";
  sender.textContent = "You";
  bubble.appendChild(sender);

  const textDiv = document.createElement("div");
  textDiv.className = "msg-text";
  textDiv.textContent = turn.text;
  bubble.appendChild(textDiv);

  row.appendChild(bubble);

  const actions = document.createElement("div");
  actions.className = "turn-actions";

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "action-chip-btn";
  copyBtn.title = "Copy query";
  copyBtn.setAttribute("aria-label", "Copy query");
  attachCopyFeedback(copyBtn, () => turn.text, "Copy");
  actions.appendChild(copyBtn);

  row.appendChild(actions);
  return row;
}

function isChatScrolledNearBottom() {
  const remaining =
    el.chatMessages.scrollHeight - el.chatMessages.scrollTop - el.chatMessages.clientHeight;
  return remaining < 96;
}

export function buildTurnSourcesMap(turn) {
  const lookupList =
    turn.candidateSources && turn.candidateSources.length > 0
      ? turn.candidateSources
      : turn.sources || [];
  const sourcesMap = new Map();
  for (const s of lookupList) {
    sourcesMap.set(Number(s.index), s);
  }
  return sourcesMap;
}

function createAssistantTurnScaffold(
  turnIndex,
  { refused = false, error = false, createRefusalTag = refused, bubbleHidden = false } = {}
) {
  const row = document.createElement("article");
  row.className = `msg-row assistant${refused ? " refused" : ""}${error ? " error" : ""}`;
  row.dataset.turnIdx = String(turnIndex);

  const body = document.createElement("div");
  body.className = "assistant-body";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.hidden = bubbleHidden;

  let refusalTag = null;
  if (createRefusalTag) {
    refusalTag = document.createElement("div");
    refusalTag.className = "refusal-tag";
    refusalTag.textContent = "Outside knowledge base";
    refusalTag.hidden = !refused;
    bubble.appendChild(refusalTag);
  }

  const contentDiv = document.createElement("div");
  contentDiv.className = "msg-content";
  bubble.appendChild(contentDiv);

  body.appendChild(bubble);
  row.appendChild(body);

  return { row, body, bubble, refusalTag, contentDiv };
}

function createStreamingAssistantTurnView(turn, turnIndex) {
  const { row, body, bubble, refusalTag, contentDiv } = createAssistantTurnScaffold(
    turnIndex,
    { createRefusalTag: true, bubbleHidden: true }
  );

  const liveRow = document.createElement("div");
  liveRow.className = "live-status-row";
  liveRow.setAttribute("role", "status");

  const dot = document.createElement("span");
  dot.className = "live-status-dot";
  dot.setAttribute("aria-hidden", "true");

  const liveLabel = document.createElement("span");
  liveLabel.textContent = turn.statusLabel || "Searching knowledge base…";

  liveRow.appendChild(dot);
  liveRow.appendChild(liveLabel);
  body.insertBefore(liveRow, bubble);

  let lastStatus = liveLabel.textContent;
  let lastText = "";
  let lastSourceCount = -1;
  let lastRefused = false;

  function sync() {
    const refused = Boolean(turn.refused);
    if (refused !== lastRefused) {
      row.classList.toggle("refused", refused);
      refusalTag.hidden = !refused;
      lastRefused = refused;
    }

    if (!turn.text) {
      const nextStatus = turn.statusLabel || "Searching knowledge base…";
      if (liveRow.hidden) {
        liveRow.hidden = false;
      }
      if (!bubble.hidden) {
        bubble.hidden = true;
      }
      if (nextStatus !== lastStatus) {
        liveLabel.textContent = nextStatus;
        lastStatus = nextStatus;
      }
      return;
    }

    // First token arrived: hide live status pill and stream in-place inside contentDiv
    if (!liveRow.hidden) {
      liveRow.hidden = true;
    }
    if (bubble.hidden) {
      bubble.hidden = false;
    }

    const sourcesMap = buildTurnSourcesMap(turn);
    if (turn.text !== lastText || sourcesMap.size !== lastSourceCount) {
      contentDiv.innerHTML = formatAssistantMarkdownAndCitations(turn.text, sourcesMap);
      lastText = turn.text;
      lastSourceCount = sourcesMap.size;
    }
  }

  sync();
  const view = { row, sync };
  turn._liveView = view;
  return view;
}

export function createTranslationBlockDOM({
  profileId,
  turn,
  langCode,
  initialText = "",
  row,
  streaming = false,
}) {
  const block = document.createElement("details");
  block.className = "translation-block";
  block.open = true;
  block.dataset.transLang = langCode;

  const langMeta = SUPPORTED_LANG_MAP[langCode] || { nativeName: langCode, name: langCode };
  const summary = document.createElement("summary");
  summary.className = "translation-header";

  const labelSpan = document.createElement("span");
  labelSpan.className = "translation-lang-label";
  labelSpan.textContent = `${langMeta.nativeName} ▾`;
  summary.appendChild(labelSpan);

  const headerActions = document.createElement("div");
  headerActions.className = "translation-header-actions";

  // "Hide original" / "Show original" toggle button
  const toggleOrigBtn = document.createElement("button");
  toggleOrigBtn.type = "button";
  toggleOrigBtn.className = "action-chip-btn tiny toggle-original-btn";
  const contentDiv = row ? row.querySelector(".msg-content") : null;
  toggleOrigBtn.textContent = contentDiv && contentDiv.hidden ? "Show original" : "Hide original";
  toggleOrigBtn.title = "Toggle original response visibility";
  toggleOrigBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (contentDiv) {
      contentDiv.hidden = !contentDiv.hidden;
      toggleOrigBtn.textContent = contentDiv.hidden ? "Show original" : "Hide original";
    }
  });
  headerActions.appendChild(toggleOrigBtn);

  // Copy translation button
  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "action-chip-btn tiny copy-trans-btn";
  copyBtn.title = "Copy translation";
  copyBtn.setAttribute("aria-label", "Copy translation");
  attachCopyFeedback(copyBtn, () => (turn.translations && turn.translations[langCode]) || "", "Copy");
  headerActions.appendChild(copyBtn);

  summary.appendChild(headerActions);
  block.appendChild(summary);

  const bodyDiv = document.createElement("div");
  bodyDiv.className = "translation-body";
  const sourcesMap = buildTurnSourcesMap(turn);
  if (streaming && !initialText) {
    bodyDiv.innerHTML = `<span class="translating-indicator">Translating into ${escapeHtml(langMeta.nativeName)}…</span>`;
  } else {
    bodyDiv.innerHTML = formatAssistantMarkdownAndCitations(initialText, sourcesMap);
  }
  block.appendChild(bodyDiv);

  return { block, bodyDiv, toggleOrigBtn, copyBtn };
}

let activeTranslateTrigger = null;
let activeTranslateTurn = null;
let activeTranslateProfileId = null;
let activeTranslateRow = null;

export function getOrCreateTranslatePopover() {
  let popover = document.getElementById("translate-dropdown-popover");
  if (!popover) {
    popover = document.createElement("div");
    popover.id = "translate-dropdown-popover";
    popover.className = "translate-dropdown-menu translate-popover";
    popover.setAttribute("role", "menu");
    popover.setAttribute("hidden", "");
    try {
      popover.setAttribute("popover", "auto");
    } catch {
      // Ignore if unsupported
    }
    document.body.appendChild(popover);
  }
  return popover;
}

export function isTranslatePopoverOpen(popover = getOrCreateTranslatePopover()) {
  if (!popover) return false;
  if (typeof popover.matches === "function" && popover.matches(":popover-open")) {
    return true;
  }
  return popover.classList.contains("fallback-open") && !popover.hasAttribute("hidden");
}

export function closeTranslatePopover() {
  const popover = document.getElementById("translate-dropdown-popover");
  if (!popover) return;
  if (typeof popover.hidePopover === "function") {
    try {
      popover.hidePopover();
    } catch {
      // Ignored if already closed
    }
  }
  popover.classList.remove("fallback-open");
  popover.setAttribute("hidden", "");
  if (activeTranslateTrigger) {
    activeTranslateTrigger.setAttribute("aria-expanded", "false");
    activeTranslateTrigger = null;
  }
  activeTranslateTurn = null;
  activeTranslateProfileId = null;
  activeTranslateRow = null;
}

export function positionTranslatePopover(trigger, popover) {
  const triggerRect = trigger.getBoundingClientRect();
  const popoverRect = popover.getBoundingClientRect();

  const popoverWidth = popoverRect.width || 175;
  const popoverHeight = popoverRect.height || 240;

  const viewportHeight = window.innerHeight;
  const viewportWidth = window.innerWidth;
  const gap = 4;

  const spaceBelow = viewportHeight - triggerRect.bottom;
  const spaceAbove = triggerRect.top;

  // Flip upward if space below is limited (e.g. near composer or bottom of screen - Issue 3)
  const flipUp = spaceBelow < (popoverHeight + gap + 10) && spaceAbove > spaceBelow;

  let top;
  if (flipUp) {
    top = Math.max(8, triggerRect.top - popoverHeight - gap);
  } else {
    top = Math.min(viewportHeight - popoverHeight - 8, triggerRect.bottom + gap);
  }

  // Align horizontally with trigger button, clamped inside viewport
  let left = triggerRect.left;
  const minLeft = 8;
  const maxLeft = viewportWidth - popoverWidth - 8;
  if (left < minLeft) {
    left = minLeft;
  } else if (left > maxLeft) {
    left = maxLeft;
  }

  popover.style.top = `${Math.round(top)}px`;
  popover.style.left = `${Math.round(left)}px`;
}

export function openTranslateMenu(triggerBtn, turn, profileId, row) {
  const popover = getOrCreateTranslatePopover();
  if (!popover) return;

  // Toggle close if clicking the currently open trigger
  if (activeTranslateTrigger === triggerBtn && isTranslatePopoverOpen(popover)) {
    closeTranslatePopover();
    return;
  }

  closeTranslatePopover();

  activeTranslateTrigger = triggerBtn;
  activeTranslateTurn = turn;
  activeTranslateProfileId = profileId;
  activeTranslateRow = row;

  triggerBtn.setAttribute("aria-expanded", "true");

  popover.innerHTML = "";
  const currentLang = turn.language?.target_language || "en";
  const activeTrans = turn.activeTranslationLang || null;
  const availableLangs = SUPPORTED_LANGUAGES.filter(
    (l) => l.code !== currentLang && l.code !== activeTrans
  );

  for (const lang of availableLangs) {
    const itemBtn = document.createElement("button");
    itemBtn.type = "button";
    itemBtn.className = "translate-menu-item";
    itemBtn.setAttribute("role", "menuitem");
    itemBtn.dataset.langCode = lang.code;

    const nativeSpan = document.createElement("span");
    nativeSpan.className = "menu-item-native";
    nativeSpan.textContent = lang.nativeName;
    itemBtn.appendChild(nativeSpan);

    if (lang.name && lang.name !== lang.nativeName) {
      const latinSpan = document.createElement("span");
      latinSpan.className = "menu-item-latin";
      latinSpan.textContent = lang.name;
      itemBtn.appendChild(latinSpan);
    }

    itemBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      closeTranslatePopover();
      triggerTranslation(profileId, turn, lang.code, row);
    });
    popover.appendChild(itemBtn);
  }

  popover.removeAttribute("hidden");
  if (typeof popover.showPopover === "function") {
    try {
      popover.showPopover();
    } catch {
      // Ignored if already open
    }
  } else {
    popover.classList.add("fallback-open");
  }

  positionTranslatePopover(triggerBtn, popover);
}

export async function triggerTranslation(profileId, turnOrIndex, targetLang, row) {
  let turn = null;
  let turnIndex = -1;
  const turns = getTranscript(profileId);

  if (typeof turnOrIndex === "object" && turnOrIndex !== null) {
    turn = turnOrIndex;
    turnIndex = turns.indexOf(turn);
  } else {
    turnIndex = Number(turnOrIndex);
    turn = turns[turnIndex];
  }

  if (!turn && row && row.dataset.turnIdx !== undefined) {
    const idx = Number(row.dataset.turnIdx);
    if (!Number.isNaN(idx) && turns[idx]) {
      turn = turns[idx];
      turnIndex = idx;
    }
  }

  if (!turn || !turn.text) return;

  turn.translations = turn.translations || {};

  // Find assistant turn body (Issue 1 fix: assistant turns use .assistant-body, not .msg-body)
  const msgBody = row ? (row.querySelector(".assistant-body") || row) : null;
  if (!msgBody) return;

  let container = row.querySelector(".translations-container");
  if (!container) {
    container = document.createElement("div");
    container.className = "translations-container";
    const actions = row.querySelector(".turn-actions");
    if (actions && actions.parentNode === msgBody) {
      msgBody.insertBefore(container, actions);
    } else {
      msgBody.appendChild(container);
    }
  }

  // Cancel any prior in-flight translation on this row
  if (row._translationAbortController) {
    row._translationAbortController.abort();
    row._translationAbortController = null;
  }

  turn.activeTranslationLang = targetLang;

  // Cache check: if already translated, overwrite existing block with cached translation
  if (turn.translations[targetLang] !== undefined) {
    container.innerHTML = "";
    const { block } = createTranslationBlockDOM({
      profileId,
      turn,
      langCode: targetLang,
      initialText: turn.translations[targetLang],
      row,
      streaming: false,
    });
    container.appendChild(block);
    block.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }

  // Overwrite previous translation with new translation block in streaming state
  container.innerHTML = "";
  const { block, bodyDiv } = createTranslationBlockDOM({
    profileId,
    turn,
    langCode: targetLang,
    initialText: "",
    row,
    streaming: true,
  });
  container.appendChild(block);
  block.scrollIntoView({ behavior: "smooth", block: "nearest" });

  const abortCtrl = new AbortController();
  row._translationAbortController = abortCtrl;

  let fullText = "";
  let hadError = false;
  const sourcesMap = buildTurnSourcesMap(turn);
  try {
    await streamSSE(
      "/api/translate",
      {
        text: turn.text,
        source_language: turn.language?.target_language || "en",
        target_language: targetLang,
        profile_id: profileId,
      },
      {
        signal: abortCtrl.signal,
        onEvent: (evType, payload) => {
          if (evType === "token") {
            fullText += payload.text || "";
            bodyDiv.innerHTML = formatAssistantMarkdownAndCitations(fullText, sourcesMap);
          } else if (evType === "done") {
            fullText = payload.full_text ?? fullText;
            turn.translations[targetLang] = fullText;
            bodyDiv.innerHTML = formatAssistantMarkdownAndCitations(fullText, sourcesMap);
          } else if (evType === "error") {
            hadError = true;
            bodyDiv.innerHTML = `<span class="helper-text error-text">${escapeHtml(payload.message || "Translation error.")}</span>`;
          }
        },
      }
    );

    if (!hadError && fullText) {
      turn.translations[targetLang] = fullText;
      bodyDiv.innerHTML = formatAssistantMarkdownAndCitations(fullText, sourcesMap);
    }

    if (state.activeSavedChatByProfile.get(profileId)) {
      await persistSavedChat(profileId).catch(() => {});
    }
  } catch (err) {
    if (abortCtrl.signal.aborted) {
      return;
    }
    bodyDiv.innerHTML = `<span class="helper-text error-text">${escapeHtml(err.message || "Translation request failed.")}</span>`;
  } finally {
    if (row._translationAbortController === abortCtrl) {
      row._translationAbortController = null;
    }
  }
}

function buildAssistantTurnDOM(turn, turnIndex, profileId) {
  if (turn.streaming) {
    return createStreamingAssistantTurnView(turn, turnIndex).row;
  }

  const isEmptyCompleted = !turn.streaming && !turn.error && (!turn.text || !turn.text.trim());
  const isErrorLike = Boolean(turn.error || isEmptyCompleted);
  const { row, body, contentDiv } = createAssistantTurnScaffold(turnIndex, {
    refused: Boolean(turn.refused),
    error: isErrorLike,
    createRefusalTag: !isErrorLike && Boolean(turn.refused),
  });

  // Error or empty generation state with prominent retry button (#4)
  if (isErrorLike) {
    const p = document.createElement("p");
    p.textContent =
      turn.text && turn.text.trim()
        ? turn.text
        : "The model returned an empty response. Click Retry to try generating an answer again.";
    contentDiv.appendChild(p);

    if (turn.errorDetails) {
      const det = document.createElement("details");
      det.className = "msg-error-details";
      const sum = document.createElement("summary");
      sum.textContent = "Technical details";
      const pre = document.createElement("pre");
      pre.tabIndex = 0;
      const code = document.createElement("code");
      code.textContent = turn.errorDetails;
      pre.appendChild(code);
      det.appendChild(sum);
      det.appendChild(pre);
      contentDiv.appendChild(det);
    }

    const retryBtn = document.createElement("button");
    retryBtn.type = "button";
    retryBtn.className = "action-chip-btn retry-prominent";
    retryBtn.innerHTML = `${SVG_ICONS.retry}<span>Retry query</span>`;
    retryBtn.addEventListener("click", () => retryAssistantTurn(profileId, turnIndex));
    contentDiv.appendChild(retryBtn);

    return row;
  }

  // Normal or Refused Response Content
  const sourcesMap = buildTurnSourcesMap(turn);
  contentDiv.innerHTML = formatAssistantMarkdownAndCitations(turn.text, sourcesMap);

  // Compact Sources list under the answer (filename + chunk number in mono)
  if (!turn.refused && Array.isArray(turn.sources) && turn.sources.length > 0) {
    const sourcesBar = document.createElement("div");
    sourcesBar.className = "sources-bar";

    const lbl = document.createElement("div");
    lbl.className = "sources-label";
    lbl.textContent = "Sources";
    sourcesBar.appendChild(lbl);

    const sourcesList = document.createElement("ul");
    sourcesList.className = "sources-list";

    for (const src of turn.sources) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "source-item-btn citation-chip";
      btn.dataset.citeIdx = String(src.index);
      const loc = src.location_label ? ` · ${src.location_label}` : "";
      btn.innerHTML = `<span class="source-index">[${escapeHtml(String(src.index))}]</span> <span class="source-filename">${escapeHtml(src.source_file)}</span> <span class="source-chunk-meta">chunk #${src.chunk_index}${escapeHtml(loc)}</span>`;
      btn.addEventListener("click", () => openCitationModal(src));
      li.appendChild(btn);
      sourcesList.appendChild(li);
    }
    sourcesBar.appendChild(sourcesList);
    body.appendChild(sourcesBar);
  }

  // Completed Assistant Turn Action Bar: Copy, Retry & Translate (§ Part 2)
  if (turn.text) {
    // Render existing translation if present (single overwritten translation view)
    if (turn.translations && typeof turn.translations === "object") {
      const transEntries = Object.entries(turn.translations).filter(([_, text]) => Boolean(text));
      if (transEntries.length > 0) {
        const activeCode =
          turn.activeTranslationLang && turn.translations[turn.activeTranslationLang]
            ? turn.activeTranslationLang
            : transEntries[transEntries.length - 1][0];
        const activeText = turn.translations[activeCode];

        if (activeText) {
          const container = document.createElement("div");
          container.className = "translations-container";
          const { block } = createTranslationBlockDOM({
            profileId,
            turn,
            langCode: activeCode,
            initialText: activeText,
            row,
            streaming: false,
          });
          container.appendChild(block);
          body.appendChild(container);
        }
      }
    }

    const actions = document.createElement("div");
    actions.className = "turn-actions";

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "action-chip-btn";
    copyBtn.title = "Copy response";
    copyBtn.setAttribute("aria-label", "Copy response");
    attachCopyFeedback(copyBtn, () => turn.text, "Copy");
    actions.appendChild(copyBtn);

    const retryBtn = document.createElement("button");
    retryBtn.type = "button";
    retryBtn.className = "action-chip-btn";
    retryBtn.title = "Regenerate response";
    retryBtn.setAttribute("aria-label", "Retry query");
    retryBtn.innerHTML = `${SVG_ICONS.retry}<span>Retry</span>`;
    retryBtn.addEventListener("click", () => retryAssistantTurn(profileId, turnIndex));
    actions.appendChild(retryBtn);

    // Translate control (Shared Top-Layer Popover § Part 2)
    const translateWrap = document.createElement("div");
    translateWrap.className = "translate-menu-wrap";

    const translateBtn = document.createElement("button");
    translateBtn.type = "button";
    translateBtn.className = "action-chip-btn translate-btn";
    translateBtn.title = "Translate response";
    translateBtn.setAttribute("aria-label", "Translate response");
    translateBtn.setAttribute("aria-haspopup", "menu");
    translateBtn.setAttribute("aria-expanded", "false");
    translateBtn.innerHTML = `${SVG_ICONS.translate}<span>Translate ▾</span>`;
    translateBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      openTranslateMenu(translateBtn, turn, profileId, row);
    });

    translateWrap.appendChild(translateBtn);
    actions.appendChild(translateWrap);

    body.appendChild(actions);
  }

  if (
    (turn.debugChunks !== null && turn.debugChunks !== undefined) ||
    turn.timings
  ) {
    body.appendChild(createDebugElement(turn));
  }

  return row;
}

export function renderConversationTranscript(profileId) {
  const turns = getTranscript(profileId);
  el.chatMessages.innerHTML = "";
  if (turns.length === 0) {
    el.chatMessages.appendChild(el.chatWelcome);
    el.chatWelcome.hidden = false;
    return;
  }

  el.chatWelcome.hidden = true;
  const frag = document.createDocumentFragment();
  turns.forEach((turn, idx) => {
    if (turn.role === "user") {
      frag.appendChild(buildUserTurnDOM(turn));
    } else {
      frag.appendChild(buildAssistantTurnDOM(turn, idx, profileId));
    }
  });
  el.chatMessages.appendChild(frag);
  el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
}

function finalizeStreamingTurnInDOM(profileId, assistantTurn, assistantTurnIndex) {
  const liveView = assistantTurn._liveView;
  assistantTurn._liveView = null;
  if (state.activeProfileId !== profileId) return;

  const stickToBottom = isChatScrolledNearBottom();
  if (liveView && liveView.row.isConnected) {
    const completedRow = buildAssistantTurnDOM(assistantTurn, assistantTurnIndex, profileId);
    liveView.row.replaceWith(completedRow);
  } else {
    renderConversationTranscript(profileId);
  }
  if (stickToBottom) {
    el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
  }
}

export async function retryAssistantTurn(profileId, assistantIndex) {
  if (!state.activeProfile || state.activeProfile.id !== profileId) return;
  if (state.chatAbortController) return;

  const turns = getTranscript(profileId);
  let userIndex = -1;
  for (let i = assistantIndex - 1; i >= 0; i -= 1) {
    if (turns[i] && turns[i].role === "user") {
      userIndex = i;
      break;
    }
  }
  if (userIndex < 0) return;

  const queryText = turns[userIndex].text;

  // Remove retried/failed turns from transcript and DOM
  turns.splice(userIndex);
  if (state.activeProfileId === profileId && !el.chatWelcome.isConnected) {
    while (el.chatMessages.children.length > userIndex) {
      el.chatMessages.lastElementChild.remove();
    }
  }
  await executeChatQuery(queryText);
}

export async function handleSendChat(event) {
  event.preventDefault();
  if (!state.activeProfile || state.chatAbortController) return;

  const message = el.chatInput.value.trim();
  if (!message) return;

  el.chatInput.value = "";
  el.chatInput.style.height = "auto";
  await executeChatQuery(message);
}

export async function executeChatQuery(message) {
  if (!state.activeProfile || state.chatAbortController) return;

  const profileId = state.activeProfile.id;
  const thinking = isThinkingEnabled(profileId);
  const turns = getTranscript(profileId);
  const history = buildHistoryForRequest(turns);

  const userTurn = { role: "user", text: message };
  turns.push(userTurn);

  const assistantTurn = {
    role: "assistant",
    text: "",
    statusLabel: thinking
      ? "Searching knowledge base (Thinking enabled)…"
      : "Searching knowledge base…",
    refused: false,
    conversational: false,
    sources: [],
    candidateSources: [],
    debugChunks: null,
    timings: null,
    bestScore: 0,
    threshold: state.activeProfile.similarity_threshold,
    retrievalQuery: "",
    language: null,
    streaming: true,
    error: false,
    errorDetails: null,
    _liveView: null,
  };
  const assistantTurnIndex = turns.length;
  turns.push(assistantTurn);

  // Incrementally append ONLY the new user & assistant nodes
  if (state.activeProfileId === profileId) {
    if (el.chatWelcome.parentNode === el.chatMessages) {
      el.chatWelcome.hidden = true;
      el.chatWelcome.remove();
    }
    el.chatMessages.appendChild(buildUserTurnDOM(userTurn));
    const liveView = createStreamingAssistantTurnView(assistantTurn, assistantTurnIndex);
    el.chatMessages.appendChild(liveView.row);
    el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
  }
  updateChatActionButtons(profileId);

  const abortController = new AbortController();
  state.chatAbortController = abortController;
  el.btnStopStream.hidden = false;
  el.btnSendChat.disabled = true;

  const requestedLanguage = el.selectAnswerLanguage
    ? el.selectAnswerLanguage.value
    : "auto";
  const requestBody = {
    message,
    history,
    requested_language: requestedLanguage,
    options: { think: thinking },
  };

  let rafId = null;
  const flushLiveTurn = () => {
    rafId = null;
    if (state.activeProfileId !== profileId) return;
    const liveView = assistantTurn._liveView;
    if (!liveView || !liveView.row.isConnected) return;
    const stickToBottom = isChatScrolledNearBottom();
    liveView.sync();
    if (stickToBottom) {
      el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
    }
  };
  const scheduleLiveTurnFlush = () => {
    if (rafId !== null || state.activeProfileId !== profileId) return;
    rafId = window.requestAnimationFrame(flushLiveTurn);
  };
  const cancelPendingFlush = () => {
    if (rafId !== null) {
      window.cancelAnimationFrame(rafId);
      rafId = null;
    }
  };

  try {
    await streamSSE(`/api/profiles/${encodeURIComponent(profileId)}/chat`, requestBody, {
      signal: abortController.signal,
      onEvent: (evType, payload) => {
        if (evType === "status") {
          if (payload.label) {
            assistantTurn.statusLabel = payload.label;
          }
        } else if (evType === "meta") {
          assistantTurn.refused = Boolean(payload.refused);
          assistantTurn.conversational = Boolean(payload.conversational);
          assistantTurn.candidateSources =
            payload.candidate_sources || payload.sources || [];
          assistantTurn.sources = payload.sources || [];
          assistantTurn.debugChunks = payload.debug_chunks || [];
          assistantTurn.bestScore = payload.best_score ?? 0;
          assistantTurn.threshold = payload.threshold ?? 0.5;
          assistantTurn.retrievalQuery = payload.retrieval_query || "";
          assistantTurn.language = payload.language || null;
        } else if (evType === "token") {
          assistantTurn.text += payload.text || "";
        } else if (evType === "done") {
          assistantTurn.streaming = false;
          assistantTurn.text = payload.full_text ?? assistantTurn.text;
          assistantTurn.refused = Boolean(payload.refused);
          assistantTurn.conversational = Boolean(
            payload.conversational ?? assistantTurn.conversational
          );
          assistantTurn.sources = payload.sources || [];
          if (payload.debug_chunks) {
            assistantTurn.debugChunks = payload.debug_chunks;
          }
          if (payload.timings) {
            assistantTurn.timings = payload.timings;
          }
          if (payload.language) {
            assistantTurn.language = payload.language;
          }
        } else if (evType === "error") {
          assistantTurn.streaming = false;
          assistantTurn.error = true;
          assistantTurn.text = payload.message || "Model execution error.";
          assistantTurn.errorDetails = payload.details || null;
        }

        if (assistantTurn.streaming) {
          scheduleLiveTurnFlush();
        }
      },
    });

    cancelPendingFlush();
    assistantTurn.streaming = false;
    finalizeStreamingTurnInDOM(profileId, assistantTurn, assistantTurnIndex);

    if (!assistantTurn.error && state.activeSavedChatByProfile.get(profileId)) {
      await persistSavedChat(profileId).catch(() => {});
    }
  } catch (err) {
    cancelPendingFlush();
    assistantTurn.streaming = false;
    if (err.name === "AbortError") {
      assistantTurn.stopped = true;
      assistantTurn.text = assistantTurn.text
        ? `${assistantTurn.text}\n\n_[Generation stopped]_`
        : "";
      if (!assistantTurn.text) {
        assistantTurn.error = true;
        assistantTurn.text = "Generation stopped before a response was produced.";
      }
      finalizeStreamingTurnInDOM(profileId, assistantTurn, assistantTurnIndex);
    } else if (err.status) {
      assistantTurn.error = true;
      assistantTurn.text = err.message || `Request failed (${err.status})`;
      assistantTurn.errorDetails = err.details || null;
      finalizeStreamingTurnInDOM(profileId, assistantTurn, assistantTurnIndex);
      await refreshAllCb(true);
    } else {
      assistantTurn.error = true;
      assistantTurn.text = err.message || "Connection lost while streaming response.";
      finalizeStreamingTurnInDOM(profileId, assistantTurn, assistantTurnIndex);
    }
  } finally {
    cancelPendingFlush();
    state.chatAbortController = null;
    el.btnStopStream.hidden = true;
    if (state.activeProfile) {
      updateInteractiveControlsLock(state.activeProfile.availability || {});
      updateChatActionButtons(state.activeProfile.id);
    }
  }
}

if (typeof document !== "undefined") {
  const popover = getOrCreateTranslatePopover();

  // Sync aria-expanded when light-dismissed by native Popover API
  if (popover && typeof popover.addEventListener === "function") {
    popover.addEventListener("toggle", (e) => {
      if (e.newState === "closed") {
        if (activeTranslateTrigger) {
          activeTranslateTrigger.setAttribute("aria-expanded", "false");
          activeTranslateTrigger = null;
        }
      }
    });
  }

  // Dismiss on click outside for browsers in fallback mode
  document.addEventListener("click", (e) => {
    if (!isTranslatePopoverOpen(popover)) return;
    if (!popover.contains(e.target) && !activeTranslateTrigger?.contains(e.target)) {
      closeTranslatePopover();
    }
  });

  // Dismiss on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && isTranslatePopoverOpen(popover)) {
      closeTranslatePopover();
    }
  });

  // Dismiss on resize or chat scroll so popover doesn't detach from button
  window.addEventListener("resize", () => {
    if (isTranslatePopoverOpen(popover)) {
      closeTranslatePopover();
    }
  }, { passive: true });

  const chatContainer = el.chatMessages || document.getElementById("chat-messages");
  if (chatContainer) {
    chatContainer.addEventListener("scroll", () => {
      if (isTranslatePopoverOpen(popover)) {
        closeTranslatePopover();
      }
    }, { passive: true });
  }
}

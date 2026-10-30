/* Chat Transcript Rendering, Streaming, Copy/Retry, Citations, Saved Chats & Export (§4.3, §4.5, #10) */

import { apiJson, streamSSE } from "./api.js";
import { SVG_ICONS } from "./constants.js";
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
    metaSpan.textContent = `${chat.turn_count || 0} turn(s)${datePart ? ` · ${datePart}` : ""}`;

    openBtn.appendChild(titleSpan);
    openBtn.appendChild(metaSpan);
    openBtn.addEventListener("click", () => loadSavedChat(pid, chat.id));

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "saved-chat-del-btn";
    delBtn.title = `Delete '${chat.title}'`;
    delBtn.setAttribute("aria-label", `Delete saved chat ${chat.title}`);
    delBtn.textContent = "✕";
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
  if (!firstUser) return "Saved Conversation";
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
        bestScore: t.bestScore ?? 0,
        threshold: t.threshold ?? 0.5,
        retrievalQuery: t.retrievalQuery || "",
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
    el.btnSaveChatLabel.textContent = "Save Chat";
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
    `- **Bot Profile:** ${prof.name}`,
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
        lines.push("**Cited Sources:**");
        for (const src of turn.sources) {
          const loc = src.location_label ? `, ${src.location_label}` : "";
          lines.push(`- \`[${src.index}]\` ${src.source_file}${loc}`);
        }
        lines.push("");
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
  el.citationModalScore.textContent = `Similarity: ${Number(source.score).toFixed(2)}`;
  const loc = source.location_label ? ` · ${source.location_label}` : "";
  el.citationModalMeta.textContent = `${source.source_file}${loc} (Chunk #${source.chunk_index})`;
  el.citationModalText.textContent = source.text || "";
  el.dialogCitation.showModal();
}

function createDebugElement(turn) {
  const wrap = document.createElement("div");
  wrap.className = "debug-panel";
  wrap.hidden = !state.debugEnabled;

  const header = document.createElement("div");
  header.className = "debug-header";
  const threshold = Number(turn.threshold ?? 0.5).toFixed(2);
  const best = Number(turn.bestScore ?? 0).toFixed(2);
  const modeNote = turn.conversational ? " · Conversational Intent" : "";
  header.innerHTML = `<span>Retrieval Candidates (${(turn.debugChunks || []).length}${modeNote})</span><span>Best: ${best} / Cutoff: ${threshold}</span>`;
  wrap.appendChild(header);

  if (turn.retrievalQuery) {
    const qLine = document.createElement("div");
    qLine.className = "helper-text";
    qLine.textContent = `Query: "${turn.retrievalQuery}"`;
    wrap.appendChild(qLine);
  }

  const chunks = turn.debugChunks || [];
  if (chunks.length === 0) {
    const empty = document.createElement("div");
    empty.className = "helper-text";
    empty.textContent = turn.conversational
      ? "Conversational / capability query answered using profile overview (no chunk retrieval required)."
      : "No chunks retrieved from vector store.";
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
      citedBadge.className = "score-pill cited";
      citedBadge.textContent = "CITED";
      badgeGroup.appendChild(citedBadge);
    }

    const scoreBadge = document.createElement("span");
    const passed = Number(c.score) >= Number(turn.threshold ?? 0.5);
    scoreBadge.className = `score-pill ${passed ? "pass" : "fail"}`;
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
  bubble.textContent = turn.text;
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

  const avatar = document.createElement("div");
  avatar.className = "assistant-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.innerHTML = SVG_ICONS.spark;
  row.appendChild(avatar);

  const body = document.createElement("div");
  body.className = "assistant-body";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.hidden = bubbleHidden;

  let refusalTag = null;
  if (createRefusalTag) {
    refusalTag = document.createElement("div");
    refusalTag.className = "refusal-tag";
    refusalTag.textContent = "Outside Knowledge Base";
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

    // First token arrived: hide placeholder status pill and stream in-place inside contentDiv
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

  // Error or Empty Generation State with Prominent Retry Button (#4)
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
    retryBtn.innerHTML = `${SVG_ICONS.retry}<span>Retry Query</span>`;
    retryBtn.addEventListener("click", () => retryAssistantTurn(profileId, turnIndex));
    contentDiv.appendChild(retryBtn);

    return row;
  }

  // Normal or Refused Response Content
  const sourcesMap = buildTurnSourcesMap(turn);
  contentDiv.innerHTML = formatAssistantMarkdownAndCitations(turn.text, sourcesMap);

  // Strictly Used/Cited Sources Only (#5)
  if (!turn.refused && Array.isArray(turn.sources) && turn.sources.length > 0) {
    const sourcesBar = document.createElement("div");
    sourcesBar.className = "sources-bar";

    const lbl = document.createElement("span");
    lbl.className = "sources-label";
    lbl.textContent =
      turn.sources.length === 1 ? "Cited source:" : `Cited sources (${turn.sources.length}):`;
    sourcesBar.appendChild(lbl);

    for (const src of turn.sources) {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "source-pill";
      const loc = src.location_label ? ` · ${src.location_label}` : "";
      pill.innerHTML = `${SVG_ICONS.doc}<span>[${escapeHtml(String(src.index))}] ${escapeHtml(src.source_file + loc)}</span>`;
      pill.addEventListener("click", () => openCitationModal(src));
      sourcesBar.appendChild(pill);
    }
    body.appendChild(sourcesBar);
  }

  // Completed Assistant Turn Action Bar: Copy & Retry (#4)
  if (turn.text) {
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

    body.appendChild(actions);
  }

  if (turn.debugChunks !== null && turn.debugChunks !== undefined) {
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

  // Remove only the retried/failed DOM rows without touching earlier conversation nodes
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
    bestScore: 0,
    threshold: state.activeProfile.similarity_threshold,
    retrievalQuery: "",
    streaming: true,
    error: false,
    errorDetails: null,
    _liveView: null,
  };
  const assistantTurnIndex = turns.length;
  turns.push(assistantTurn);

  // Incrementally append ONLY the new user & assistant nodes (never wipe existing chat DOM)
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

  const requestBody = {
    message,
    history,
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

    // If this chat was already saved by the user, keep the saved copy synced automatically
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

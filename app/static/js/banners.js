/* Profile Availability & Warning Banners (§4.7) */

import { el } from "./dom.js";
import { openPullModal } from "./pullModal.js";
import {
  openReindexConfirmDialog,
  requestEmbeddingModelSwitch,
  saveProfilePatch,
} from "./settings.js";
import { state } from "./state.js";

function makeBanner(variant, message, actions = []) {
  const banner = document.createElement("div");
  banner.className = `inline-banner ${variant}`;

  const msg = document.createElement("span");
  msg.textContent = message;
  banner.appendChild(msg);

  if (actions.length > 0) {
    const actionsWrap = document.createElement("div");
    actionsWrap.className = "banner-actions";
    for (const node of actions) {
      actionsWrap.appendChild(node);
    }
    banner.appendChild(actionsWrap);
  }

  return banner;
}

function makeModelPicker(models, ariaLabel) {
  const picker = document.createElement("select");
  picker.setAttribute("aria-label", ariaLabel);
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = `${m.name} (${m.size_human})`;
    picker.appendChild(opt);
  }
  return picker;
}

function makeActionButton(label, className, onClick) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = className;
  btn.textContent = label;
  btn.addEventListener("click", onClick);
  return btn;
}

export function renderProfileBanners(prof, avail) {
  el.profileBanners.innerHTML = "";
  if (!state.health || !state.health.ollama_reachable) {
    return;
  }

  // 1. Embedding model missing (§4.7 row 4 — no fallback)
  if (avail.embedding_model_installed === false) {
    const actions = [
      makeActionButton(`Download ${prof.embedding_model}`, "btn small primary", () =>
        openPullModal(prof.embedding_model)
      ),
    ];
    const installedEmbeds = state.health.embedding_models || [];
    if (installedEmbeds.length > 0) {
      const picker = makeModelPicker(installedEmbeds, "Switch to installed embedding model");
      actions.push(
        picker,
        makeActionButton("Switch and re-index…", "btn small", () => {
          requestEmbeddingModelSwitch(picker.value);
        })
      );
    }
    el.profileBanners.appendChild(
      makeBanner(
        "danger",
        `Embedding model '${prof.embedding_model}' is not installed. Chat and uploads are blocked.`,
        actions
      )
    );
  }

  // 2. Collection metadata mismatch (§4.7 row 5)
  if (avail.metadata_ok === false) {
    const message =
      avail.block_message ||
      "Embedding model or vector dimension does not match the knowledge base index metadata.";
    const docCount = prof.document_count || 0;
    el.profileBanners.appendChild(
      makeBanner("danger", message, [
        makeActionButton("Re-index knowledge base…", "btn small primary", () => {
          openReindexConfirmDialog({
            mode: "explicit_reindex",
            message: `Re-index all ${docCount} document${docCount === 1 ? "" : "s"} (~${prof.total_chunks || 0} chunks) using '${prof.embedding_model}'?`,
          });
        }),
      ])
    );
  }

  // 3. Chat model fallback active (§4.7 row 3a)
  if (avail.fallback_active && avail.fallback_banner) {
    el.profileBanners.appendChild(
      makeBanner("warning", avail.fallback_banner, [
        makeActionButton(`Download ${prof.chat_model}`, "btn small", () =>
          openPullModal(prof.chat_model)
        ),
      ])
    );
  }

  // 4. Chat model missing & no fallback (§4.7 row 3b)
  if (avail.chat_model_installed === false && !avail.fallback_active) {
    const actions = [];
    if (!state.dismissedChatBlockForProfile.has(prof.id)) {
      const installedChats = state.health.completion_models || [];
      if (installedChats.length > 0) {
        const picker = makeModelPicker(installedChats, "Pick an installed chat model");
        actions.push(
          picker,
          makeActionButton("Use selected model", "btn small", async () => {
            await saveProfilePatch({ chat_model: picker.value });
          })
        );
      }
      actions.push(
        makeActionButton(`Download ${prof.chat_model}`, "btn small primary", () =>
          openPullModal(prof.chat_model)
        ),
        makeActionButton("Dismiss", "btn small subtle", () => {
          state.dismissedChatBlockForProfile.add(prof.id);
          renderProfileBanners(prof, avail);
        })
      );
    }
    el.profileBanners.appendChild(
      makeBanner(
        "danger",
        `Chat model '${prof.chat_model}' is not installed. Chat is blocked.`,
        actions
      )
    );
  }
}

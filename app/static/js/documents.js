/* Organized Knowledge Base Summary, Manager Modal & Upload/Delete (§4.2, #7) */

import { apiJson } from "./api.js";
import { el } from "./dom.js";
import { state } from "./state.js";
import { formatBytes, formatShortDate, getFileExtensionBadge, showFeedback } from "./utils.js";

let selectProfileCb = async () => {};

export function configureDocuments({ selectProfile }) {
  selectProfileCb = selectProfile;
}

export function renderDocumentSummaryAndModal() {
  const docs = state.documents || [];
  const indexedDocs = docs.filter((d) => d.status === "indexed");
  const warnDocs = docs.filter((d) => d.status !== "indexed");
  const totalChunks =
    state.activeProfile?.total_chunks ??
    docs.reduce((acc, d) => acc + (d.chunk_count || 0), 0);

  el.docsSummaryBadge.textContent = `${docs.length} file${docs.length === 1 ? "" : "s"}`;
  el.kbCardHeadline.textContent = `${indexedDocs.length} Indexed Document${indexedDocs.length === 1 ? "" : "s"}`;
  el.kbCardSubline.textContent = `${totalChunks} vector chunk${totalChunks === 1 ? "" : "s"} · PDF, DOCX, MD, TXT`;

  if (warnDocs.length > 0) {
    el.kbWarningPill.textContent = `${warnDocs.length} warning${warnDocs.length === 1 ? "" : "s"}`;
    el.kbWarningPill.hidden = false;
  } else {
    el.kbWarningPill.hidden = true;
  }

  el.btnOpenDocsLabel.textContent =
    docs.length > 0 ? `Manage Files (${docs.length})` : "View Files";

  // Compact preview: show at most the 3 most recent documents so the sidebar never gets crowded
  el.documentPreviewList.innerHTML = "";
  if (docs.length === 0) {
    el.documentPreviewList.hidden = true;
  } else {
    el.documentPreviewList.hidden = false;
    const previewSlice = docs.slice(-3).reverse();
    for (const doc of previewSlice) {
      const li = document.createElement("li");
      li.className = "doc-preview-row";

      const nameSpan = document.createElement("span");
      nameSpan.className = "doc-preview-name";
      nameSpan.textContent = doc.filename;

      const metaSpan = document.createElement("span");
      metaSpan.className = "doc-preview-chunks";
      metaSpan.textContent =
        doc.status === "scanned_unsupported"
          ? "Warning"
          : `${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"}`;

      li.appendChild(nameSpan);
      li.appendChild(metaSpan);
      el.documentPreviewList.appendChild(li);
    }
    if (docs.length > 3) {
      const moreLi = document.createElement("li");
      moreLi.className = "helper-text";
      moreLi.textContent = `+ ${docs.length - 3} more file(s) in document manager`;
      el.documentPreviewList.appendChild(moreLi);
    }
  }

  renderDocsModalList();
}

export function openDocsManagerModal() {
  if (!state.activeProfile) return;
  state.docSearchQuery = "";
  state.docFilter = "all";
  el.docsSearchInput.value = "";
  el.modalUploadStatus.hidden = true;
  document.querySelectorAll(".filter-pill").forEach((pill) => {
    pill.classList.toggle("active", pill.dataset.docFilter === "all");
  });
  renderDocsModalList();
  el.dialogManageDocs.showModal();
}

export function renderDocsModalList() {
  const docs = state.documents || [];
  const profName = state.activeProfile ? state.activeProfile.name : "Profile";
  const totalChunks =
    state.activeProfile?.total_chunks ??
    docs.reduce((acc, d) => acc + (d.chunk_count || 0), 0);

  el.docsModalTitle.textContent = `${profName} — Knowledge Base`;
  el.docsModalSubtitle.textContent = `${docs.length} document${docs.length === 1 ? "" : "s"} · ${totalChunks} indexed chunk${totalChunks === 1 ? "" : "s"}`;

  const query = state.docSearchQuery.trim().toLowerCase();
  const filtered = docs.filter((doc) => {
    if (query && !doc.filename.toLowerCase().includes(query)) {
      return false;
    }
    if (state.docFilter === "indexed" && doc.status !== "indexed") {
      return false;
    }
    if (state.docFilter === "warnings" && doc.status === "indexed") {
      return false;
    }
    return true;
  });

  el.modalDocumentList.innerHTML = "";
  if (filtered.length === 0) {
    const empty = document.createElement("li");
    empty.className = "docs-empty-state";
    empty.textContent =
      docs.length === 0
        ? "No documents uploaded yet. Click 'Upload Files…' above to add PDF, DOCX, Markdown, or TXT files."
        : "No documents match your current search or filter.";
    el.modalDocumentList.appendChild(empty);
    return;
  }

  for (const doc of filtered) {
    const li = document.createElement("li");
    li.className = "modal-doc-row";

    const main = document.createElement("div");
    main.className = "modal-doc-main";

    const badge = document.createElement("span");
    badge.className = "ext-badge";
    badge.textContent = getFileExtensionBadge(doc.filename);
    main.appendChild(badge);

    const info = document.createElement("div");
    info.className = "modal-doc-info";

    const name = document.createElement("span");
    name.className = "modal-doc-name";
    name.textContent = doc.filename;
    info.appendChild(name);

    const meta = document.createElement("span");
    const sizeStr = formatBytes(doc.size_bytes);
    const dateStr = formatShortDate(doc.uploaded_at);
    if (doc.status === "scanned_unsupported") {
      meta.className = "modal-doc-meta scanned";
      meta.textContent = `Warning: ${doc.warning || "likely scanned PDF, unsupported"} · ${sizeStr}`;
    } else {
      meta.className = "modal-doc-meta";
      meta.textContent = `Indexed · ${doc.chunk_count} chunk(s) · ${sizeStr}${dateStr ? ` · Uploaded ${dateStr}` : ""}`;
    }
    info.appendChild(meta);
    main.appendChild(info);

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "btn danger-outline small";
    delBtn.setAttribute("aria-label", `Remove ${doc.filename}`);
    delBtn.textContent = "Remove";
    delBtn.addEventListener("click", () => handleDeleteDocument(doc.id, doc.filename));

    li.appendChild(main);
    li.appendChild(delBtn);
    el.modalDocumentList.appendChild(li);
  }
}

export async function handleFileUpload(fileList, opts = {}) {
  if (!state.activeProfile || !fileList || fileList.length === 0) return;
  const profileId = state.activeProfile.id;
  const files = Array.from(fileList);
  const statusTarget = opts.fromModal ? el.modalUploadStatus : el.uploadStatusMsg;

  let indexedCount = 0;
  const warnings = [];

  for (let i = 0; i < files.length; i += 1) {
    const file = files[i];
    showFeedback(
      statusTarget,
      `Ingesting '${file.name}' (${i + 1}/${files.length})…`,
      "success"
    );

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await apiJson(
        `/api/profiles/${encodeURIComponent(profileId)}/documents`,
        {
          method: "POST",
          body: formData,
        }
      );
      if (res.warning) {
        warnings.push(`${file.name}: ${res.warning}`);
      } else {
        indexedCount += 1;
      }
    } catch (err) {
      warnings.push(`${file.name}: ${err.message}`);
    }
  }

  el.fileUploadInput.value = "";
  el.modalFileUploadInput.value = "";
  await selectProfileCb(profileId);

  if (warnings.length > 0) {
    showFeedback(
      statusTarget,
      warnings.join(" | "),
      warnings.some((w) => w.includes("failed") || w.includes("blocked"))
        ? "error"
        : "warning"
    );
  } else {
    showFeedback(
      statusTarget,
      `✓ Indexed ${indexedCount} document${indexedCount === 1 ? "" : "s"}.`,
      "success"
    );
    setTimeout(() => {
      statusTarget.hidden = true;
    }, 3600);
  }
}

export async function handleDeleteDocument(docId, filename) {
  if (!state.activeProfile) return;
  try {
    await apiJson(
      `/api/profiles/${encodeURIComponent(state.activeProfile.id)}/documents/${encodeURIComponent(docId)}`,
      { method: "DELETE" }
    );
    await selectProfileCb(state.activeProfile.id);
    const target = el.dialogManageDocs.open ? el.modalUploadStatus : el.uploadStatusMsg;
    showFeedback(target, `Removed '${filename}' and its chunks.`, "success");
    setTimeout(() => {
      target.hidden = true;
    }, 3200);
  } catch (err) {
    const target = el.dialogManageDocs.open ? el.modalUploadStatus : el.uploadStatusMsg;
    showFeedback(target, err.message, "error");
  }
}

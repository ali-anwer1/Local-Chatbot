/* Model Pull Modal (§4.7) */

import { streamSSE } from "./api.js";
import { el } from "./dom.js";
import { state } from "./state.js";
import { estimateModelSize } from "./utils.js";

let refreshAllCb = async () => {};

export function configurePullModal({ refreshAll }) {
  refreshAllCb = refreshAll;
}

export function openPullModal(prefillModel = "") {
  el.pullModelName.value = prefillModel;
  el.pullModelName.disabled = false;
  el.pullConfirmBox.hidden = true;
  el.pullStepProgress.hidden = true;
  el.pullErrorBox.hidden = true;
  el.pullErrorDetailsWrap.hidden = true;
  el.btnStepConfirmPull.hidden = false;
  el.btnStartPull.hidden = true;
  el.btnCancelPull.textContent = "Close";
  el.dialogPullModel.showModal();
}

export function reviewPullConfirmation() {
  const tag = el.pullModelName.value.trim();
  if (!tag) {
    el.pullModelName.focus();
    return;
  }
  const approxSize = estimateModelSize(tag);
  el.pullConfirmText.textContent = `Confirm download of '${tag}' (${approxSize})? This requires an active internet connection.`;
  el.pullConfirmBox.hidden = false;
  el.pullErrorBox.hidden = true;
  el.btnStepConfirmPull.hidden = true;
  el.btnStartPull.hidden = false;
}

export async function executeModelPull() {
  const tag = el.pullModelName.value.trim();
  if (!tag || state.pullAbortController) return;

  const abortController = new AbortController();
  state.pullAbortController = abortController;

  el.pullModelName.disabled = true;
  el.btnStartPull.hidden = true;
  el.pullConfirmBox.hidden = true;
  el.pullErrorBox.hidden = true;
  el.pullStepProgress.hidden = false;
  el.pullProgressBar.value = 0;
  el.pullStatusLabel.textContent = `Connecting to download ${tag}…`;
  el.pullPercentLabel.textContent = "0%";
  el.btnCancelPull.textContent = "Cancel Download";

  let succeeded = false;

  try {
    await streamSSE(
      "/api/models/pull",
      { model: tag, confirmed: true },
      {
        signal: abortController.signal,
        onEvent: (evType, payload) => {
          if (evType === "progress") {
            el.pullStatusLabel.textContent = payload.status || "Downloading…";
            if (payload.percent !== null && payload.percent !== undefined) {
              el.pullProgressBar.value = payload.percent;
              el.pullPercentLabel.textContent = `${payload.percent}%`;
            }
          } else if (evType === "done") {
            succeeded = true;
            el.pullProgressBar.value = 100;
            el.pullPercentLabel.textContent = "100%";
            el.pullStatusLabel.textContent = `Installed '${tag}'!`;
          } else if (evType === "error") {
            showPullError(payload.message || "Download failed.", payload.details || null);
          }
        },
      }
    );

    if (succeeded) {
      await refreshAllCb(true);
      setTimeout(() => {
        if (el.dialogPullModel.open) {
          el.dialogPullModel.close();
        }
      }, 700);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      showPullError("Download cancelled by user.", null);
    } else if (err.status) {
      showPullError(err.payload?.message || "Download failed.", err.details || null);
    } else {
      showPullError("No internet connection; download unavailable offline", err.message);
    }
  } finally {
    state.pullAbortController = null;
    el.pullModelName.disabled = false;
    el.btnCancelPull.textContent = "Close";
  }
}

export function showPullError(message, details) {
  el.pullStepProgress.hidden = true;
  el.pullErrorMsg.textContent = message;
  if (details) {
    el.pullErrorDetails.textContent = details;
    el.pullErrorDetailsWrap.hidden = false;
  } else {
    el.pullErrorDetailsWrap.hidden = true;
  }
  el.pullErrorBox.hidden = false;
  el.btnStepConfirmPull.hidden = false;
}

export function closeOrCancelPull() {
  if (state.pullAbortController) {
    state.pullAbortController.abort();
  } else {
    el.dialogPullModel.close();
  }
}

import { SETTING_INFO_TIPS, SVG_ICONS } from "./constants.js";

/**
 * infoTips.js - Accessible setting info tooltips (ⓘ).
 *
 * Implements WCAG 2.1 / 1.4.13 (Content on Hover or Focus):
 * - Hover or keyboard focus shows the popover (unpinned).
 * - Click or tap toggles it (pins it open on touch/click).
 * - Escape key dismisses the popover immediately and keeps focus on the trigger.
 * - Outside click dismisses the popover.
 * - Only one popover open at a time.
 * - Positions itself within the settings drawer / dialog / viewport and flips above if needed.
 */

let popoverEl = null;
let activeTrigger = null;
let isPinned = false;
let hideTimer = null;

function clearHideTimer() {
  if (hideTimer) {
    clearTimeout(hideTimer);
    hideTimer = null;
  }
}

function scheduleHide() {
  clearHideTimer();
  if (isPinned) return;
  hideTimer = setTimeout(() => {
    hidePopover(true);
  }, 160);
}

function getOrCreatePopover() {
  let el = document.getElementById("info-tooltip-popover");
  if (!el) {
    el = document.createElement("div");
    el.id = "info-tooltip-popover";
    el.className = "info-popover";
    el.setAttribute("role", "tooltip");
    el.setAttribute("hidden", "");
    try {
      el.setAttribute("popover", "manual");
    } catch {
      // Ignore if unsupported
    }
    document.body.appendChild(el);
  }
  return el;
}

function positionPopover(trigger, popover) {
  const triggerRect = trigger.getBoundingClientRect();

  // Find enclosing container boundary (settings panel, modal dialog, or viewport)
  const container = trigger.closest("#settings-panel, dialog[open], .drawer-scroll") || document.body;
  const containerRect =
    container && container !== document.body
      ? container.getBoundingClientRect()
      : { left: 8, right: window.innerWidth - 8, top: 8, bottom: window.innerHeight - 8 };

  // Temporarily make visible for measurement
  popover.style.visibility = "hidden";
  popover.removeAttribute("hidden");
  popover.style.display = "block";

  const popoverRect = popover.getBoundingClientRect();
  const popoverWidth = popoverRect.width || 260;
  const popoverHeight = popoverRect.height || 80;

  const gap = 6;
  let top = triggerRect.bottom + gap;

  // Flip above if overflowing bottom of viewport or container
  const maxBottom = Math.min(window.innerHeight - 12, containerRect.bottom || window.innerHeight - 12);
  if (top + popoverHeight > maxBottom) {
    const flippedTop = triggerRect.top - popoverHeight - gap;
    if (flippedTop >= 12) {
      top = flippedTop;
    }
  }

  // Horizontal clamping within container and viewport
  let left = triggerRect.left;
  const minLeft = Math.max(12, (containerRect.left || 0) + 8);
  const maxLeft = Math.min(window.innerWidth - popoverWidth - 12, (containerRect.right || window.innerWidth) - popoverWidth - 8);

  if (left < minLeft) {
    left = minLeft;
  } else if (left > maxLeft) {
    left = maxLeft;
  }

  popover.style.top = `${Math.round(top)}px`;
  popover.style.left = `${Math.round(left)}px`;
  popover.style.visibility = "visible";
}

export function showPopover(trigger, pin = false) {
  clearHideTimer();
  const key = trigger.dataset.tipKey;
  const info = SETTING_INFO_TIPS[key];
  if (!info) return;

  if (!popoverEl) {
    popoverEl = getOrCreatePopover();
  }

  if (activeTrigger && activeTrigger !== trigger) {
    activeTrigger.setAttribute("aria-expanded", "false");
  }

  activeTrigger = trigger;
  isPinned = pin;
  activeTrigger.setAttribute("aria-expanded", "true");

  popoverEl.innerHTML = `
    <div class="info-popover-title">${info.title}</div>
    <div class="info-popover-text">${info.text}</div>
  `;

  if (typeof popoverEl.showPopover === "function") {
    try {
      popoverEl.showPopover();
    } catch {
      // Ignored if already open
    }
  }

  positionPopover(trigger, popoverEl);
}

export function hidePopover(immediate = false) {
  clearHideTimer();
  if (!popoverEl || popoverEl.hasAttribute("hidden")) {
    if (activeTrigger) {
      activeTrigger.setAttribute("aria-expanded", "false");
      activeTrigger = null;
    }
    isPinned = false;
    return;
  }

  if (immediate) {
    if (typeof popoverEl.hidePopover === "function") {
      try {
        popoverEl.hidePopover();
      } catch {
        // Ignored if already closed
      }
    }
    popoverEl.setAttribute("hidden", "");
    popoverEl.style.display = "none";
    if (activeTrigger) {
      activeTrigger.setAttribute("aria-expanded", "false");
      activeTrigger = null;
    }
    isPinned = false;
  } else {
    scheduleHide();
  }
}

function bindTriggerEvents(btn) {
  btn.addEventListener("pointerenter", () => {
    if (!isPinned) {
      showPopover(btn, false);
    }
  });

  btn.addEventListener("pointerleave", () => {
    if (!isPinned) {
      scheduleHide();
    }
  });

  btn.addEventListener("focusin", () => {
    if (!isPinned) {
      showPopover(btn, false);
    }
  });

  btn.addEventListener("focusout", (e) => {
    if (!isPinned) {
      // If focus moved inside popover, keep open
      if (popoverEl && e.relatedTarget && popoverEl.contains(e.relatedTarget)) {
        return;
      }
      hidePopover(true);
    }
  });

  btn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (activeTrigger === btn && isPinned) {
      hidePopover(true);
    } else {
      showPopover(btn, true);
    }
  });
}

export function initInfoTips() {
  popoverEl = getOrCreatePopover();

  popoverEl.addEventListener("pointerenter", () => {
    clearHideTimer();
  });

  popoverEl.addEventListener("pointerleave", () => {
    if (!isPinned) {
      scheduleHide();
    }
  });

  // Ensure every setting defined in SETTING_INFO_TIPS has an info-tip button
  for (const [key, info] of Object.entries(SETTING_INFO_TIPS)) {
    let btn = document.querySelector(`.info-tip-btn[data-tip-key="${key}"]`);
    if (!btn) {
      // Find label matching key
      const labels = document.querySelectorAll(`label[for="${key}"]`);
      let targetLabel = null;
      for (const l of labels) {
        if (!l.classList.contains("checkbox-row")) {
          targetLabel = l;
          break;
        }
      }
      if (!targetLabel && labels.length > 0) {
        targetLabel = labels[0];
      }

      if (targetLabel) {
        btn = document.createElement("button");
        btn.type = "button";
        btn.className = "info-tip-btn";
        btn.setAttribute("aria-label", `About ${info.title}`);
        btn.setAttribute("aria-describedby", "info-tooltip-popover");
        btn.setAttribute("aria-expanded", "false");
        btn.dataset.tipKey = key;
        btn.innerHTML = SVG_ICONS.info;

        // If label is inside .field-label-row, append inside; else wrap or place after
        const parentRow = targetLabel.closest(".field-label-row");
        if (parentRow) {
          parentRow.appendChild(btn);
        } else {
          const wrapper = document.createElement("div");
          wrapper.className = "field-label-row";
          targetLabel.parentNode.insertBefore(wrapper, targetLabel);
          wrapper.appendChild(targetLabel);
          wrapper.appendChild(btn);
        }
      }
    }

    if (btn && !btn._hasInfoTipBound) {
      btn._hasInfoTipBound = true;
      bindTriggerEvents(btn);
    }
  }

  // Also bind any statically present buttons that match
  document.querySelectorAll(".info-tip-btn").forEach((btn) => {
    if (!btn._hasInfoTipBound) {
      btn._hasInfoTipBound = true;
      bindTriggerEvents(btn);
    }
  });

  // Dismiss on Escape
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && activeTrigger) {
      const trigger = activeTrigger;
      e.preventDefault();
      e.stopPropagation();
      hidePopover(true);
      trigger.focus();
    }
  });

  // Dismiss on outside click
  document.addEventListener("pointerdown", (e) => {
    if (activeTrigger) {
      if (popoverEl && popoverEl.contains(e.target)) return;
      if (activeTrigger.contains(e.target)) return;
      hidePopover(true);
    }
  });

  // Reposition on scroll / resize
  window.addEventListener(
    "scroll",
    () => {
      if (activeTrigger && popoverEl && !popoverEl.hasAttribute("hidden")) {
        const rect = activeTrigger.getBoundingClientRect();
        if (rect.bottom < 0 || rect.top > window.innerHeight) {
          hidePopover(true);
        } else {
          positionPopover(activeTrigger, popoverEl);
        }
      }
    },
    { capture: true, passive: true }
  );

  window.addEventListener(
    "resize",
    () => {
      if (activeTrigger && popoverEl && !popoverEl.hasAttribute("hidden")) {
        positionPopover(activeTrigger, popoverEl);
      }
    },
    { passive: true }
  );
}

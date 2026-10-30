/* Shared constants & inline SVG icons */

export const KNOWN_MODEL_SIZE_ESTIMATES = {
  "qwen3.5:4b": "~2.6 GB",
  "qwen3:4b": "~2.6 GB",
  "qwen2.5:3b": "~1.9 GB",
  "nomic-embed-text": "~274 MB",
  "nomic-embed-text:latest": "~274 MB",
  "mxbai-embed-large": "~670 MB",
  "bge-m3": "~1.2 GB",
  "all-minilm": "~46 MB",
};

export const SVG_ICONS = {
  copy: `<svg class="icon-sm" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.65" aria-hidden="true"><rect x="7" y="7" width="9.5" height="9.5" rx="1.75"/><path d="M4.5 13V4.75A1.25 1.25 0 015.75 3.5H14" stroke-linecap="round"/></svg>`,
  check: `<svg class="icon-sm" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4.5 10.5l3.5 3.5 7.5-7.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  retry: `<svg class="icon-sm" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M3.75 10a6.25 6.25 0 0110.67-4.42L16.25 7.4M16.25 3.75V7.4h-3.65M16.25 10a6.25 6.25 0 01-10.67 4.42L3.75 12.6M3.75 16.25V12.6h3.65" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  spark: `<svg class="icon-sm" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path d="M10 2a.75.75 0 01.67.415l1.882 3.765 4.155.604a.75.75 0 01.416 1.279l-3.007 2.93.71 4.139a.75.75 0 01-1.088.791L10 13.97l-3.738 1.953a.75.75 0 01-1.088-.79l.71-4.14-3.007-2.93a.75.75 0 01.416-1.28l4.155-.603L9.33 2.415A.75.75 0 0110 2z"/></svg>`,
  doc: `<svg class="icon-sm" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M5.5 3.5h6l3.5 3.5v9.5a1 1 0 01-1 1h-8.5a1 1 0 01-1-1v-12a1 1 0 011-1z" stroke-linejoin="round"/><path d="M11.5 3.5V7H15" stroke-linejoin="round"/></svg>`,
};

/* Assistant Markdown and inline citation chip formatter */

import { escapeHtml } from "./utils.js";

export function expandCitationExpression(expr) {
  const indices = [];
  for (const rawPart of expr.split(",")) {
    const part = rawPart.trim();
    if (!part) continue;
    if (part.includes("-") || part.includes("–")) {
      const bounds = part.split(/[-–]/);
      if (bounds.length === 2) {
        const s = parseInt(bounds[0].trim(), 10);
        const e = parseInt(bounds[1].trim(), 10);
        if (!Number.isNaN(s) && !Number.isNaN(e)) {
          const low = Math.min(s, e);
          const high = Math.max(s, e);
          for (let i = low; i <= high; i += 1) {
            indices.push(i);
          }
        }
      }
    } else {
      const n = parseInt(part, 10);
      if (!Number.isNaN(n)) {
        indices.push(n);
      }
    }
  }
  return indices;
}

export function formatAssistantMarkdownAndCitations(text, sourcesMap) {
  if (!text) return "";
  let safe = escapeHtml(text);

  // Code blocks ```...```
  const codeBlocks = [];
  safe = safe.replace(/```(?:[a-zA-Z0-9_-]+\n)?([\s\S]*?)```/g, (_m, code) => {
    const placeholder = `@@CODEBLOCK_${codeBlocks.length}@@`;
    codeBlocks.push(`<pre tabindex="0"><code>${code.trim()}</code></pre>`);
    return placeholder;
  });

  // Inline code `...`
  safe = safe.replace(/`([^`\n]+)`/g, "<code>$1</code>");

  // Bold **...**
  safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // Clickable inline citations [1], [1, 2], [1-3], [Source 1], [Passage 2]
  safe = safe.replace(
    /\[(?:(?:passage|source|doc|ref)s?\s*)?(\d+(?:\s*(?:,|-|–)\s*\d+)*)\]/gi,
    (fullMatch, expr) => {
      const nums = expandCitationExpression(expr);
      if (nums.length === 0) return fullMatch;
      const chips = nums.map((num) => {
        if (sourcesMap && sourcesMap.has(num)) {
          return `<button type="button" class="citation-chip" data-cite-idx="${num}" aria-label="View source citation ${num}">[${num}]</button>`;
        }
        return `[${num}]`;
      });
      return chips.join(" ");
    }
  );

  // Split paragraphs & lists
  const blocks = safe.split(/\n{2,}/);
  const htmlBlocks = blocks.map((block) => {
    const trimmed = block.trim();
    if (!trimmed) return "";
    if (/^@@CODEBLOCK_\d+@@$/.test(trimmed)) return trimmed;

    const lines = trimmed.split("\n");
    if (lines.length === 1 && /^#{1,3}\s+/.test(lines[0])) {
      return `<h4>${lines[0].replace(/^#{1,3}\s+/, "")}</h4>`;
    }
    if (lines.every((l) => /^[-*]\s+/.test(l.trim()))) {
      const items = lines
        .map((l) => `<li>${l.trim().replace(/^[-*]\s+/, "")}</li>`)
        .join("");
      return `<ul>${items}</ul>`;
    }
    if (lines.every((l) => /^\d+\.\s+/.test(l.trim()))) {
      const items = lines
        .map((l) => `<li>${l.trim().replace(/^\d+\.\s+/, "")}</li>`)
        .join("");
      return `<ol>${items}</ol>`;
    }
    return `<p>${trimmed.replace(/\n/g, "<br>")}</p>`;
  });

  let joined = htmlBlocks.join("");
  codeBlocks.forEach((html, idx) => {
    joined = joined.replace(`@@CODEBLOCK_${idx}@@`, html);
  });
  return joined;
}

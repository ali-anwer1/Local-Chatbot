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
  copy: `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>`,
  check: `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>`,
  retry: `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>`,
  translate: `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="m22 22-5-10-5 10"/><path d="M14 18h6"/></svg>`,
  doc: `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>`,
  chevronRight: `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>`,
  info: `<svg class="icon info-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>`,
};

export const SETTING_INFO_TIPS = {
  "setting-name": {
    title: "Display name",
    text: "Name shown in the sidebar and chat header. Cosmetic only.",
  },
  "setting-default-language": {
    title: "Default language",
    text: "The language used when no language can be detected from the question (e.g. very short queries) and no specific language was chosen.",
  },
  "setting-preamble": {
    title: "System preamble",
    text: "Instructions sent to the model before every conversation: the bot's role, tone and scope. It does not add knowledge — facts must come from the uploaded documents.",
  },
  "setting-chat-model": {
    title: "Chat model",
    text: "The model that writes the answers. Larger models usually answer better but are slower and use more memory.",
  },
  "setting-thinking": {
    title: "Thinking",
    text: "Lets models that support it reason step by step before answering. Can help on complex questions but makes responses much slower. No effect on models without thinking support.",
  },
  "setting-embedding-model": {
    title: "Embedding model",
    text: "Turns documents and questions into vectors so relevant passages can be found. Documents and questions must use the same embedding model, so changing it re-indexes every document.",
  },
  "setting-embedding-device": {
    title: "Embedding device",
    text: "Where the embedding model runs. CPU leaves the iGPU's memory for the chat model; Auto lets Ollama decide. Applies to all profiles.",
  },
  "setting-threshold": {
    title: "Refusal similarity threshold",
    text: "How closely the best-matching passage must match the question (0–1) before the model is asked to answer. Below this, the bot replies with the refusal message without calling the model. Higher = stricter: fewer unsupported answers, more refusals. Lower = answers more often, with more risk of off-topic answers.",
  },
  "setting-temperature": {
    title: "Temperature",
    text: "How much the wording varies. Low (0–0.3) keeps answers consistent and close to the sources; higher values are more varied and more likely to drift.",
  },
  "setting-topk": {
    title: "Top-K chunks",
    text: "How many of the best-matching passages are given to the model with each question. More covers more ground but makes the prompt longer and responses slower.",
  },
  "setting-chunk-size": {
    title: "Chunk size",
    text: "Length, in characters, of the pieces documents are split into. Smaller pieces match questions more precisely but carry less context each; larger pieces carry more context but match less precisely. Applies to new uploads; existing documents are not re-indexed automatically.",
  },
  "setting-chunk-overlap": {
    title: "Overlap",
    text: "Characters repeated between neighbouring chunks so sentences at a boundary aren't cut in half. Usually 10–20% of the chunk size. Applies to new uploads; existing documents are not re-indexed automatically.",
  },
  "setting-history-turns": {
    title: "Conversation history turns",
    text: "How many previous question-and-answer pairs are sent with each new question, so follow-ups like \"what about the second one?\" work. More turns = better follow-ups but longer prompts and slower responses.",
  },
  "setting-refusal-msg": {
    title: "Refusal message",
    text: "The exact text shown when the answer can't be found in the knowledge base.",
  },
  "setting-refusal-other-langs": {
    title: "Refusal message in other languages",
    text: "The message shown when an answer cannot be found, customized for each supported language. If left empty, default translations are used.",
  },
  "new-profile-name": null,
  "new-profile-preamble": null,
  "new-profile-chat-model": null,
  "new-profile-embed-model": null,
};

SETTING_INFO_TIPS["new-profile-name"] = SETTING_INFO_TIPS["setting-name"];
SETTING_INFO_TIPS["new-profile-preamble"] = SETTING_INFO_TIPS["setting-preamble"];
SETTING_INFO_TIPS["new-profile-chat-model"] = SETTING_INFO_TIPS["setting-chat-model"];
SETTING_INFO_TIPS["new-profile-embed-model"] = SETTING_INFO_TIPS["setting-embedding-model"];

/* Shared browser application state */

export const state = {
  health: null,
  profiles: [],
  activeProfileId: null,
  activeProfile: null,
  documents: [],
  debugEnabled: false,
  leftSidebarCollapsed: false,
  settingsPanelOpen: false, // Closed by default (ChatGPT / Gemini style)
  docSearchQuery: "",
  docFilter: "all", // "all" | "indexed" | "warnings"
  settingsFeedbackTimer: null,
  chatAbortController: null,
  pullAbortController: null,
  pendingReindexPayload: null,
  // Per-profile browser session state
  transcripts: new Map(), // profileId -> Array<TurnItem>
  thinkingByProfile: new Map(), // profileId -> boolean
  savedChatsByProfile: new Map(), // profileId -> Array<SavedChatSummary>
  activeSavedChatByProfile: new Map(), // profileId -> { id, title } | null
  dismissedChatBlockForProfile: new Set(),
};

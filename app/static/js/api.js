/* JSON and SSE HTTP client helpers */

function buildHttpError(resp, data) {
  const err = new Error(data.message || `Request failed (${resp.status})`);
  err.code = data.code || "http_error";
  err.details = data.details || null;
  err.payload = data;
  err.status = resp.status;
  return err;
}

export async function apiJson(url, options = {}) {
  const resp = await fetch(url, {
    headers: {
      ...(options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...(options.headers || {}),
    },
    ...options,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw buildHttpError(resp, data);
  }
  return data;
}

function parseSSEChunkBuffer(buffer, onEvent) {
  const frames = buffer.split("\n\n");
  const remainder = frames.pop() ?? "";
  for (const frame of frames) {
    if (!frame.trim()) continue;
    let eventType = "message";
    const dataLines = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }
    if (dataLines.length > 0) {
      try {
        const parsed = JSON.parse(dataLines.join("\n"));
        onEvent(eventType, parsed);
      } catch {
        // ignore malformed partial frame
      }
    }
  }
  return remainder;
}

export async function streamSSE(url, body, { signal, onEvent } = {}) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw buildHttpError(resp, data);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = parseSSEChunkBuffer(buffer, onEvent);
  }
  if (buffer.trim()) {
    parseSSEChunkBuffer(`${buffer}\n\n`, onEvent);
  }
}

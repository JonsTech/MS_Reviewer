import express from "express";
import fs from "fs";
import path from "path";

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use(express.static("public"));

/* ================= HARD SETTINGS ================= */
// IMPORTANT: force IPv4 (fixes localhost/IPv6 issues)
const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://127.0.0.1:11434";
const MODEL = process.env.OLLAMA_MODEL ?? "gpt-oss:20b";

// Keep model loaded (Ollama supports keep_alive on /api/chat)
const KEEP_ALIVE_RAW = process.env.OLLAMA_KEEP_ALIVE;
const KEEP_ALIVE = KEEP_ALIVE_RAW === undefined
  ? -1
  : (Number.isFinite(+KEEP_ALIVE_RAW) ? +KEEP_ALIVE_RAW : KEEP_ALIVE_RAW);

// Speed caps
const SMOKE_TIMEOUT_MS = Number(process.env.OLLAMA_SMOKE_TIMEOUT_MS ?? 120000); // first load can be slow
const REVIEW_TIMEOUT_MS = Number(process.env.OLLAMA_REVIEW_TIMEOUT_MS ?? 600000);

const SMOKE_NUM_PREDICT = Number(process.env.OLLAMA_SMOKE_NUM_PREDICT ?? 16);
const SMOKE_NUM_CTX = Number(process.env.OLLAMA_SMOKE_NUM_CTX ?? 256);

const REVIEW_NUM_PREDICT = Number(process.env.OLLAMA_REVIEW_NUM_PREDICT ?? 800);
const REVIEW_NUM_CTX = Number(process.env.OLLAMA_REVIEW_NUM_CTX ?? 2048);
const TEMPERATURE = Number(process.env.OLLAMA_TEMPERATURE ?? 0);
const THINK_LEVEL = process.env.OLLAMA_THINK ?? "low"; // gpt-oss expects: low|medium|high
const STREAM_FORMAT_DEFAULT = process.env.OLLAMA_STREAM_FORMAT ?? "schema"; // json|schema|none
/* ================================================= */

const now = () => new Date().toISOString();
const CABINET_PATH = path.join(process.cwd(), "cabinet.json");

const logBuffer = [];
const LOG_LIMIT = 500;

function logEvent(level, msg, data) {
  const entry = { ts: now(), level, msg, data };
  logBuffer.push(entry);
  if (logBuffer.length > LOG_LIMIT) logBuffer.shift();
  const suffix = data ? ` ${JSON.stringify(data)}` : "";
  console.log(`[${entry.ts}] ${level.toUpperCase()} ${msg}${suffix}`);
  return entry;
}

function withTimeout(promise, ms, label = "") {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Timeout ${label} after ${ms}ms`)), ms)
    ),
  ]);
}

function countWords(s) {
  return String(s ?? "")
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
}

function validateReview(r) {
  const issues = [];
  if (!r) {
    issues.push("Missing review object");
    return issues;
  }
  if (countWords(r.title) > 8) issues.push(`Title too long (${countWords(r.title)} words)`);
  if (countWords(r.summary) > 12) issues.push(`Summary too long (${countWords(r.summary)} words)`);
  const detailsWords = countWords(r.details);
  if (detailsWords < 50 || detailsWords > 150) {
    issues.push(`Details word count out of range (${detailsWords})`);
  }
  if (!Array.isArray(r.actions) || r.actions.length !== 3) {
    issues.push("Actions must be an array with exactly 3 items");
  } else {
    r.actions.forEach((a, i) => {
      const w = countWords(a);
      if (w > 12) issues.push(`Action ${i + 1} too long (${w} words)`);
    });
  }
  return issues;
}

function readCabinet() {
  const raw = fs.readFileSync(CABINET_PATH, "utf8");
  return JSON.parse(raw);
}

// Ollama /api/chat supports "format": "json" or JSON schema  :contentReference[oaicite:7]{index=7}
async function ollamaChat({ messages, format, options, think }) {
  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages,
      stream: false,
      keep_alive: KEEP_ALIVE,
      format,   // can be undefined, "json", or schema
      options,  // num_predict/num_ctx/temperature/etc
      think,    // thinking control (gpt-oss expects low|medium|high)
    }),
  });

  if (!res.ok) throw new Error(`Ollama error ${res.status}: ${await res.text()}`);
  return await res.json();
}

/* ========= Preload model at server start =========
Ollama FAQ: send an empty request to /api/chat to preload  :contentReference[oaicite:8]{index=8} */
async function preloadModel() {
  try {
    const data = await withTimeout(
      ollamaChat({
        messages: [{ role: "user", content: "" }],
        options: { num_predict: 1, num_ctx: 256, temperature: 0 },
        think: THINK_LEVEL,
      }),
      120000,
      "preload"
    );

    logEvent("info", "preload ok", { model: MODEL, load_duration: data.load_duration, total_duration: data.total_duration });
  } catch (e) {
    logEvent("error", "preload failed", { error: e.message });
  }
}

const REVIEW_SCHEMA = {
  type: "array",
  items: {
    type: "object",
    properties: {
      id: { type: "string", description: "Must match the persona ID exactly" },
      title: { type: "string" },
      summary: { type: "string" },
      details: { type: "string" },
      actions: { type: "array", items: { type: "string" }, minItems: 3, maxItems: 3 },
    },
    required: ["id", "title", "summary", "details", "actions"],
    additionalProperties: false,
  },
};

function buildPrompt(personas, text, opts = {}) {
  const { includeSchema = true } = opts;
  const ids = personas.map(p => p.id).join(", ");
  const lines = [
    "You are a strict management-summary reviewer.",
    `You must act as the following personas: ${JSON.stringify(personas)}`,
    "",
    "Instructions:",
    "1. Analyze the text below from the perspective of EACH persona.",
    "2. Return a JSON ARRAY containing exactly one review object per persona.",
    `3. Each review object MUST have the 'id' field set to the corresponding persona ID (${ids}).`,
    "4. Output Format: JSON Array (e.g. `[{...}, {...}]`). Do NOT output a single object.",
    "5. The response MUST be a JSON array starting with `[`.",
    "6. Be blunt and critical.",
    "",
    "Rules:",
    "- No markdown.",
    "- No conversational text.",
    "- Title <= 8 words.",
    "- Summary <= 12 words.",
    "- Details 50-150 words.",
    "- Actions: exactly 3 items, each <= 12 words.",
    "- Be blunt when the persona strategy implies it.",
    "- Do NOT rewrite or restate the summary.",
    "- Do NOT describe findings as facts; critique the writing.",
    "- Actions must be improvements to the report, not remediation steps.",
    "",
    "Management summary:",
    text,
  ];

  if (includeSchema) {
    lines.splice(4, 0, "JSON Schema:", JSON.stringify(REVIEW_SCHEMA), "");
  }

  return lines.join("\n");
}

function safeJsonParse(raw) {
  try {
    return { ok: true, value: JSON.parse(raw) };
  } catch (e) {
    const start = raw.indexOf("[");
    const end = raw.lastIndexOf("]");
    if (start !== -1 && end !== -1 && start < end) {
      try {
        const jsonStr = raw.slice(start, end + 1);
        return { ok: true, value: JSON.parse(jsonStr), repaired: true };
      } catch (e2) {
        return { ok: false, error: e2.message };
      }
    }
    return { ok: false, error: e.message };
  }
}

function createArrayObjectExtractor() {
  let buffer = "";
  let cursor = 0;
  let inString = false;
  let escape = false;
  let arrayStarted = false;
  let depth = 0;
  let objStart = null;

  return function feed(chunk) {
    buffer += chunk;
    const objects = [];

    for (let i = cursor; i < buffer.length; i += 1) {
      const ch = buffer[i];

      if (escape) {
        escape = false;
        continue;
      }

      if (inString) {
        if (ch === "\\\\") {
          escape = true;
        } else if (ch === "\"") {
          inString = false;
        }
        continue;
      }

      if (ch === "\"") {
        inString = true;
        continue;
      }

      if (!arrayStarted) {
        if (ch === "[") arrayStarted = true;
        continue;
      }

      if (ch === "{") {
        if (depth === 0) objStart = i;
        depth += 1;
        continue;
      }

      if (ch === "}") {
        if (depth > 0) {
          depth -= 1;
          if (depth === 0 && objStart !== null) {
            objects.push(buffer.slice(objStart, i + 1));
            objStart = null;
          }
        }
      }
    }

    cursor = buffer.length;

    if (objStart !== null && objStart > 0) {
      buffer = buffer.slice(objStart);
      cursor -= objStart;
      objStart = 0;
    } else if (objStart === null && buffer.length > 8192) {
      buffer = buffer.slice(-2048);
      cursor = buffer.length;
    }

    return objects;
  };
}

function resolveOptions(userOptions = {}) {
  const think = typeof userOptions.think === "string" ? userOptions.think : THINK_LEVEL;
  const options = {
    temperature: Number.isFinite(+userOptions.temperature) ? +userOptions.temperature : TEMPERATURE,
    num_predict: Number.isFinite(+userOptions.num_predict) ? +userOptions.num_predict : REVIEW_NUM_PREDICT,
    num_ctx: Number.isFinite(+userOptions.num_ctx) ? +userOptions.num_ctx : REVIEW_NUM_CTX,
    top_p: Number.isFinite(+userOptions.top_p) ? +userOptions.top_p : 0.9,
    repeat_penalty: Number.isFinite(+userOptions.repeat_penalty) ? +userOptions.repeat_penalty : 1.1,
  };
  return { think, options };
}

function resolveFormat(userOptions = {}) {
  const raw = typeof userOptions.format === "string"
    ? userOptions.format
    : STREAM_FORMAT_DEFAULT;
  const normalized = raw.toLowerCase();
  if (normalized === "schema") {
    return { format: REVIEW_SCHEMA, includeSchema: true, label: "schema" };
  }
  if (normalized === "json") {
    return { format: "json", includeSchema: false, label: "json" };
  }
  return { format: undefined, includeSchema: false, label: "none" };
}

function sendSse(res, event, data) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
  if (typeof res.flush === "function") {
    res.flush();
  }
}

/* ================= ROUTES ================= */

app.get("/api/smoke", async (req, res) => {
  try {
    const data = await withTimeout(
      ollamaChat({
        messages: [
          { role: "system", content: "Answer with ONE token only." },
          { role: "user", content: "2+2=?" },
        ],
        options: { num_predict: SMOKE_NUM_PREDICT, num_ctx: SMOKE_NUM_CTX, temperature: 0 },
        think: THINK_LEVEL,
      }),
      SMOKE_TIMEOUT_MS,
      "smoke"
    );

    // Ollama includes durations in the response  :contentReference[oaicite:9]{index=9}
    res.json({
      ok: true,
      model: MODEL,
      content: data?.message?.content ?? "",
      total_duration: data.total_duration,
      load_duration: data.load_duration,
      eval_count: data.eval_count,
      eval_duration: data.eval_duration,
    });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post("/api/review/stream", async (req, res) => {
  const trace = [];
  const tracePush = (level, msg, data) => trace.push(logEvent(level, msg, data));
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REVIEW_TIMEOUT_MS);
  let clientClosed = false;
  let streamClosed = false;
  const handleClientClose = () => {
    if (clientClosed) return;
    clientClosed = true;
    console.log(`[${now()}] Client closed connection.`);
    if (!res.writableEnded && !streamClosed) {
      controller.abort();
    }
  };
  req.on("aborted", handleClientClose);
  res.on("close", () => {
    if (res.writableEnded || streamClosed) return;
    handleClientClose();
  });

  console.log(`[${now()}] Starting review stream for ${req.body?.text?.length} chars...`);

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });
  if (res.flushHeaders) res.flushHeaders();
  if (res.socket && typeof res.socket.setNoDelay === "function") {
    res.socket.setNoDelay(true);
  }
  res.write(": stream open\n\n");
  if (typeof res.flush === "function") {
    res.flush();
  }

  const startedAt = Date.now();
  const pingId = setInterval(() => {
    if (res.writableEnded) return;
    sendSse(res, "ping", { ts: now(), elapsedMs: Date.now() - startedAt });
  }, 5000);

  let raw = "";
  let thinking = "";
  let prompt = "";
  let firstTokenAt = null;

  try {
    const text = req.body?.text;
    if (!text) {
      sendSse(res, "error", { error: "Missing text" });
      return;
    }
    const userOptions = req.body?.options ?? {};
    const cabinet = readCabinet();
    
    // Filter reviewers if specific IDs are requested
    const requestedIds = req.body?.reviewerIds;
    const activeCabinet = (Array.isArray(requestedIds) && requestedIds.length > 0)
      ? cabinet.filter(p => requestedIds.includes(p.id))
      : cabinet;

    const personas = activeCabinet.map(p => ({
      id: p.id,
      name: p.name,
      role: p.role,
      strategy: p.strategy,
      rubric: p.rubric,
    }));

    const { think, options } = resolveOptions(userOptions);
    const { format, includeSchema, label: formatLabel } = resolveFormat(userOptions);
    prompt = buildPrompt(personas, text, { includeSchema });

    tracePush("info", "review stream start", { reviewers: activeCabinet.length, prompt_len: prompt.length });
    sendSse(res, "start", {
      model: MODEL,
      reviewers: activeCabinet.length,
      request: { model: MODEL, think, options, format: formatLabel },
    });

    console.log(`[${now()}] Sending request to Ollama (stream: true): ${OLLAMA_URL}/api/chat`);
    const t0 = Date.now();
    const ollamaRes = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          { role: "system", content: "Return JSON only. No extra text." },
          { role: "user", content: prompt },
        ],
        stream: true,
        keep_alive: KEEP_ALIVE,
        format,
        options,
        think: think === "none" ? false : think,
      }),
      signal: controller.signal,
    });

    console.log(`[${now()}] Ollama responded with status: ${ollamaRes.status}`);
    if (!ollamaRes.ok) {
      const errText = await ollamaRes.text();
      throw new Error(`Ollama error ${ollamaRes.status}: ${errText}`);
    }
    tracePush("info", "ollama headers received", { ms: Date.now() - t0, status: ollamaRes.status });
    sendSse(res, "phase", { stage: "headers", ms: Date.now() - t0 });

    const decoder = new TextDecoder();
    let lineBuffer = "";
    const extract = createArrayObjectExtractor();
    const seenIds = new Set();
    const personaById = new Map(activeCabinet.map(p => [p.id, p]));
    let doneMeta = null;

    for await (const chunk of ollamaRes.body) {
      lineBuffer += decoder.decode(chunk);
      let idx;
      while ((idx = lineBuffer.indexOf("\n")) !== -1) {
        const line = lineBuffer.slice(0, idx).trim();
        lineBuffer = lineBuffer.slice(idx + 1);
        if (!line) continue;
        let part;
        try {
          part = JSON.parse(line);
        } catch {
          continue;
        }

        if (part.message?.content) {
          if (!firstTokenAt) {
            firstTokenAt = Date.now();
            tracePush("info", "first content token", { ms: firstTokenAt - t0 });
            sendSse(res, "first_token", { type: "content", ms: firstTokenAt - t0 });
          }
          raw += part.message.content;
          sendSse(res, "raw", { delta: part.message.content, totalLen: raw.length });
          const objects = extract(part.message.content);
          for (const objStr of objects) {
            try {
              const review = JSON.parse(objStr);
              if (review?.id && !seenIds.has(review.id)) {
                seenIds.add(review.id);
                const persona = personaById.get(review.id);
                const validation = validateReview(review);
                sendSse(res, "review", {
                  person: persona
                    ? { id: persona.id, name: persona.name, role: persona.role, color: persona.color, icon: persona.icon }
                    : { id: review.id, name: review.id, role: "Unknown", color: "#111827", icon: "?" },
                  review,
                  validation,
                  ms: Math.round((Date.now() - t0) / Math.max(1, seenIds.size)),
                });
              }
            } catch {
              // Ignore partial object parse errors.
            }
          }
        }

        if (part.message?.thinking) {
          if (!firstTokenAt) {
            firstTokenAt = Date.now();
            tracePush("info", "first thinking token", { ms: firstTokenAt - t0 });
            sendSse(res, "first_token", { type: "thinking", ms: firstTokenAt - t0 });
          }
          thinking += part.message.thinking;
          sendSse(res, "thinking", { delta: part.message.thinking, totalLen: thinking.length });
        }

        if (part.done) {
          doneMeta = {
            done_reason: part.done_reason,
            total_duration: part.total_duration,
            load_duration: part.load_duration,
            eval_count: part.eval_count,
            eval_duration: part.eval_duration,
          };
        }
      }
    }

    const parsedAttempt = safeJsonParse(raw);
    if (!parsedAttempt.ok) {
      tracePush("error", "parse failed", { error: parsedAttempt.error });
      sendSse(res, "error", { error: `Model did not return valid JSON. ${parsedAttempt.error}`, raw, thinking, prompt, trace });
      return;
    }

    const parsed = parsedAttempt.value;
    let reviews = [];
    if (Array.isArray(parsed)) {
      reviews = parsed;
    } else if (parsed && typeof parsed === "object") {
      if (parsed.id) {
        reviews = [parsed];
      } else {
        const key = Object.keys(parsed).find(k => Array.isArray(parsed[k]));
        reviews = key ? parsed[key] : [];
      }
    }
    const results = [];
    const errors = [];

    for (const p of activeCabinet) {
      const r = reviews.find(x => x && x.id === p.id);
      if (!r) {
        errors.push({ id: p.id, name: p.name, role: p.role, error: "Missing review" });
        continue;
      }
      const validation = validateReview(r);
      results.push({
        person: { id: p.id, name: p.name, role: p.role, color: p.color, icon: p.icon },
        review: r,
        validation,
        ms: Math.round((Date.now() - t0) / Math.max(1, activeCabinet.length)),
      });
    }

    tracePush("info", "review stream done", {
      msTotal: Date.now() - t0,
      load_duration: doneMeta?.load_duration,
      eval_count: doneMeta?.eval_count,
      eval_duration: doneMeta?.eval_duration,
    });

    sendSse(res, "done", {
      model: MODEL,
      reviewers: cabinet.length,
      msTotal: Date.now() - t0,
      ok: results.length,
      err: errors.length,
      results,
      errors,
      raw,
      thinking,
      prompt,
      trace,
      durations: doneMeta ?? {},
      request: { model: MODEL, think, options },
    });
  } catch (e) {
    let message = e.message;
    if (e.name === "AbortError") {
      message = clientClosed
        ? "Client closed connection (request cancelled)"
        : `Timeout after ${Date.now() - t0}ms (Limit: ${REVIEW_TIMEOUT_MS}ms)`;
    }
    sendSse(res, "error", { error: message, raw, thinking, prompt, trace });
  } finally {
    clearTimeout(timeoutId);
    clearInterval(pingId);
    streamClosed = true;
    res.end();
  }
});

app.post("/api/review", async (req, res) => {
  const trace = [];
  const tracePush = (level, msg, data) => trace.push(logEvent(level, msg, data));
  try {
    const text = req.body?.text;
    if (!text) return res.status(400).json({ error: "Missing text" });
    const userOptions = req.body?.options ?? {};

    const cabinet = readCabinet();

    // Keep persona info short
    const personas = cabinet.map(p => ({
      id: p.id,
      name: p.name,
      role: p.role,
      strategy: p.strategy,
      rubric: p.rubric,
    }));

    const prompt = buildPrompt(personas, text);

    tracePush("info", "review start", { reviewers: cabinet.length, prompt_len: prompt.length });

    const t0 = Date.now();
    const { think: reqThink, options: reqOptions } = resolveOptions(userOptions);

    const data = await withTimeout(
      ollamaChat({
        messages: [
          { role: "system", content: "Return JSON only. No extra text." },
          { role: "user", content: prompt },
        ],
        format: REVIEW_SCHEMA,
        options: reqOptions,
        think: reqThink,
      }),
      REVIEW_TIMEOUT_MS,
      "review"
    );
    const msTotal = Date.now() - t0;

    tracePush("info", "ollama response", {
      total_duration: data.total_duration,
      load_duration: data.load_duration,
      eval_count: data.eval_count,
      eval_duration: data.eval_duration,
      done_reason: data.done_reason,
    });

    const raw = data?.message?.content ?? "";
    const thinking = data?.message?.thinking ?? "";

    if (!raw) {
      tracePush("warn", "empty content", { thinking_len: thinking.length });
    }

    const parsedAttempt = safeJsonParse(raw);
    if (!parsedAttempt.ok) {
      tracePush("error", "parse failed", { error: parsedAttempt.error });
      const err = new Error(`Model did not return valid JSON. ${parsedAttempt.error}`);
      err.raw = raw || thinking;
      err.thinking = thinking;
      err.prompt = prompt;
      throw err;
    }
    const parsed = parsedAttempt.value;

    const reviews = Array.isArray(parsed) ? parsed : [];
    const results = [];
    const errors = [];

    for (const p of cabinet) {
      const r = reviews.find(x => x && x.id === p.id);
      if (!r) {
        errors.push({ id: p.id, name: p.name, role: p.role, error: "Missing review" });
        continue;
      }
      const validation = validateReview(r);
      results.push({
        person: { id: p.id, name: p.name, role: p.role, color: p.color, icon: p.icon },
        review: r,
        validation,
        ms: Math.round(msTotal / Math.max(1, cabinet.length)),
      });
    }

    tracePush("info", "review done", {
      msTotal,
      load_duration: data.load_duration,
      eval_count: data.eval_count,
      eval_duration: data.eval_duration,
    });

    res.json({
      model: MODEL,
      reviewers: cabinet.length,
      msTotal,
      ok: results.length,
      err: errors.length,
      results,
      errors,
      raw,
      thinking,
      prompt,
      trace,
      durations: {
        total_duration: data.total_duration,
        load_duration: data.load_duration,
        eval_count: data.eval_count,
        eval_duration: data.eval_duration,
      },
      request: {
        model: MODEL,
        think: reqThink,
        options: reqOptions,
        format: "schema",
      },
    });
  } catch (e) {
    res.status(500).json({
      error: e.message,
      raw: e.raw,
      thinking: e.thinking,
      prompt: e.prompt,
      trace,
    });
  }
});

app.get("/api/cabinet", (req, res) => {
  try {
    const cabinet = readCabinet();
    res.json({ cabinet });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/logs", (req, res) => {
  res.json({ logs: logBuffer });
});

app.listen(3000, async () => {
  console.log("Open http://localhost:3000");
  console.log(`Using Ollama model: ${MODEL}`);
  await preloadModel();
});

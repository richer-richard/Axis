// Simple Node/Express backend for Axis AI endpoints (DeepSeek / OpenAI / Gemini).
// IMPORTANT: Do NOT hard-code API keys here. Use environment variables in `.env`.

const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const { z } = require("zod");
const crypto = require("crypto");
const path = require("path");
const fs = require("fs").promises;
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const { SocksProxyAgent } = require("socks-proxy-agent");
const { HttpsProxyAgent } = require("https-proxy-agent");
const { fetch: undiciFetch, Agent } = require("undici");
const nodeFetch = require("node-fetch");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 3000;
const USERS_FILE = path.join(__dirname, "users.json");
const USER_DATA_DIR = path.join(__dirname, "user_data");

// Auto-generate JWT_SECRET if not set or too short
const JWT_SECRET = (() => {
  const envSecret = process.env.JWT_SECRET;
  if (envSecret && envSecret.trim().length >= 32) {
    return envSecret.trim();
  }
  // Generate a secure random secret for this session
  const generatedSecret = crypto.randomBytes(48).toString("base64");
  console.log("⚠️  JWT_SECRET not set or too short. Auto-generated a secure secret for this session.");
  console.log("   Note: Users will need to re-login if the server restarts.");
  console.log("   For persistent sessions, add JWT_SECRET to your .env file (>= 32 chars).");
  return generatedSecret;
})();

function normalizeApiKey(key) {
  if (!key) return "";
  return String(key).trim().replace(/^Bearer\s+/i, "");
}

function maskApiKey(key) {
  if (!key) return "";
  if (key.length <= 8) return "********";
  return `${key.slice(0, 3)}…${key.slice(-4)}`;
}

const LLM_PROVIDER = (process.env.LLM_PROVIDER || process.env.AI_PROVIDER || "deepseek")
  .trim()
  .toLowerCase();

const DEEPSEEK_API_KEY = normalizeApiKey(process.env.DEEPSEEK_API_KEY);
const DEEPSEEK_BASE_URL =
  (process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com/v1/chat/completions").trim();
const DEEPSEEK_MODEL = (process.env.DEEPSEEK_MODEL || "deepseek-chat").trim();

const OPENAI_API_KEY = normalizeApiKey(process.env.OPENAI_API_KEY);
const OPENAI_BASE_URL =
  (process.env.OPENAI_BASE_URL || "https://api.openai.com/v1/chat/completions").trim();
const OPENAI_MODEL = (process.env.OPENAI_MODEL || "gpt-5.2").trim();

const GEMINI_API_KEY = normalizeApiKey(process.env.GEMINI_API_KEY);
const GEMINI_BASE_URL = (process.env.GEMINI_BASE_URL || "https://generativelanguage.googleapis.com/v1beta")
  .trim()
  .replace(/\/+$/, "");
const GEMINI_MODEL = (process.env.GEMINI_MODEL || "gemini-3-pro-preview").trim();

// ---------- Proxy Configuration ----------
const PROXY_URL = (process.env.PROXY_URL || "").trim();

function getProxyAgent() {
  if (!PROXY_URL) return undefined;
  if (PROXY_URL.startsWith("socks5://") || PROXY_URL.startsWith("socks5h://") || PROXY_URL.startsWith("socks4://")) {
    return new SocksProxyAgent(PROXY_URL);
  }
  if (PROXY_URL.startsWith("http://") || PROXY_URL.startsWith("https://")) {
    return new HttpsProxyAgent(PROXY_URL);
  }
  console.warn(`⚠️  WARNING: Invalid PROXY_URL format: ${PROXY_URL}. Proxy disabled.`);
  return undefined;
}

const proxyAgent = getProxyAgent();
if (proxyAgent) {
  console.log(`✓ Proxy configured: ${PROXY_URL.replace(/\/\/[^:]+:[^@]+@/, "//***:***@")}`);
}

const SUPPORTED_LLM_PROVIDERS = new Set(["deepseek", "openai", "gemini"]);
if (!SUPPORTED_LLM_PROVIDERS.has(LLM_PROVIDER)) {
  console.warn(`⚠️  WARNING: Unsupported LLM_PROVIDER "${LLM_PROVIDER}". Falling back to "deepseek".`);
}

if (LLM_PROVIDER === "deepseek" && (!DEEPSEEK_API_KEY || DEEPSEEK_API_KEY === "your_deepseek_api_key_here")) {
  console.warn(
    "⚠️  WARNING: DEEPSEEK_API_KEY is not set or still has placeholder value.",
  );
  console.warn(
    "   Please edit the .env file and add your actual DeepSeek API key.",
  );
  console.warn(
    "   Get your API key from: https://platform.deepseek.com/",
  );
  console.warn(
    "   DEEPSEEK_API_KEY: ", maskApiKey(DEEPSEEK_API_KEY),
  );
}

if (LLM_PROVIDER === "openai" && (!OPENAI_API_KEY || OPENAI_API_KEY === "your_openai_api_key_here")) {
  console.warn(
    "⚠️  WARNING: OPENAI_API_KEY is not set or still has placeholder value.",
  );
  console.warn(
    "   Please edit the .env file and add your actual OpenAI API key.",
  );
  console.warn(
    "   Get your API key from: https://platform.openai.com/",
  );
  console.warn(
    "   OPENAI_API_KEY: ", maskApiKey(OPENAI_API_KEY),
  );
}

if (LLM_PROVIDER === "gemini" && (!GEMINI_API_KEY || GEMINI_API_KEY === "your_gemini_api_key_here")) {
  console.warn(
    "⚠️  WARNING: GEMINI_API_KEY is not set or still has placeholder value.",
  );
  console.warn(
    "   Please edit the .env file and add your actual Gemini API key.",
  );
  console.warn(
    "   Get your API key from: https://aistudio.google.com/app/apikey",
  );
  console.warn(
    "   GEMINI_API_KEY: ", maskApiKey(GEMINI_API_KEY),
  );
}

function normalizeProvider(value) {
  const raw = String(value || "").trim().toLowerCase();
  return SUPPORTED_LLM_PROVIDERS.has(raw) ? raw : "";
}

function isProviderConfigured(provider) {
  const normalized = normalizeProvider(provider);
  if (normalized === "deepseek") {
    return Boolean(DEEPSEEK_API_KEY && DEEPSEEK_API_KEY !== "your_deepseek_api_key_here");
  }
  if (normalized === "openai") {
    return Boolean(OPENAI_API_KEY && OPENAI_API_KEY !== "your_openai_api_key_here");
  }
  if (normalized === "gemini") {
    return Boolean(GEMINI_API_KEY && GEMINI_API_KEY !== "your_gemini_api_key_here");
  }
  return false;
}

function listConfiguredProviders() {
  return Array.from(SUPPORTED_LLM_PROVIDERS).filter(isProviderConfigured).sort();
}

function resolveProviderForUserData(data) {
  const safeData = ensureAxisUserDataShape(data);
  const selected = normalizeProvider(safeData?.settings?.aiProvider);
  if (selected && isProviderConfigured(selected)) return selected;

  const envDefault = normalizeProvider(LLM_PROVIDER) || "deepseek";
  if (isProviderConfigured(envDefault)) return envDefault;

  const configured = listConfiguredProviders();
  if (configured.length) return configured[0];

  return envDefault;
}

function resolveProviderForRequest(requestedProvider) {
  const requested = normalizeProvider(requestedProvider);
  if (requested && isProviderConfigured(requested)) return requested;

  const envDefault = normalizeProvider(LLM_PROVIDER) || "deepseek";
  if (isProviderConfigured(envDefault)) return envDefault;

  const configured = listConfiguredProviders();
  if (configured.length) return configured[0];

  return requested || envDefault;
}

// ---------- LLM helpers ----------
function safeParseJSON(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function extractFirstJsonObject(text) {
  const input = String(text || "");
  const start = input.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < input.length; i++) {
    const ch = input[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }

    if (ch === '"') {
      inString = true;
      continue;
    }

    if (ch === "{") depth += 1;
    if (ch === "}") depth -= 1;
    if (depth === 0) {
      return input.slice(start, i + 1);
    }
  }
  return null;
}

function safeParseJSONFromText(text) {
  if (text === undefined || text === null) return null;
  const raw = String(text).trim();
  const direct = safeParseJSON(raw);
  if (direct) return direct;

  const withoutFences = raw.replace(/```(?:json)?/gi, "").trim();
  const fenceParsed = safeParseJSON(withoutFences);
  if (fenceParsed) return fenceParsed;

  const extracted = extractFirstJsonObject(raw) || extractFirstJsonObject(withoutFences);
  if (extracted) return safeParseJSON(extracted);
  return null;
}

async function callLlmWithJsonRepair({
  provider,
  system,
  user,
  temperature = 0.2,
  maxTokens = 900,
  schema,
  schemaHint,
}) {
  const raw = await callLlm({
    provider,
    system,
    user,
    temperature,
    maxTokens,
    expectJSON: true,
  });

  const parsed = safeParseJSONFromText(raw);
  const validated = schema.safeParse(parsed);
  if (validated.success) {
    return { ok: true, raw, data: validated.data, repaired: false };
  }

  const repairSystem =
    "You are a strict JSON formatter. Output ONLY valid JSON. No markdown fences, no commentary, no trailing commas.";
  const repairUser = [
    "Fix the following text into strict JSON that matches this schema:",
    schemaHint ? `Schema: ${schemaHint}` : "",
    "",
    "Text to fix:",
    raw,
  ]
    .filter(Boolean)
    .join("\n");

  const repairedRaw = await callLlm({
    provider,
    system: repairSystem,
    user: repairUser,
    temperature: 0,
    maxTokens: Math.max(300, Math.min(maxTokens, 1200)),
    expectJSON: true,
  });

  const repairedParsed = safeParseJSONFromText(repairedRaw);
  const repairedValidated = schema.safeParse(repairedParsed);
  if (repairedValidated.success) {
    return { ok: true, raw: repairedRaw, data: repairedValidated.data, repaired: true };
  }

  return { ok: false, raw, repairedRaw };
}

// ---------- iCalendar (.ics) helpers ----------
function icsEscapeText(value) {
  return String(value ?? "")
    .replace(/\\/g, "\\\\")
    .replace(/\r/g, "")
    .replace(/\n/g, "\\n")
    .replace(/;/g, "\\;")
    .replace(/,/g, "\\,");
}

function formatIcsDateTimeUtc(date) {
  const iso = date.toISOString(); // 2026-01-05T14:00:00.000Z
  return iso.replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function foldIcsLine(line) {
  const parts = [];
  let remaining = String(line);
  // RFC 5545: 75 octets; approximate with 75 chars for ASCII output.
  while (remaining.length > 75) {
    parts.push(remaining.slice(0, 75));
    remaining = ` ${remaining.slice(75)}`;
  }
  parts.push(remaining);
  return parts;
}

function buildAxisIcs({ userId, data, options }) {
  const includeFixedBlocks = options?.includeFixedBlocks !== false;
  const includeCompletedTasks = options?.includeCompletedTasks === true;
  const reminderMinutes = Number.isFinite(options?.reminderMinutes)
    ? Math.max(0, Math.min(240, Math.round(options.reminderMinutes)))
    : 15;

  const tasks = Array.isArray(data?.tasks) ? data.tasks : [];
  const tasksById = new Map(tasks.map((t) => [t.id, t]));
  const completedTaskIds = new Set(tasks.filter((t) => t?.completed).map((t) => t.id));

  const scheduleBlocks = Array.isArray(data?.schedule) ? data.schedule : [];
  const fixedBlocks = includeFixedBlocks && Array.isArray(data?.fixedBlocks) ? data.fixedBlocks : [];

  const now = new Date();
  const dtStamp = formatIcsDateTimeUtc(now);
  const calName = "Axis Schedule";

  const lines = [];
  const push = (line) => foldIcsLine(line).forEach((l) => lines.push(l));

  push("BEGIN:VCALENDAR");
  push("VERSION:2.0");
  push("CALSCALE:GREGORIAN");
  push("METHOD:PUBLISH");
  push("PRODID:-//Axis//EN");
  push(`X-WR-CALNAME:${icsEscapeText(calName)}`);
  push("X-WR-TIMEZONE:UTC");

  const addEvent = ({ summary, description, start, end, categories, uidSeed }) => {
    const uid = crypto
      .createHash("sha1")
      .update(String(uidSeed || `${userId}:${summary}:${start.toISOString()}:${end.toISOString()}`))
      .digest("hex");

    push("BEGIN:VEVENT");
    push(`UID:${uid}@axis`);
    push(`DTSTAMP:${dtStamp}`);
    push(`DTSTART:${formatIcsDateTimeUtc(start)}`);
    push(`DTEND:${formatIcsDateTimeUtc(end)}`);
    push(`SUMMARY:${icsEscapeText(summary)}`);
    if (description) push(`DESCRIPTION:${icsEscapeText(description)}`);
    if (categories?.length) push(`CATEGORIES:${categories.map((c) => icsEscapeText(c)).join(",")}`);

    if (reminderMinutes > 0) {
      push("BEGIN:VALARM");
      push("ACTION:DISPLAY");
      push(`DESCRIPTION:${icsEscapeText(summary)}`);
      push(`TRIGGER:-PT${reminderMinutes}M`);
      push("END:VALARM");
    }

    push("END:VEVENT");
  };

  scheduleBlocks.forEach((b) => {
    const taskId = b?.taskId;
    if (!taskId) return;
    if (!includeCompletedTasks && completedTaskIds.has(taskId)) return;
    const start = new Date(b.start);
    const end = new Date(b.end);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime()) || end <= start) return;

    const task = tasksById.get(taskId) || null;
    const summary = task?.task_name ? String(task.task_name) : "Task";
    const category = task?.task_category ? String(task.task_category) : "";
    const priority = task?.task_priority ? String(task.task_priority) : "";
    const deadline = task?.task_deadline ? `${task.task_deadline} ${task.task_deadline_time || ""}`.trim() : "";

    const descriptionParts = [];
    if (category) descriptionParts.push(`Category: ${category}`);
    if (priority) descriptionParts.push(`Priority: ${priority}`);
    if (deadline) descriptionParts.push(`Deadline: ${deadline}`);

    addEvent({
      summary,
      description: descriptionParts.join("\n"),
      start,
      end,
      categories: category ? [category] : [],
      uidSeed: `${userId}:task:${taskId}:${start.toISOString()}:${end.toISOString()}`,
    });
  });

  fixedBlocks.forEach((b) => {
    const start = new Date(b.start);
    const end = new Date(b.end);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime()) || end <= start) return;

    const label = String(b.label || "Fixed block");
    const category = b.category ? String(b.category) : "";
    addEvent({
      summary: label,
      description: category ? `Category: ${category}` : "",
      start,
      end,
      categories: category ? [category] : [],
      uidSeed: `${userId}:fixed:${label}:${start.toISOString()}:${end.toISOString()}`,
    });
  });

  push("END:VCALENDAR");
  return `${lines.join("\r\n")}\r\n`;
}

async function callDeepSeek({
  system,
  user,
  temperature = 0.35,
  maxTokens = 900,
  expectJSON = false,
}) {
  if (!DEEPSEEK_API_KEY) {
    throw new Error("DEEPSEEK_API_KEY is not configured on the server.");
  }

  const basePayload = {
    model: DEEPSEEK_MODEL,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    temperature,
    max_tokens: maxTokens,
  };

  const payload = expectJSON
    ? {
        ...basePayload,
        // DeepSeek supports OpenAI-style response_format on newer models.
        response_format: { type: "json_object" },
      }
    : basePayload;

  const requestInit = (body) => ({
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  console.log("[DeepSeek] Request to:", DEEPSEEK_BASE_URL);
  console.log("[DeepSeek] Model:", DEEPSEEK_MODEL);

  let response;
  try {
    response = await fetch(DEEPSEEK_BASE_URL, requestInit(payload));
  } catch (fetchErr) {
    console.error("[DeepSeek] Fetch failed:", fetchErr.message);
    console.error("[DeepSeek] Fetch error cause:", fetchErr.cause);
    throw fetchErr;
  }
  
  let errorText = "";

  if (!response.ok && expectJSON) {
    errorText = await response.text();
    const parsed = safeParseJSON(errorText);
    const message = parsed?.error?.message || errorText;
    // Some DeepSeek deployments reject response_format; retry without it.
    if (/response_format|json_object|response format/i.test(message)) {
      console.log("[DeepSeek] Retrying without response_format...");
      try {
        response = await fetch(DEEPSEEK_BASE_URL, requestInit(basePayload));
      } catch (retryErr) {
        console.error("[DeepSeek] Retry fetch failed:", retryErr.message);
        console.error("[DeepSeek] Retry fetch error cause:", retryErr.cause);
        throw retryErr;
      }
      errorText = "";
    }
  }

  if (!response.ok) {
    if (!errorText) errorText = await response.text();
    console.error("[DeepSeek] API error status:", response.status);
    console.error("[DeepSeek] API error headers:", Object.fromEntries(response.headers.entries()));
    console.error("[DeepSeek] API error body:", errorText);
    let errorMessage = "Upstream DeepSeek API error.";
    const parsed = safeParseJSON(errorText);
    if (parsed?.error?.message) {
      errorMessage = parsed.error.message;
    }
    throw new Error(`${errorMessage} (status ${response.status})`);
  }

  const data = await response.json();
  const reply = data.choices?.[0]?.message?.content;
  if (!reply) {
    throw new Error("DeepSeek reply missing content");
  }
  return reply.trim();
}

async function callOpenAI({
  system,
  user,
  temperature = 0.35,
  maxTokens = 900,
  expectJSON = false,
}) {
  if (!OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not configured on the server.");
  }

  // Detect if using the Responses API (URL ends with /responses)
  const isResponsesApi = OPENAI_BASE_URL.includes("/responses");

  let payload;
  if (isResponsesApi) {
    // OpenAI Responses API format
    payload = {
      model: OPENAI_MODEL,
      input: user,
      instructions: system,
      temperature,
      max_output_tokens: maxTokens,
      stream: true,
    };

    if (expectJSON) {
      payload.text = { format: { type: "json_object" } };
    }
  } else {
    // Standard Chat Completions API format
    payload = {
      model: OPENAI_MODEL,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      max_tokens: maxTokens,
    };

    if (expectJSON) {
      payload.response_format = { type: "json_object" };
    }
  }

  console.log("[OpenAI] Request to:", OPENAI_BASE_URL);
  console.log("[OpenAI] Model:", OPENAI_MODEL);
  console.log("[OpenAI] Responses API mode:", isResponsesApi);
  console.log("[OpenAI] Streaming enabled:", isResponsesApi);

  const MAX_RETRIES = 3;
  let lastErr;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 60000); // 60s timeout

      const fetchOptions = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_API_KEY}`,
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      };
      if (proxyAgent) fetchOptions.dispatcher = proxyAgent;

      const response = await fetch(OPENAI_BASE_URL, fetchOptions);

      clearTimeout(timeout);

      if (!response.ok) {
        const text = await response.text();
        console.error("[OpenAI] API error status:", response.status);
        console.error("[OpenAI] API error body:", text);
        let errorMessage = "Upstream OpenAI API error.";
        const parsed = safeParseJSON(text);
        if (parsed?.error?.message) {
          errorMessage = parsed.error.message;
        }
        throw new Error(`${errorMessage} (status ${response.status})`);
      }

      // For streaming responses, collect all chunks
      if (isResponsesApi && payload.stream) {
        let fullText = "";
        await readSseEvents(response, async ({ data }) => {
          const trimmed = String(data || "").trim();
          if (!trimmed || trimmed === "[DONE]") return trimmed !== "[DONE]";
          const parsed = safeParseJSON(trimmed);
          if (parsed?.delta) {
            fullText += parsed.delta;
          } else if (parsed?.type === "response.output_text.done" && parsed?.text) {
            fullText = parsed.text;
          }
          return true;
        });
        if (!fullText) throw new Error("OpenAI streaming reply empty");
        return fullText.trim();
      }

      const data = await response.json();

      // Parse response based on API type
      let reply;
      if (isResponsesApi) {
        reply = data.output?.[0]?.content?.[0]?.text || data.output_text;
        if (!reply && Array.isArray(data.output)) {
          for (const item of data.output) {
            if (item.type === "message" && Array.isArray(item.content)) {
              for (const content of item.content) {
                if (content.type === "output_text" || content.type === "text") {
                  reply = content.text;
                  break;
                }
              }
            }
            if (reply) break;
          }
        }
      } else {
        reply = data.choices?.[0]?.message?.content;
      }

      if (!reply) throw new Error("OpenAI reply missing content");
      return reply.trim();
    } catch (err) {
      lastErr = err;
      const isRetryable = err.code === "ECONNRESET" || err.name === "AbortError" || /ETIMEDOUT|ENOTFOUND|socket/.test(err.message);
      console.error(`[OpenAI] Attempt ${attempt}/${MAX_RETRIES} failed:`, err.message);
      if (isRetryable && attempt < MAX_RETRIES) {
        const delay = 1000 * attempt;
        console.log(`[OpenAI] Retrying in ${delay}ms...`);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw err;
    }
  }
  throw lastErr;
}

async function callGemini({
  system,
  user,
  temperature = 0.35,
  maxTokens = 8192,
  expectJSON = false,
}) {
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not configured on the server.");
  }

  const modelName = GEMINI_MODEL.replace(/^models\//, "");
  const endpoint = `${GEMINI_BASE_URL}/models/${encodeURIComponent(modelName)}:generateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`;

  const payload = {
    ...(system
      ? {
          systemInstruction: {
            parts: [{ text: system }],
          },
        }
      : {}),
    contents: [
      {
        role: "user",
        parts: [{ text: user }],
      },
    ],
    generationConfig: {
      temperature,
      maxOutputTokens: maxTokens,
      ...(expectJSON ? { responseMimeType: "application/json" } : {}),
    },
  };

  console.log("[Gemini] Request to:", endpoint.replace(/key=[^&]+/, "key=***"));
  console.log("[Gemini] Model:", modelName);
  console.log("[Gemini] Payload keys:", Object.keys(payload));
  console.log("[Gemini] Using proxy:", proxyAgent ? "yes" : "no");

  const MAX_RETRIES = 3;
  let lastErr;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    let response;
    try {
      const fetchOptions = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      };
      // Use node-fetch with agent option for proxy support (native fetch doesn't support agent)
      if (proxyAgent) {
        fetchOptions.agent = proxyAgent;
        response = await nodeFetch(endpoint, fetchOptions);
      } else {
        response = await fetch(endpoint, fetchOptions);
      }
    } catch (fetchErr) {
      lastErr = fetchErr;
      console.error(`[Gemini] Attempt ${attempt}/${MAX_RETRIES} fetch failed:`, fetchErr.message);
      console.error("[Gemini] Fetch error cause:", fetchErr.cause);
      const isRetryable = fetchErr.code === "ECONNRESET" || /ETIMEDOUT|ENOTFOUND|socket/i.test(fetchErr.message);
      if (isRetryable && attempt < MAX_RETRIES) {
        const delay = 1000 * attempt;
        console.log(`[Gemini] Retrying in ${delay}ms...`);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw fetchErr;
    }

    if (!response.ok) {
      const text = await response.text();
      console.error("[Gemini] API error status:", response.status);
      console.error("[Gemini] API error headers:", Object.fromEntries(response.headers.entries()));
      console.error("[Gemini] API error body:", text);
      let errorMessage = "Upstream Gemini API error.";
      const parsed = safeParseJSON(text);
      if (parsed?.error?.message) {
        errorMessage = parsed.error.message;
      }
      lastErr = new Error(`${errorMessage} (status ${response.status})`);
      // Retry on 500/502/503/504 errors
      const isRetryable = response.status >= 500 && response.status < 600;
      if (isRetryable && attempt < MAX_RETRIES) {
        const delay = 1000 * attempt;
        console.log(`[Gemini] Retrying in ${delay}ms...`);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw lastErr;
    }

    const data = await response.json();
    
    // Check for safety block
    const blockReason = data.promptFeedback?.blockReason;
    if (blockReason) {
      console.error("[Gemini] Prompt blocked by safety filter:", blockReason);
      throw new Error(`Gemini blocked the prompt: ${blockReason}`);
    }

    const candidate = data.candidates?.[0];
    const finishReason = candidate?.finishReason;
    const parts = candidate?.content?.parts;
    const reply = Array.isArray(parts) ? parts.map((p) => p?.text || "").join("") : "";
    
    if (!reply) {
      // Log detailed info for debugging the known Gemini empty response bug
      console.error(`[Gemini] Attempt ${attempt}/${MAX_RETRIES} - Empty content received`);
      console.error("[Gemini] finishReason:", finishReason);
      console.error("[Gemini] candidates:", JSON.stringify(data.candidates, null, 2));
      console.error("[Gemini] promptFeedback:", JSON.stringify(data.promptFeedback, null, 2));
      
      lastErr = new Error("Gemini reply missing content");
      // Known Gemini bug: empty response with STOP finish reason - retry
      if (attempt < MAX_RETRIES) {
        const delay = 1000 * attempt;
        console.log(`[Gemini] Retrying in ${delay}ms due to empty response bug...`);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw lastErr;
    }
    
    return reply.trim();
  }
  
  throw lastErr || new Error("Gemini request failed after retries");
}

async function callLlm(options) {
  const provider = resolveProviderForRequest(options?.provider || LLM_PROVIDER || "deepseek");

  if (provider === "openai") return callOpenAI(options);
  if (provider === "gemini") return callGemini(options);
  return callDeepSeek(options);
}

function setSseHeaders(res) {
  res.status(200);
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders?.();
}

function sseSend(res, event, payload) {
  if (event) res.write(`event: ${event}\n`);
  const data = payload === undefined ? {} : payload;
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

async function readSseEvents(response, onEvent) {
  if (!response.body) {
    throw new Error("Upstream response has no body to stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  let eventName = "";
  let dataLines = [];
  let stop = false;

  const dispatch = async () => {
    if (!dataLines.length) {
      eventName = "";
      return;
    }
    const data = dataLines.join("\n");
    dataLines = [];
    const keepGoing = await onEvent({ event: eventName || "message", data });
    eventName = "";
    if (keepGoing === false) stop = true;
  };

  while (!stop) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while (!stop && (idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx).replace(/\r$/, "");
      buffer = buffer.slice(idx + 1);

      if (line.startsWith("event:")) {
        eventName = line.slice("event:".length).trim();
        continue;
      }
      if (line.startsWith("data:")) {
        dataLines.push(line.slice("data:".length).trim());
        continue;
      }
      if (line.startsWith(":")) {
        continue;
      }
      if (line === "") {
        await dispatch();
      }
    }
  }

  if (!stop && buffer.length) {
    const leftover = buffer.replace(/\r$/, "").trim();
    if (leftover.startsWith("data:")) {
      dataLines.push(leftover.slice("data:".length).trim());
    } else if (leftover && leftover !== "[DONE]") {
      dataLines.push(leftover);
    }
  }

  if (!stop) {
    await dispatch();
  }

  try {
    await reader.cancel();
  } catch {}
}

async function readJsonLines(response, onJson) {
  if (!response.body) {
    throw new Error("Upstream response has no body to stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let stop = false;

  while (!stop) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx).replace(/\r$/, "").trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      const normalized = line.startsWith("data:") ? line.slice("data:".length).trim() : line;
      if (!normalized) continue;
      if (normalized === "[DONE]") {
        stop = true;
        break;
      }
      const parsed = safeParseJSONFromText(normalized);
      if (parsed) {
        const keepGoing = await onJson(parsed);
        if (keepGoing === false) {
          stop = true;
          break;
        }
      }
    }
  }

  if (!stop) {
    const rest = buffer.replace(/\r$/, "").trim();
    if (rest) {
      const normalized = rest.startsWith("data:") ? rest.slice("data:".length).trim() : rest;
      const parsed = safeParseJSONFromText(normalized);
      if (parsed) await onJson(parsed);
    }
  }

  try {
    await reader.cancel();
  } catch {}
}

async function callDeepSeekStream({
  system,
  user,
  temperature = 0.35,
  maxTokens = 900,
  onToken,
  signal,
}) {
  if (!DEEPSEEK_API_KEY) {
    throw new Error("DEEPSEEK_API_KEY is not configured on the server.");
  }

  const payload = {
    model: DEEPSEEK_MODEL,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    temperature,
    max_tokens: maxTokens,
    stream: true,
  };

  const response = await fetch(DEEPSEEK_BASE_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const text = await response.text();
    console.error("DeepSeek API error:", response.status, text);
    let errorMessage = "Upstream DeepSeek API error.";
    const parsed = safeParseJSON(text);
    if (parsed?.error?.message) {
      errorMessage = parsed.error.message;
    }
    throw new Error(`${errorMessage} (status ${response.status})`);
  }

  let full = "";
  await readSseEvents(response, async ({ data }) => {
    const trimmed = String(data || "").trim();
    if (!trimmed) return;
    if (trimmed === "[DONE]") return false;
    const parsed = safeParseJSON(trimmed);
    const delta = parsed?.choices?.[0]?.delta?.content;
    if (typeof delta === "string" && delta.length) {
      full += delta;
      try {
        onToken?.(delta);
      } catch {}
    }
  });

  return full;
}

async function callOpenAIStream({
  system,
  user,
  temperature = 0.35,
  maxTokens = 900,
  onToken,
  signal,
}) {
  if (!OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not configured on the server.");
  }

  const isResponsesApi = OPENAI_BASE_URL.includes("/responses");

  let payload;
  if (isResponsesApi) {
    payload = {
      model: OPENAI_MODEL,
      input: user,
      instructions: system,
      temperature,
      max_output_tokens: maxTokens,
      stream: true,
    };
  } else {
    payload = {
      model: OPENAI_MODEL,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      max_tokens: maxTokens,
      stream: true,
    };
  }

  const response = await fetch(OPENAI_BASE_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const text = await response.text();
    console.error("OpenAI-compatible API error:", response.status, text);
    let errorMessage = "Upstream OpenAI-compatible API error.";
    const parsed = safeParseJSON(text);
    if (parsed?.error?.message) {
      errorMessage = parsed.error.message;
    }
    throw new Error(`${errorMessage} (status ${response.status})`);
  }

  let full = "";
  let lastResponsesText = "";

  await readSseEvents(response, async ({ data }) => {
    const trimmed = String(data || "").trim();
    if (!trimmed) return;
    if (trimmed === "[DONE]") return false;
    const parsed = safeParseJSON(trimmed);
    if (!parsed) return;

    if (isResponsesApi) {
      if (typeof parsed.delta === "string" && parsed.delta.length) {
        full += parsed.delta;
        lastResponsesText += parsed.delta;
        try {
          onToken?.(parsed.delta);
        } catch {}
        return;
      }

      const maybeText =
        parsed.output?.[0]?.content?.[0]?.text ||
        parsed.output_text ||
        (Array.isArray(parsed.output)
          ? parsed.output
              .flatMap((item) => (Array.isArray(item?.content) ? item.content : []))
              .map((c) => c?.text || "")
              .join("")
          : "");

      if (typeof maybeText === "string" && maybeText && maybeText.startsWith(lastResponsesText)) {
        const delta = maybeText.slice(lastResponsesText.length);
        if (delta) {
          full += delta;
          lastResponsesText = maybeText;
          try {
            onToken?.(delta);
          } catch {}
        }
      }
      return;
    }

    const delta = parsed?.choices?.[0]?.delta?.content;
    if (typeof delta === "string" && delta.length) {
      full += delta;
      try {
        onToken?.(delta);
      } catch {}
    }
  });

  return full;
}

async function callGeminiStream({
  system,
  user,
  temperature = 0.35,
  maxTokens = 8192,
  onToken,
  signal,
}) {
  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not configured on the server.");
  }

  const modelName = GEMINI_MODEL.replace(/^models\//, "");
  const endpoint = `${GEMINI_BASE_URL}/models/${encodeURIComponent(modelName)}:streamGenerateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`;

  const payload = {
    ...(system
      ? {
          systemInstruction: {
            parts: [{ text: system }],
          },
        }
      : {}),
    contents: [
      {
        role: "user",
        parts: [{ text: user }],
      },
    ],
    generationConfig: {
      temperature,
      maxOutputTokens: maxTokens,
    },
  };

  console.log("[Gemini Stream] Request to:", endpoint.replace(/key=[^&]+/, "key=***"));
  console.log("[Gemini Stream] Model:", modelName);
  console.log("[Gemini Stream] Using proxy:", proxyAgent ? "yes" : "no");

  const MAX_RETRIES = 3;
  let lastErr;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    let response;
    try {
      const fetchOptions = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream, application/json",
        },
        body: JSON.stringify(payload),
        signal,
      };

      // Use node-fetch with agent option for proxy support
      if (proxyAgent) {
        fetchOptions.agent = proxyAgent;
        response = await nodeFetch(endpoint, fetchOptions);
      } else {
        response = await fetch(endpoint, fetchOptions);
      }
    } catch (fetchErr) {
      lastErr = fetchErr;
      // Don't retry if user aborted
      if (fetchErr.name === "AbortError") throw fetchErr;
      console.error(`[Gemini Stream] Attempt ${attempt}/${MAX_RETRIES} fetch failed:`, fetchErr.message);
      const isRetryable = fetchErr.code === "ECONNRESET" || /ETIMEDOUT|ENOTFOUND|socket/i.test(fetchErr.message);
      if (isRetryable && attempt < MAX_RETRIES) {
        const delay = 1000 * attempt;
        console.log(`[Gemini Stream] Retrying in ${delay}ms...`);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw fetchErr;
    }

    if (!response.ok) {
      const text = await response.text();
      console.error("[Gemini Stream] API error status:", response.status);
      console.error("[Gemini Stream] API error body:", text);
      let errorMessage = "Upstream Gemini API error.";
      const parsed = safeParseJSON(text);
      if (parsed?.error?.message) {
        errorMessage = parsed.error.message;
      }
      lastErr = new Error(`${errorMessage} (status ${response.status})`);
      // Retry on 500/502/503/504 errors
      const isRetryable = response.status >= 500 && response.status < 600;
      if (isRetryable && attempt < MAX_RETRIES) {
        const delay = 1000 * attempt;
        console.log(`[Gemini Stream] Retrying in ${delay}ms...`);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw lastErr;
    }

    let full = "";
    let last = "";
    let hasContent = false;
    
    await readJsonLines(response, async (chunk) => {
      // Check for safety block in streaming response
      const blockReason = chunk?.promptFeedback?.blockReason;
      if (blockReason) {
        console.error("[Gemini Stream] Prompt blocked by safety filter:", blockReason);
        throw new Error(`Gemini blocked the prompt: ${blockReason}`);
      }
      
      const parts = chunk?.candidates?.[0]?.content?.parts;
      const text = Array.isArray(parts) ? parts.map((p) => p?.text || "").join("") : "";
      if (!text) return;
      hasContent = true;
      if (text.startsWith(last)) {
        const delta = text.slice(last.length);
        if (delta) {
          full += delta;
          try {
            onToken?.(delta);
          } catch {}
        }
      } else {
        full += text;
        try {
          onToken?.(text);
        } catch {}
      }
      last = text;
    });

    // Check for empty response (known Gemini bug)
    if (!full && !hasContent) {
      console.error(`[Gemini Stream] Attempt ${attempt}/${MAX_RETRIES} - Empty content received`);
      lastErr = new Error("Gemini streaming reply empty");
      if (attempt < MAX_RETRIES) {
        const delay = 1000 * attempt;
        console.log(`[Gemini Stream] Retrying in ${delay}ms due to empty response bug...`);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw lastErr;
    }

    return full;
  }

  throw lastErr || new Error("Gemini streaming request failed after retries");
}

async function callLlmStream(options) {
  const provider = resolveProviderForRequest(options?.provider || LLM_PROVIDER || "deepseek");
  if (provider === "openai") return callOpenAIStream(options);
  if (provider === "gemini") return callGeminiStream(options);
  return callDeepSeekStream(options);
}

// --- Security / hardening middleware ---
app.disable("x-powered-by");

// In production you should lock this down to your real domain(s).
app.use(cors({ origin: true, credentials: false }));
app.use(helmet());
app.use(express.json({ limit: "256kb" }));

// Basic rate limits (tunable)
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 20,
  standardHeaders: true,
  legacyHeaders: false,
});

const apiLimiter = rateLimit({
  windowMs: 60 * 1000,
  limit: 120,
  standardHeaders: true,
  legacyHeaders: false,
});

app.use("/api/", apiLimiter);
app.use("/api/auth/", authLimiter);

function requireJwtSecret(res) {
  if (!JWT_SECRET || JWT_SECRET.trim().length < 32) {
    res
      .status(500)
      .json({ error: "Server misconfigured: JWT_SECRET must be set (>= 32 chars)." });
    return false;
  }
  return true;
}

// --- Request validation schemas ---
function emptyStringToUndefined(value) {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  return trimmed ? trimmed : undefined;
}

const identifierSchema = z.preprocess(emptyStringToUndefined, z.string().min(1).max(254).optional());
const optionalUsernameSchema = z.preprocess(
  emptyStringToUndefined,
  z
    .string()
    .min(3)
    .max(30)
    .regex(/^[a-zA-Z0-9_]+$/, "Username can only contain letters, numbers, and underscores")
    .optional(),
);

const registerSchema = z.object({
  email: z.string().trim().email().max(254),
  password: z.string().min(8).max(200),
  name: z.string().trim().min(1).max(100),
  username: optionalUsernameSchema,
});

const loginSchema = z
  .object({
    identifier: identifierSchema, // Can be name, username, or email
    email: identifierSchema, // Backwards-compatible alias for older clients
    password: z.string().min(1).max(200),
  })
  .refine((data) => Boolean(data.identifier || data.email), {
    message: "Missing identifier",
    path: ["identifier"],
  })
  .transform((data) => ({
    identifier: data.identifier || data.email,
    password: data.password,
  }));

const aiRescheduleSchema = z.object({
  tasks: z
    .array(
      z.object({
        id: z.string().min(1).max(200),
        task_name: z.string().optional(),
        task_priority: z.string().optional(),
        task_category: z.string().optional(),
        task_deadline: z.string().optional(),
        task_deadline_time: z.string().optional(),
        task_duration_hours: z.number().optional().nullable(),
        completed: z.boolean().optional(),
      }),
    )
    .max(800)
    .default([]),
  fixedBlocks: z
    .array(
      z.object({
        start: z.string().min(1).max(60),
        end: z.string().min(1).max(60),
        label: z.string().optional(),
        category: z.string().optional(),
        kind: z.string().optional(),
      }),
    )
    .max(1200)
    .default([]),
  schedule: z
    .array(
      z.object({
        taskId: z.string().optional(),
        start: z.string().min(1).max(60),
        end: z.string().min(1).max(60),
        kind: z.string().optional(),
      }),
    )
    .max(2000)
    .default([]),
  profile: z.any().optional(),
  horizonDays: z.number().int().min(1).max(21).default(7),
  maxHoursPerDay: z.number().min(1).max(16).default(10),
});

const calendarExportSchema = z.object({
  includeFixedBlocks: z.boolean().optional().default(true),
  includeCompletedTasks: z.boolean().optional().default(false),
  reminderMinutes: z.number().int().min(0).max(240).optional().default(15),
});

const assistantAgentRequestSchema = z.object({
  message: z.string().min(1).max(4000),
});

const taskCreateSchema = z.object({
  task_name: z.string().min(1).max(200),
  task_priority: z.string().optional(),
  task_category: z.string().optional(),
  task_deadline: z.string().optional(),
  task_deadline_time: z.string().optional(),
  task_duration_hours: z.number().optional().nullable(),
  computer_required: z.boolean().optional(),
});

const taskUpdateSchema = z.object({
  task_name: z.string().min(1).max(200).optional(),
  task_priority: z.string().optional(),
  task_category: z.string().optional(),
  task_deadline: z.string().optional(),
  task_deadline_time: z.string().optional(),
  task_duration_hours: z.number().optional().nullable(),
  computer_required: z.boolean().optional(),
  completed: z.boolean().optional(),
});

const habitCreateSchema = z.object({
  name: z.string().min(1).max(120),
  time: z.string().min(1).max(40),
  description: z.string().max(240).optional().default(""),
});

function validateBody(schema) {
  return (req, res, next) => {
    const parsed = schema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({
        error: "Invalid request body",
        details: parsed.error.issues.map((i) => ({ path: i.path, message: i.message })),
      });
    }
    req.body = parsed.data;
    next();
  };
}

// Ensure user_data directory exists and users file is initialized
(async () => {
  try {
    await fs.mkdir(USER_DATA_DIR, { recursive: true });
    console.log("✓ user_data directory ready");
  } catch (err) {
    console.error("Error creating user_data directory:", err);
  }
  
  try {
    await fs.access(USERS_FILE);
    console.log("✓ users.json file exists");
  } catch {
    await fs.writeFile(USERS_FILE, JSON.stringify({}, null, 2));
    console.log("✓ users.json file created");
  }
})();

// Helper functions for user management
async function getUsers() {
  try {
    const data = await fs.readFile(USERS_FILE, "utf8");
    return JSON.parse(data);
  } catch {
    return {};
  }
}

async function saveUsers(users) {
  await fs.writeFile(USERS_FILE, JSON.stringify(users, null, 2));
}

async function getUserData(userId) {
  const filePath = path.join(USER_DATA_DIR, `${userId}.json`);
  try {
    const data = await fs.readFile(filePath, "utf8");
    return JSON.parse(data);
  } catch {
    return null;
  }
}

async function getEffectiveProviderForUserId(userId) {
  const data = await getUserData(userId);
  return resolveProviderForUserData(data);
}

async function saveUserData(userId, data) {
  const filePath = path.join(USER_DATA_DIR, `${userId}.json`);
  await fs.writeFile(filePath, JSON.stringify(data, null, 2));
}

function ensureAxisUserDataShape(data) {
  if (!data || typeof data !== "object") {
    return {
      settings: { aiProvider: null },
      profile: null,
      tasks: [],
      rankedTasks: [],
      schedule: [],
      fixedBlocks: [],
      goals: [],
      reflections: [],
      blockingRules: [],
      dailyHabits: [],
      focusSessions: [],
      assistantHistory: [],
      weeklyInsights: null,
      achievements: {},
      taskTemplates: [],
      calendarExportSettings: null,
      firstReflectionDueDate: null,
    };
  }

  const rawSettings = data.settings && typeof data.settings === "object" ? data.settings : {};
  const aiProviderRaw = typeof rawSettings.aiProvider === "string" ? rawSettings.aiProvider.trim().toLowerCase() : "";
  const aiProvider = SUPPORTED_LLM_PROVIDERS.has(aiProviderRaw) ? aiProviderRaw : null;

  return {
    settings: { aiProvider },
    profile: data.profile ?? null,
    tasks: Array.isArray(data.tasks) ? data.tasks : [],
    rankedTasks: Array.isArray(data.rankedTasks) ? data.rankedTasks : [],
    schedule: Array.isArray(data.schedule) ? data.schedule : [],
    fixedBlocks: Array.isArray(data.fixedBlocks) ? data.fixedBlocks : [],
    goals: Array.isArray(data.goals) ? data.goals : [],
    reflections: Array.isArray(data.reflections) ? data.reflections : [],
    blockingRules: Array.isArray(data.blockingRules) ? data.blockingRules : [],
    dailyHabits: Array.isArray(data.dailyHabits) ? data.dailyHabits : [],
    focusSessions: Array.isArray(data.focusSessions) ? data.focusSessions : [],
    assistantHistory: Array.isArray(data.assistantHistory) ? data.assistantHistory : [],
    weeklyInsights: data.weeklyInsights ?? null,
    achievements: data.achievements && typeof data.achievements === "object" ? data.achievements : {},
    taskTemplates: Array.isArray(data.taskTemplates) ? data.taskTemplates : [],
    calendarExportSettings: data.calendarExportSettings ?? null,
    firstReflectionDueDate: data.firstReflectionDueDate ?? null,
  };
}

function normalizeTaskPriority(value) {
  if (!value) return "";
  const v = String(value).trim();
  const allowed = new Set([
    "Urgent & Important",
    "Urgent, Not Important",
    "Important, Not Urgent",
    "Not Urgent & Not Important",
  ]);
  if (allowed.has(v)) return v;

  const legacy = {
    "urgent-important": "Urgent & Important",
    "urgent-not-important": "Urgent, Not Important",
    "important-not-urgent": "Important, Not Urgent",
    "not-urgent-not-important": "Not Urgent & Not Important",
  };
  const key = v.toLowerCase().replace(/[^a-z]+/g, "-");
  return legacy[key] || "";
}

function normalizeTaskCategory(value) {
  if (!value) return "study";
  return String(value).trim().toLowerCase();
}

function clampTaskDurationHours(value) {
  if (value === null || value === undefined) return 0;
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) return 0;
  return Math.min(24, n);
}

function createAxisTask(input) {
  const taskName = String(input?.task_name || "").trim();
  const id = `task_${Date.now()}_${crypto.randomBytes(6).toString("hex")}`;
  return {
    id,
    task_name: taskName,
    task_priority: normalizeTaskPriority(input?.task_priority) || "Important, Not Urgent",
    task_category: normalizeTaskCategory(input?.task_category),
    task_deadline: String(input?.task_deadline || "").trim(),
    task_deadline_time: String(input?.task_deadline_time || "23:59").trim(),
    task_duration_hours: clampTaskDurationHours(input?.task_duration_hours),
    computer_required: Boolean(input?.computer_required),
    completed: false,
    createdAt: new Date().toISOString(),
  };
}

function localDateKey(date) {
  const d = date instanceof Date ? date : new Date(date);
  if (Number.isNaN(d.getTime())) return "";
  const local = new Date(d.getTime() - d.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 10);
}

const GOAL_COLOR_PALETTE = [
  { bg: "rgba(139, 92, 246, 0.15)", border: "rgba(139, 92, 246, 0.3)", text: "#7c3aed" },
  { bg: "rgba(14, 165, 233, 0.15)", border: "rgba(14, 165, 233, 0.3)", text: "#0284c7" },
  { bg: "rgba(236, 72, 153, 0.15)", border: "rgba(236, 72, 153, 0.3)", text: "#db2777" },
  { bg: "rgba(34, 197, 94, 0.15)", border: "rgba(34, 197, 94, 0.3)", text: "#16a34a" },
  { bg: "rgba(251, 146, 60, 0.15)", border: "rgba(251, 146, 60, 0.3)", text: "#ea580c" },
  { bg: "rgba(168, 85, 247, 0.15)", border: "rgba(168, 85, 247, 0.3)", text: "#7c3aed" },
];

function normalizeGoalLevel(value) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw) return "lifetime";
  const normalized = raw.replace(/[^a-z]+/g, "-");
  const mapping = {
    lifetime: "lifetime",
    life: "lifetime",
    "long-term": "lifetime",
    longterm: "lifetime",
    yearly: "yearly",
    annual: "yearly",
    year: "yearly",
    seasonal: "seasonal",
    quarterly: "seasonal",
    quarter: "seasonal",
    monthly: "monthly",
    month: "monthly",
    weekly: "weekly",
    week: "weekly",
    daily: "daily",
    day: "daily",
  };
  return mapping[normalized] || "lifetime";
}

function clampGoalProgress(value) {
  if (value === undefined || value === null) return 0;
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, Math.round(n)));
}

function normalizeGoalMilestones(value) {
  const fallback = [25, 50, 75];
  if (!value) return fallback;
  const arr = Array.isArray(value)
    ? value
    : String(value)
        .split(",")
        .map((v) => v.trim())
        .filter(Boolean);
  const cleaned = arr
    .map((v) => Number(v))
    .filter((n) => Number.isFinite(n))
    .map((n) => Math.max(0, Math.min(100, Math.round(n))));
  const unique = Array.from(new Set(cleaned)).sort((a, b) => a - b);
  return unique.length ? unique : fallback;
}

function goalSlug(goal) {
  const name = typeof goal === "string" ? goal : goal?.name;
  return String(name || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-");
}

function defaultGoalDates(level) {
  if (level === "lifetime") {
    return { startDate: "", endDate: "" };
  }

  const now = new Date();
  const start = new Date(now);
  start.setHours(0, 0, 0, 0);
  const end = new Date(start);

  if (level === "yearly") {
    start.setMonth(0, 1);
    end.setMonth(11, 31);
  } else if (level === "seasonal") {
    const quarterStart = Math.floor(start.getMonth() / 3) * 3;
    start.setMonth(quarterStart, 1);
    end.setMonth(quarterStart + 3, 0);
  } else if (level === "monthly") {
    start.setDate(1);
    end.setMonth(end.getMonth() + 1, 0);
  } else if (level === "weekly") {
    const dow = (start.getDay() + 6) % 7;
    start.setDate(start.getDate() - dow);
    end.setDate(start.getDate() + 6);
  } else if (level === "daily") {
    // keep today
  }

  return { startDate: localDateKey(start), endDate: localDateKey(end) };
}

function createAxisGoal(input, index = 0) {
  const name = String(input?.name || "").trim();
  const level = normalizeGoalLevel(input?.level);
  const id = `goal_${Date.now()}_${crypto.randomBytes(6).toString("hex")}`;
  const color = GOAL_COLOR_PALETTE[index % GOAL_COLOR_PALETTE.length];

  let startDate = String(input?.startDate || "").trim();
  let endDate = String(input?.endDate || "").trim();
  if (!startDate && !endDate) {
    const defaults = defaultGoalDates(level);
    startDate = defaults.startDate;
    endDate = defaults.endDate;
  }

  return {
    id,
    name,
    level,
    parentId: input?.parentId ? String(input.parentId).trim() : null,
    color,
    createdAt: new Date().toISOString(),
    manualProgress: clampGoalProgress(input?.manualProgress),
    milestones: normalizeGoalMilestones(input?.milestones),
    startDate,
    endDate,
    completed: false,
    completedAt: "",
  };
}

function ensureDailyGoalTask(data, goal) {
  if (!goal || goal.level !== "daily") return { created: false, updated: false };
  if (!Array.isArray(data.tasks)) data.tasks = [];
  const slug = goalSlug(goal);
  const existing = data.tasks.find((t) => t?.fromDailyGoal && t?.goalId === goal.id);
  if (!existing) {
    data.tasks.push({
      id: `task_goal_${goal.id}`,
      task_name: goal.name,
      task_priority: "Important, Not Urgent",
      task_category: slug || "study",
      task_deadline: localDateKey(new Date()),
      task_deadline_time: "23:59",
      task_duration_hours: 1,
      computer_required: false,
      completed: false,
      fromDailyGoal: true,
      goalId: goal.id,
      createdAt: new Date().toISOString(),
    });
    return { created: true, updated: false };
  }

  let updated = false;
  if (goal.name && existing.task_name !== goal.name) {
    existing.task_name = goal.name;
    updated = true;
  }
  if (slug && existing.task_category !== slug) {
    existing.task_category = slug;
    updated = true;
  }

  return { created: false, updated };
}

function removeDailyGoalTasks(data, goalId) {
  if (!Array.isArray(data.tasks)) return false;
  const removedIds = new Set();
  data.tasks = data.tasks.filter((t) => {
    if (t?.fromDailyGoal && t?.goalId === goalId) {
      if (t?.id) removedIds.add(String(t.id));
      return false;
    }
    return true;
  });
  if (removedIds.size && Array.isArray(data.schedule)) {
    data.schedule = data.schedule.filter((b) => !removedIds.has(String(b?.taskId || "")));
  }
  return removedIds.size > 0;
}

const ASSISTANT_HISTORY_LIMIT = 20;
const ASSISTANT_HISTORY_CONTEXT = 12;

function ensureAssistantHistory(data) {
  if (!Array.isArray(data.assistantHistory)) {
    data.assistantHistory = [];
  }
  return data.assistantHistory;
}

function appendAssistantHistory(data, entries) {
  const history = ensureAssistantHistory(data);
  const list = Array.isArray(entries) ? entries : [entries];
  let changed = false;
  list.forEach((entry) => {
    if (!entry || typeof entry !== "object") return;
    const role = entry.role === "assistant" ? "assistant" : "user";
    const content = String(entry.content || "").trim();
    if (!content) return;
    history.push({
      role,
      content: content.slice(0, 1200),
      ts: new Date().toISOString(),
    });
    changed = true;
  });
  if (history.length > ASSISTANT_HISTORY_LIMIT) {
    history.splice(0, history.length - ASSISTANT_HISTORY_LIMIT);
  }
  return changed;
}

function formatAssistantHistory(data, limit = ASSISTANT_HISTORY_CONTEXT) {
  const history = Array.isArray(data.assistantHistory) ? data.assistantHistory : [];
  if (!history.length) return "";
  return history
    .slice(-limit)
    .map((entry) => `${entry.role === "assistant" ? "Assistant" : "User"}: ${entry.content}`)
    .join("\n");
}

function buildAssistantSnapshot(data) {
  const safeData = ensureAxisUserDataShape(data);
  const tasks = safeData.tasks.slice(0, 120).map((t) => ({
    id: String(t?.id || ""),
    name: String(t?.task_name || t?.name || "").slice(0, 180),
    priority: String(t?.task_priority || t?.priority || ""),
    category: String(t?.task_category || t?.category || ""),
    deadline: String(t?.task_deadline || t?.deadline || ""),
    deadlineTime: String(t?.task_deadline_time || t?.deadlineTime || ""),
    durationHours: Number(t?.task_duration_hours ?? t?.durationHours ?? t?.estimatedHours ?? 0) || 0,
    completed: Boolean(t?.completed),
  }));

  const goals = safeData.goals.slice(0, 120).map((g) => ({
    id: String(g?.id || ""),
    name: String(g?.name || "").slice(0, 180),
    level: String(g?.level || ""),
    parentId: String(g?.parentId || ""),
    startDate: String(g?.startDate || ""),
    endDate: String(g?.endDate || ""),
    manualProgress: Number(g?.manualProgress ?? g?.progress ?? 0) || 0,
    completed: Boolean(g?.completed),
  }));

  const schedule = safeData.schedule.slice(0, 200).map((b) => ({
    kind: String(b?.kind || ""),
    taskId: String(b?.taskId || ""),
    start: String(b?.start || ""),
    end: String(b?.end || ""),
    reason: typeof b?.reason === "string" ? b.reason.slice(0, 140) : "",
  }));

  const fixedBlocks = safeData.fixedBlocks.slice(0, 200).map((b) => ({
    label: String(b?.label || b?.kind || "Fixed").slice(0, 80),
    start: String(b?.start || ""),
    end: String(b?.end || ""),
    category: String(b?.category || ""),
  }));

  const dailyHabits = safeData.dailyHabits.slice(0, 80).map((h) => ({
    id: String(h?.id || ""),
    name: String(h?.name || "").slice(0, 120),
    time: String(h?.time || "").slice(0, 40),
    description: String(h?.description || "").slice(0, 180),
  }));

  const profile = safeData.profile && typeof safeData.profile === "object" ? safeData.profile : null;
  const profileBrief = profile
    ? {
        user_name: typeof profile.user_name === "string" ? profile.user_name.slice(0, 100) : "",
        user_age_group: typeof profile.user_age_group === "string" ? profile.user_age_group : "",
        most_productive_time: profile.most_productive_time || "",
        preferred_work_style: profile.preferred_work_style || "",
        preferred_study_method: profile.preferred_study_method || "",
        procrastinator_type: profile.procrastinator_type || "",
        has_trouble_finishing: profile.has_trouble_finishing || "",
      }
    : null;

  return {
    profile: profileBrief,
    tasks,
    goals,
    schedule,
    fixedBlocks,
    dailyHabits,
  };
}

// Middleware to verify JWT token
function authenticateToken(req, res, next) {
  if (!requireJwtSecret(res)) return;
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];

  if (!token) {
    return res.status(401).json({ error: "Authentication required" });
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ error: "Invalid or expired token" });
    }
    req.user = user;
    next();
  });
}

async function getOrCreateCalendarToken(userId) {
  const users = await getUsers();
  let foundEmail = null;
  let record = null;

  for (const [email, user] of Object.entries(users)) {
    if (user?.id === userId) {
      foundEmail = email;
      record = user;
      break;
    }
  }

  if (!record || !foundEmail) return null;
  if (record.calendarToken && typeof record.calendarToken === "string" && record.calendarToken.length >= 24) {
    return record.calendarToken;
  }

  const token = crypto.randomBytes(24).toString("hex");
  record.calendarToken = token;
  record.updatedAt = new Date().toISOString();
  users[foundEmail] = record;
  await saveUsers(users);
  return token;
}

async function getUserIdByCalendarToken(token) {
  if (!token || typeof token !== "string" || token.length < 24) return null;
  const users = await getUsers();
  for (const user of Object.values(users)) {
    if (user?.calendarToken === token) return user.id;
  }
  return null;
}

// Authentication endpoints
app.post("/api/auth/register", validateBody(registerSchema), async (req, res) => {
  try {
    if (!requireJwtSecret(res)) return;

    const { email, password, name, username } = req.body;
    const emailKey = String(email || "").trim().toLowerCase();
    const nameClean = String(name || "").trim();
    let usernameClean = typeof username === "string" ? username.trim() : "";

    const users = await getUsers();
    
    // Check if email already exists
    const existingEmailKey = Object.keys(users).find((k) => k.toLowerCase() === emailKey);
    if (existingEmailKey) {
      return res.status(409).json({ error: "Email already registered" });
    }

    const isUsernameTaken = (candidate) =>
      Object.values(users).some((u) => u.username && u.username.toLowerCase() === candidate.toLowerCase());

    if (usernameClean && isUsernameTaken(usernameClean)) {
      return res.status(409).json({ error: "Username already taken" });
    }

    if (!usernameClean) {
      const localPart = emailKey.split("@")[0] || "user";
      const base = localPart
        .toLowerCase()
        .replace(/[^a-z0-9_]/g, "_")
        .replace(/_+/g, "_")
        .replace(/^_+|_+$/g, "")
        .slice(0, 24);
      usernameClean = base.length >= 3 ? base : `user_${crypto.randomBytes(3).toString("hex")}`;
    }

    // Ensure uniqueness (handles auto-generated and user-provided edge cases).
    if (isUsernameTaken(usernameClean)) {
      const preferred = usernameClean;
      for (let attempt = 0; attempt < 25; attempt++) {
        const suffix = crypto.randomBytes(2).toString("hex"); // 4 chars
        const base = preferred.slice(0, Math.max(3, 30 - (suffix.length + 1)));
        const candidate = `${base}_${suffix}`;
        if (!isUsernameTaken(candidate)) {
          usernameClean = candidate;
          break;
        }
      }
    }

    if (isUsernameTaken(usernameClean)) {
      return res.status(409).json({ error: "Unable to allocate a unique username. Please try again." });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    users[emailKey] = {
      id: userId,
      email: emailKey,
      username: usernameClean,
      password: hashedPassword,
      name: nameClean,
      createdAt: new Date().toISOString(),
    };

    await saveUsers(users);

    // Initialize user data
    await saveUserData(userId, {
      profile: null,
      tasks: [],
      rankedTasks: [],
      schedule: [],
      fixedBlocks: [],
      goals: [],
      reflections: [],
      blockingRules: [],
    });

    const token = jwt.sign({ userId, email: emailKey, username: usernameClean }, JWT_SECRET, { expiresIn: "30d" });
    res.json({ token, user: { id: userId, email: emailKey, username: usernameClean, name: nameClean } });
  } catch (err) {
    console.error("Registration error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/api/auth/login", validateBody(loginSchema), async (req, res) => {
  try {
    if (!requireJwtSecret(res)) return;

    const { identifier, password } = req.body;
    const identifierRaw = String(identifier || "").trim();
    const identifierLower = identifierRaw.toLowerCase();

    const users = await getUsers();
    
    // Find user by email, username, or display name
    let user = null;
    let userEmail = null;
    
    // First try direct email lookup
    if (users[identifierRaw]) {
      user = users[identifierRaw];
      userEmail = identifierRaw;
    } else if (users[identifierLower]) {
      user = users[identifierLower];
      userEmail = identifierLower;
    } else {
      // Search by email key (case-insensitive)
      for (const [email, u] of Object.entries(users)) {
        if (email.toLowerCase() === identifierLower) {
          user = u;
          userEmail = email;
          break;
        }
      }
    }

    // Search by username (case-insensitive)
    if (!user) {
      for (const [email, u] of Object.entries(users)) {
        if (u.username && u.username.toLowerCase() === identifierLower) {
          user = u;
          userEmail = email;
          break;
        }
      }
    }

    // Search by name (case-insensitive). Only allow if it matches exactly one account.
    if (!user) {
      const matches = [];
      for (const [email, u] of Object.entries(users)) {
        if (u.name && u.name.toLowerCase() === identifierLower) {
          matches.push([email, u]);
        }
      }
      if (matches.length === 1) {
        userEmail = matches[0][0];
        user = matches[0][1];
      } else if (matches.length > 1) {
        return res.status(400).json({ error: "Multiple accounts share that name. Please log in with email or username." });
      }
    }
    
    if (!user) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    if (!user.password) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const token = jwt.sign({ userId: user.id, email: userEmail, username: user.username }, JWT_SECRET, { expiresIn: "30d" });
    res.json({ token, user: { id: user.id, email: userEmail, username: user.username, name: user.name } });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/api/auth/google", async (req, res) => {
  try {
    const { email, name, googleId } = req.body;
    if (!email || !name || !googleId) {
      return res.status(400).json({ error: "Email, name, and googleId are required" });
    }

    const users = await getUsers();
    let user = users[email];

    if (!user) {
      // Create new user
      const userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      user = {
        id: userId,
        email,
        name,
        googleId,
        createdAt: new Date().toISOString(),
      };
      users[email] = user;
      await saveUsers(users);

      // Initialize user data
      await saveUserData(userId, {
        profile: null,
        tasks: [],
        rankedTasks: [],
        schedule: [],
        fixedBlocks: [],
        goals: [],
        reflections: [],
        blockingRules: [],
      });
    } else if (!user.googleId) {
      // Link Google account to existing user
      user.googleId = googleId;
      users[email] = user;
      await saveUsers(users);
    }

    const token = jwt.sign({ userId: user.id, email }, JWT_SECRET, { expiresIn: "30d" });
    res.json({ token, user: { id: user.id, email, name: user.name } });
  } catch (err) {
    console.error("Google auth error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// User data endpoints
app.get("/api/user/data", authenticateToken, async (req, res) => {
  try {
    const data = await getUserData(req.user.userId);
    if (!data) {
      return res.status(404).json({ error: "User data not found" });
    }
    res.json(data);
  } catch (err) {
    console.error("Get user data error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/api/user/data", authenticateToken, async (req, res) => {
  try {
    const existing = await getUserData(req.user.userId);
    const incoming = req.body && typeof req.body === "object" ? req.body : {};
    if (incoming.assistantHistory === undefined && Array.isArray(existing?.assistantHistory)) {
      incoming.assistantHistory = existing.assistantHistory;
    }
    await saveUserData(req.user.userId, incoming);
    res.json({ success: true });
  } catch (err) {
    console.error("Save user data error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ---------- Task + habit APIs ----------

app.get("/api/tasks", authenticateToken, async (req, res) => {
  try {
    const data = await getUserData(req.user.userId);
    if (!data) return res.status(404).json({ error: "User data not found" });
    const safeData = ensureAxisUserDataShape(data);
    res.json({ tasks: safeData.tasks });
  } catch (err) {
    console.error("tasks:list error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/api/tasks", authenticateToken, validateBody(taskCreateSchema), async (req, res) => {
  try {
    const data = await getUserData(req.user.userId);
    if (!data) return res.status(404).json({ error: "User data not found" });
    const safeData = ensureAxisUserDataShape(data);

    const created = createAxisTask(req.body);
    safeData.tasks.push(created);
    await saveUserData(req.user.userId, safeData);

    res.json({ task: created });
  } catch (err) {
    console.error("tasks:create error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.patch("/api/tasks/:taskId", authenticateToken, validateBody(taskUpdateSchema), async (req, res) => {
  try {
    const taskId = String(req.params.taskId || "").trim();
    if (!taskId) return res.status(400).json({ error: "Missing taskId" });

    const data = await getUserData(req.user.userId);
    if (!data) return res.status(404).json({ error: "User data not found" });
    const safeData = ensureAxisUserDataShape(data);

    const idx = safeData.tasks.findIndex((t) => t && String(t.id) === taskId);
    if (idx < 0) return res.status(404).json({ error: "Task not found" });

    const task = safeData.tasks[idx] && typeof safeData.tasks[idx] === "object" ? safeData.tasks[idx] : {};
    const patch = req.body || {};

    if (patch.task_name !== undefined) task.task_name = String(patch.task_name || "").trim();
    if (patch.task_priority !== undefined) {
      const normalized = normalizeTaskPriority(patch.task_priority);
      if (normalized) task.task_priority = normalized;
    }
    if (patch.task_category !== undefined) task.task_category = normalizeTaskCategory(patch.task_category);
    if (patch.task_deadline !== undefined) task.task_deadline = String(patch.task_deadline || "").trim();
    if (patch.task_deadline_time !== undefined) task.task_deadline_time = String(patch.task_deadline_time || "23:59").trim();
    if (patch.task_duration_hours !== undefined) task.task_duration_hours = clampTaskDurationHours(patch.task_duration_hours);
    if (patch.computer_required !== undefined) task.computer_required = Boolean(patch.computer_required);
    if (patch.completed !== undefined) {
      task.completed = Boolean(patch.completed);
      task.completedAt = task.completed ? new Date().toISOString() : undefined;
    }

    task.updatedAt = new Date().toISOString();
    safeData.tasks[idx] = task;
    await saveUserData(req.user.userId, safeData);

    res.json({ task });
  } catch (err) {
    console.error("tasks:update error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.delete("/api/tasks/:taskId", authenticateToken, async (req, res) => {
  try {
    const taskId = String(req.params.taskId || "").trim();
    if (!taskId) return res.status(400).json({ error: "Missing taskId" });

    const data = await getUserData(req.user.userId);
    if (!data) return res.status(404).json({ error: "User data not found" });
    const safeData = ensureAxisUserDataShape(data);

    const before = safeData.tasks.length;
    safeData.tasks = safeData.tasks.filter((t) => t && String(t.id) !== taskId);
    safeData.schedule = safeData.schedule.filter((b) => !(b && b.kind === "task" && String(b.taskId) === taskId));
    if (safeData.tasks.length === before) return res.status(404).json({ error: "Task not found" });

    await saveUserData(req.user.userId, safeData);
    res.json({ success: true });
  } catch (err) {
    console.error("tasks:delete error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.get("/api/habits", authenticateToken, async (req, res) => {
  try {
    const data = await getUserData(req.user.userId);
    if (!data) return res.status(404).json({ error: "User data not found" });
    const safeData = ensureAxisUserDataShape(data);
    res.json({ dailyHabits: safeData.dailyHabits });
  } catch (err) {
    console.error("habits:list error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/api/habits", authenticateToken, validateBody(habitCreateSchema), async (req, res) => {
  try {
    const data = await getUserData(req.user.userId);
    if (!data) return res.status(404).json({ error: "User data not found" });
    const safeData = ensureAxisUserDataShape(data);

    const habit = {
      id: `habit_${Date.now()}_${crypto.randomBytes(6).toString("hex")}`,
      name: String(req.body.name || "").trim(),
      time: String(req.body.time || "").trim(),
      description: String(req.body.description || "").trim(),
      createdAt: new Date().toISOString(),
    };
    safeData.dailyHabits.push(habit);
    await saveUserData(req.user.userId, safeData);
    res.json({ habit });
  } catch (err) {
    console.error("habits:create error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.delete("/api/habits/:habitId", authenticateToken, async (req, res) => {
  try {
    const habitId = String(req.params.habitId || "").trim();
    if (!habitId) return res.status(400).json({ error: "Missing habitId" });

    const data = await getUserData(req.user.userId);
    if (!data) return res.status(404).json({ error: "User data not found" });
    const safeData = ensureAxisUserDataShape(data);

    const before = safeData.dailyHabits.length;
    safeData.dailyHabits = safeData.dailyHabits.filter((h) => h && String(h.id) !== habitId);
    if (safeData.dailyHabits.length === before) return res.status(404).json({ error: "Habit not found" });

    await saveUserData(req.user.userId, safeData);
    res.json({ success: true });
  } catch (err) {
    console.error("habits:delete error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ---------- Calendar Export (.ics) ----------

app.get("/api/calendar/token", authenticateToken, async (req, res) => {
  try {
    const token = await getOrCreateCalendarToken(req.user.userId);
    if (!token) {
      return res.status(404).json({ error: "User not found" });
    }
    const baseUrl = `${req.protocol}://${req.get("host")}`;
    const subscribeUrl = `${baseUrl}/api/calendar/subscribe/${token}.ics`;
    const webcalUrl = subscribeUrl.replace(/^https?:\/\//, "webcal://");
    res.json({ token, subscribeUrl, webcalUrl });
  } catch (err) {
    console.error("calendar token error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.post(
  "/api/calendar/export",
  authenticateToken,
  validateBody(calendarExportSchema),
  async (req, res) => {
    try {
      const data = await getUserData(req.user.userId);
      if (!data) {
        return res.status(404).json({ error: "User data not found" });
      }

      const options = {
        ...(data.calendarExportSettings && typeof data.calendarExportSettings === "object"
          ? data.calendarExportSettings
          : {}),
        ...req.body,
      };

      const ics = buildAxisIcs({ userId: req.user.userId, data, options });
      res.setHeader("Content-Type", "text/calendar; charset=utf-8");
      res.setHeader("Content-Disposition", 'attachment; filename="axis-schedule.ics"');
      res.send(ics);
    } catch (err) {
      console.error("calendar export error:", err);
      res.status(500).json({ error: "Calendar export failed" });
    }
  },
);

app.get("/api/calendar/subscribe/:token.ics", async (req, res) => {
  try {
    const token = req.params.token;
    const userId = await getUserIdByCalendarToken(token);
    if (!userId) {
      return res.status(404).send("Not found");
    }

    const data = await getUserData(userId);
    if (!data) {
      return res.status(404).send("Not found");
    }

    const options =
      data.calendarExportSettings && typeof data.calendarExportSettings === "object"
        ? data.calendarExportSettings
        : {};

    const ics = buildAxisIcs({ userId, data, options });
    res.setHeader("Content-Type", "text/calendar; charset=utf-8");
    res.setHeader("Cache-Control", "no-store");
    res.send(ics);
  } catch (err) {
    console.error("calendar subscribe error:", err);
    res.status(500).send("Calendar subscribe failed");
  }
});

// ---------- Assistant (agentic) ----------

const assistantToolCallSchema = z.object({
  name: z.string().min(1).max(80),
  arguments: z.any().optional(),
});

const assistantPlannerResponseSchema = z.object({
  assistant_reply: z.string().min(1).max(6000),
  tool_calls: z.array(assistantToolCallSchema).optional().default([]),
});

const assistantFinalResponseSchema = z.object({
  reply: z.string().min(1).max(6000),
});

const assistantUpdateTaskSchema = taskUpdateSchema.extend({
  id: z.string().min(1).max(200),
});

const assistantDeleteTaskSchema = z.object({
  id: z.string().min(1).max(200),
});

const assistantDeleteHabitSchema = z.object({
  id: z.string().min(1).max(200),
});

const goalCreateSchema = z.object({
  name: z.string().min(1).max(200),
  level: z.string().optional(),
  parentId: z.string().optional().nullable(),
  startDate: z.string().optional(),
  endDate: z.string().optional(),
  manualProgress: z.number().min(0).max(100).optional(),
  milestones: z.array(z.number()).optional(),
});

const goalUpdateSchema = z.object({
  id: z.string().min(1).max(200),
  name: z.string().min(1).max(200).optional(),
  level: z.string().optional(),
  parentId: z.string().optional().nullable(),
  startDate: z.string().optional(),
  endDate: z.string().optional(),
  manualProgress: z.number().min(0).max(100).optional(),
  milestones: z.array(z.number()).optional(),
  completed: z.boolean().optional(),
});

const assistantDeleteGoalSchema = z.object({
  id: z.string().min(1).max(200),
});

const assistantRebalanceSchema = z.object({
  horizonDays: z.number().int().min(1).max(21).optional().default(7),
  maxHoursPerDay: z.number().min(1).max(16).optional().default(10),
});

const ASSISTANT_TOOLS = [
  {
    name: "get_snapshot",
    description: "Get a compact snapshot of the user's profile, tasks, goals, schedule, fixed blocks, and daily habits.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "list_tasks",
    description: "List tasks. Optionally include completed tasks.",
    inputSchema: {
      type: "object",
      properties: { includeCompleted: { type: "boolean", default: false } },
    },
  },
  {
    name: "create_task",
    description:
      "Create a new task in the user's task list (canonical fields: task_name, task_priority, task_category, task_deadline, task_deadline_time, task_duration_hours).",
    inputSchema: {
      type: "object",
      required: ["task_name"],
      properties: {
        task_name: { type: "string" },
        task_priority: { type: "string" },
        task_category: { type: "string" },
        task_deadline: { type: "string" },
        task_deadline_time: { type: "string" },
        task_duration_hours: { type: "number" },
        computer_required: { type: "boolean" },
      },
    },
  },
  {
    name: "update_task",
    description: "Update an existing task by id (supports completing, renaming, changing deadline, etc.).",
    inputSchema: {
      type: "object",
      required: ["id"],
      properties: {
        id: { type: "string" },
        task_name: { type: "string" },
        task_priority: { type: "string" },
        task_category: { type: "string" },
        task_deadline: { type: "string" },
        task_deadline_time: { type: "string" },
        task_duration_hours: { type: "number" },
        computer_required: { type: "boolean" },
        completed: { type: "boolean" },
      },
    },
  },
  {
    name: "delete_task",
    description: "Delete an existing task by id (only if the user explicitly asked to delete).",
    inputSchema: {
      type: "object",
      required: ["id"],
      properties: { id: { type: "string" } },
    },
  },
  {
    name: "create_goal",
    description: "Create a new goal (name + timeframe level, optional parentId and dates).",
    inputSchema: {
      type: "object",
      required: ["name"],
      properties: {
        name: { type: "string" },
        level: { type: "string" },
        parentId: { type: "string" },
        startDate: { type: "string" },
        endDate: { type: "string" },
        manualProgress: { type: "number" },
        milestones: { type: "array", items: { type: "number" } },
      },
    },
  },
  {
    name: "update_goal",
    description: "Update an existing goal by id (rename, change level, dates, progress, or parent).",
    inputSchema: {
      type: "object",
      required: ["id"],
      properties: {
        id: { type: "string" },
        name: { type: "string" },
        level: { type: "string" },
        parentId: { type: "string" },
        startDate: { type: "string" },
        endDate: { type: "string" },
        manualProgress: { type: "number" },
        milestones: { type: "array", items: { type: "number" } },
        completed: { type: "boolean" },
      },
    },
  },
  {
    name: "delete_goal",
    description: "Delete a goal by id (only if the user explicitly asked to delete).",
    inputSchema: {
      type: "object",
      required: ["id"],
      properties: { id: { type: "string" } },
    },
  },
  {
    name: "add_habit",
    description: "Add a daily habit (name + time).",
    inputSchema: {
      type: "object",
      required: ["name", "time"],
      properties: {
        name: { type: "string" },
        time: { type: "string" },
        description: { type: "string" },
      },
    },
  },
  {
    name: "delete_habit",
    description: "Delete a daily habit by id.",
    inputSchema: {
      type: "object",
      required: ["id"],
      properties: { id: { type: "string" } },
    },
  },
  {
    name: "rebalance_schedule",
    description:
      "Use the LLM to rebalance the user's schedule into time blocks for the next N days, respecting fixed blocks.",
    inputSchema: {
      type: "object",
      properties: {
        horizonDays: { type: "integer", default: 7, minimum: 1, maximum: 21 },
        maxHoursPerDay: { type: "number", default: 10, minimum: 1, maximum: 16 },
      },
    },
  },
  {
    name: "get_calendar_links",
    description: "Get calendar subscription links (subscribeUrl + webcalUrl).",
    inputSchema: { type: "object", properties: {} },
  },
];

async function aiRebalanceScheduleFromUserData({ data, horizonDays, maxHoursPerDay }) {
  const safeData = ensureAxisUserDataShape(data);
  const provider = resolveProviderForUserData(safeData);
  const tasks = Array.isArray(safeData.tasks) ? safeData.tasks : [];
  const fixedBlocks = Array.isArray(safeData.fixedBlocks) ? safeData.fixedBlocks : [];
  const schedule = Array.isArray(safeData.schedule) ? safeData.schedule : [];
  const profile = safeData.profile && typeof safeData.profile === "object" ? safeData.profile : {};

  const tasksBrief = tasks
    .filter((t) => t && typeof t === "object" && typeof t.id === "string")
    .filter((t) => !t.completed)
    .slice(0, 350)
    .map((t) => ({
      id: t.id,
      name: String(t.task_name || "").slice(0, 140),
      priority: String(t.task_priority || ""),
      category: String(t.task_category || ""),
      deadline: `${t.task_deadline || ""}T${t.task_deadline_time || "23:59"}`,
      durationHours: Number(t.task_duration_hours || 0) || 0,
    }));

  const fixedBrief = fixedBlocks
    .filter((b) => b && typeof b === "object" && b.start && b.end)
    .slice(0, 500)
    .map((b) => ({
      start: b.start,
      end: b.end,
      label: String(b.label || b.kind || "Fixed").slice(0, 80),
      category: String(b.category || ""),
    }));

  const scheduleBrief = schedule
    .filter((b) => b && typeof b === "object" && b.start && b.end)
    .slice(0, 500)
    .map((b) => ({
      taskId: b.taskId || "",
      start: b.start,
      end: b.end,
    }));

  const profileBrief = (() => {
    if (!profile || typeof profile !== "object") return {};
    const keep = [
      "user_name",
      "user_age_group",
      "procrastinator_type",
      "preferred_work_style",
      "preferred_study_method",
      "most_productive_time",
      "is_procrastinator",
      "has_trouble_finishing",
      "productive_windows",
      "weekly_personal_time",
      "weekly_review_hours",
    ];
    const out = {};
    keep.forEach((k) => {
      if (profile[k] !== undefined) out[k] = profile[k];
    });
    return out;
  })();

  const today = new Date();
  const todayIso = today.toISOString().slice(0, 10);

  const userPrompt = `
Rebalance the user's schedule for the next ${horizonDays} days starting ${todayIso}.
Return JSON only: {"blocks":[{"taskId":"task-id","start":"ISO-8601 UTC","end":"ISO-8601 UTC","reason":"short"}]}.

Hard rules:
- "start" and "end" MUST be ISO-8601 timestamps in UTC with a trailing "Z".
- Do not create blocks that overlap fixedBlocks.
- Do not overlap your own blocks.
- Only use taskIds from the provided tasks list.
- Keep total scheduled work per day <= ${maxHoursPerDay} hours.
- Each block must be at least 15 minutes and end after start.

Soft rules:
- Prefer scheduling higher priority and earlier deadlines first.
- Split long tasks into multiple blocks, adding small buffers when reasonable.
- Use the user's focus preferences when provided in profile.

Tasks: ${JSON.stringify(tasksBrief).slice(0, 7000)}
Fixed blocks (unavailable): ${JSON.stringify(fixedBrief).slice(0, 7000)}
Current schedule (may be ignored): ${JSON.stringify(scheduleBrief).slice(0, 6000)}
Profile: ${JSON.stringify(profileBrief).slice(0, 2000)}
`.trim();

  const reply = await callLlm({
    system: "You are a time-blocking assistant. Return strict JSON only.",
    user: userPrompt,
    temperature: 0.25,
    maxTokens: 1200,
    expectJSON: true,
    provider,
  });

  const parsed = safeParseJSONFromText(reply) || {};
  const blocks = Array.isArray(parsed.blocks) ? parsed.blocks : [];
  if (!blocks.length) {
    throw new Error("AI returned no schedule blocks.");
  }

  const taskIdSet = new Set(tasksBrief.map((t) => t.id));
  const normalized = [];
  for (const b of blocks) {
    if (!b || typeof b !== "object") continue;
    if (!taskIdSet.has(b.taskId)) continue;
    const start = new Date(b.start);
    const end = new Date(b.end);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) continue;
    if (end <= start) continue;
    normalized.push({
      kind: "task",
      taskId: b.taskId,
      start: start.toISOString(),
      end: end.toISOString(),
    });
  }

  if (!normalized.length) {
    throw new Error("AI returned invalid schedule blocks.");
  }

  return normalized;
}

async function executeAssistantTool({ name, args, userId, data, req }) {
  const toolName = String(name || "").trim();
  const input = args && typeof args === "object" ? args : {};

  if (toolName === "get_snapshot") {
    return { snapshot: buildAssistantSnapshot(data) };
  }

  if (toolName === "list_tasks") {
    const includeCompleted = Boolean(input.includeCompleted);
    const safeData = ensureAxisUserDataShape(data);
    return {
      tasks: safeData.tasks
        .filter((t) => (includeCompleted ? true : !t?.completed))
        .slice(0, 500)
        .map((t) => ({
          id: t.id,
          task_name: t.task_name,
          task_priority: t.task_priority,
          task_category: t.task_category,
          task_deadline: t.task_deadline,
          task_deadline_time: t.task_deadline_time,
          task_duration_hours: t.task_duration_hours,
          completed: Boolean(t.completed),
        })),
    };
  }

  if (toolName === "create_task") {
    const parsed = taskCreateSchema.safeParse(input);
    if (!parsed.success) throw new Error("Invalid create_task arguments");
    const safeData = ensureAxisUserDataShape(data);
    const task = createAxisTask(parsed.data);
    safeData.tasks.push(task);
    data.tasks = safeData.tasks;
    return { task };
  }

  if (toolName === "update_task") {
    const parsed = assistantUpdateTaskSchema.safeParse(input);
    if (!parsed.success) throw new Error("Invalid update_task arguments");
    const safeData = ensureAxisUserDataShape(data);
    const idx = safeData.tasks.findIndex((t) => t && String(t.id) === parsed.data.id);
    if (idx < 0) throw new Error("Task not found");
    const task = safeData.tasks[idx] && typeof safeData.tasks[idx] === "object" ? safeData.tasks[idx] : {};

    const patch = parsed.data;
    if (patch.task_name !== undefined) task.task_name = String(patch.task_name || "").trim();
    if (patch.task_priority !== undefined) {
      const normalized = normalizeTaskPriority(patch.task_priority);
      if (normalized) task.task_priority = normalized;
    }
    if (patch.task_category !== undefined) task.task_category = normalizeTaskCategory(patch.task_category);
    if (patch.task_deadline !== undefined) task.task_deadline = String(patch.task_deadline || "").trim();
    if (patch.task_deadline_time !== undefined) task.task_deadline_time = String(patch.task_deadline_time || "23:59").trim();
    if (patch.task_duration_hours !== undefined) task.task_duration_hours = clampTaskDurationHours(patch.task_duration_hours);
    if (patch.computer_required !== undefined) task.computer_required = Boolean(patch.computer_required);
    if (patch.completed !== undefined) {
      task.completed = Boolean(patch.completed);
      task.completedAt = task.completed ? new Date().toISOString() : undefined;
    }
    task.updatedAt = new Date().toISOString();

    safeData.tasks[idx] = task;
    data.tasks = safeData.tasks;
    return { task };
  }

  if (toolName === "delete_task") {
    const parsed = assistantDeleteTaskSchema.safeParse(input);
    if (!parsed.success) throw new Error("Invalid delete_task arguments");
    const safeData = ensureAxisUserDataShape(data);
    const before = safeData.tasks.length;
    safeData.tasks = safeData.tasks.filter((t) => t && String(t.id) !== parsed.data.id);
    safeData.schedule = safeData.schedule.filter(
      (b) => !(b && b.kind === "task" && String(b.taskId) === parsed.data.id),
    );
    if (safeData.tasks.length === before) throw new Error("Task not found");
    data.tasks = safeData.tasks;
    data.schedule = safeData.schedule;
    return { success: true };
  }

  if (toolName === "create_goal") {
    const parsed = goalCreateSchema.safeParse(input);
    if (!parsed.success) throw new Error("Invalid create_goal arguments");
    const safeData = ensureAxisUserDataShape(data);
    const goal = createAxisGoal(parsed.data, safeData.goals.length);
    safeData.goals.push(goal);
    if (goal.level === "daily") {
      ensureDailyGoalTask(safeData, goal);
      data.tasks = safeData.tasks;
    }
    data.goals = safeData.goals;
    return { goal };
  }

  if (toolName === "update_goal") {
    const parsed = goalUpdateSchema.safeParse(input);
    if (!parsed.success) throw new Error("Invalid update_goal arguments");
    const safeData = ensureAxisUserDataShape(data);
    const idx = safeData.goals.findIndex((g) => g && String(g.id) === parsed.data.id);
    if (idx < 0) throw new Error("Goal not found");

    const existing = safeData.goals[idx] && typeof safeData.goals[idx] === "object" ? safeData.goals[idx] : {};
    const patch = parsed.data;
    const prevLevel = normalizeGoalLevel(existing.level);

    if (patch.name !== undefined) existing.name = String(patch.name || "").trim();
    if (patch.level !== undefined) existing.level = normalizeGoalLevel(patch.level);
    if (patch.parentId !== undefined) {
      existing.parentId = patch.parentId ? String(patch.parentId).trim() : null;
    }
    if (patch.startDate !== undefined) existing.startDate = String(patch.startDate || "").trim();
    if (patch.endDate !== undefined) existing.endDate = String(patch.endDate || "").trim();
    if (patch.manualProgress !== undefined) existing.manualProgress = clampGoalProgress(patch.manualProgress);
    if (patch.milestones !== undefined) existing.milestones = normalizeGoalMilestones(patch.milestones);
    if (patch.completed !== undefined) {
      existing.completed = Boolean(patch.completed);
      existing.completedAt = existing.completed ? new Date().toISOString() : "";
    }
    existing.updatedAt = new Date().toISOString();

    safeData.goals[idx] = existing;
    data.goals = safeData.goals;

    if (prevLevel === "daily" && existing.level !== "daily") {
      removeDailyGoalTasks(safeData, existing.id);
      data.tasks = safeData.tasks;
      data.schedule = safeData.schedule;
    } else if (existing.level === "daily") {
      ensureDailyGoalTask(safeData, existing);
      data.tasks = safeData.tasks;
    }

    return { goal: existing };
  }

  if (toolName === "delete_goal") {
    const parsed = assistantDeleteGoalSchema.safeParse(input);
    if (!parsed.success) throw new Error("Invalid delete_goal arguments");
    const safeData = ensureAxisUserDataShape(data);
    const idx = safeData.goals.findIndex((g) => g && String(g.id) === parsed.data.id);
    if (idx < 0) throw new Error("Goal not found");
    const goal = safeData.goals[idx];
    safeData.goals = safeData.goals.filter((g) => g && String(g.id) !== parsed.data.id);

    if (goal?.level === "daily") {
      removeDailyGoalTasks(safeData, parsed.data.id);
      data.tasks = safeData.tasks;
      data.schedule = safeData.schedule;
    }

    const slug = goalSlug(goal);
    if (slug && Array.isArray(safeData.tasks)) {
      safeData.tasks.forEach((t) => {
        if (!t || typeof t !== "object") return;
        if (t.goalId === parsed.data.id) t.goalId = null;
        if (t.task_category === slug) t.task_category = "study";
      });
      data.tasks = safeData.tasks;
    }

    data.goals = safeData.goals;
    return { success: true };
  }

  if (toolName === "add_habit") {
    const parsed = habitCreateSchema.safeParse(input);
    if (!parsed.success) throw new Error("Invalid add_habit arguments");
    const safeData = ensureAxisUserDataShape(data);
    const habit = {
      id: `habit_${Date.now()}_${crypto.randomBytes(6).toString("hex")}`,
      name: String(parsed.data.name || "").trim(),
      time: String(parsed.data.time || "").trim(),
      description: String(parsed.data.description || "").trim(),
      createdAt: new Date().toISOString(),
    };
    safeData.dailyHabits.push(habit);
    data.dailyHabits = safeData.dailyHabits;
    return { habit };
  }

  if (toolName === "delete_habit") {
    const parsed = assistantDeleteHabitSchema.safeParse(input);
    if (!parsed.success) throw new Error("Invalid delete_habit arguments");
    const safeData = ensureAxisUserDataShape(data);
    const before = safeData.dailyHabits.length;
    safeData.dailyHabits = safeData.dailyHabits.filter((h) => h && String(h.id) !== parsed.data.id);
    if (safeData.dailyHabits.length === before) throw new Error("Habit not found");
    data.dailyHabits = safeData.dailyHabits;
    return { success: true };
  }

  if (toolName === "rebalance_schedule") {
    const parsed = assistantRebalanceSchema.safeParse(input);
    if (!parsed.success) throw new Error("Invalid rebalance_schedule arguments");
    const blocks = await aiRebalanceScheduleFromUserData({
      data,
      horizonDays: parsed.data.horizonDays,
      maxHoursPerDay: parsed.data.maxHoursPerDay,
    });
    data.schedule = blocks;
    data.lastRebalancedAt = new Date().toISOString();
    return { blocksAdded: blocks.length };
  }

  if (toolName === "get_calendar_links") {
    const token = await getOrCreateCalendarToken(userId);
    if (!token) throw new Error("User not found");
    const baseUrl = `${req.protocol}://${req.get("host")}`;
    const subscribeUrl = `${baseUrl}/api/calendar/subscribe/${token}.ics`;
    const webcalUrl = subscribeUrl.replace(/^https?:\/\//, "webcal://");
    return { token, subscribeUrl, webcalUrl };
  }

  throw new Error(`Unknown tool: ${toolName}`);
}

function buildAssistantPlannerPrompt({ message, snapshot, history }) {
  const toolsJson = JSON.stringify(ASSISTANT_TOOLS, null, 2);
  const snapshotJson = JSON.stringify(snapshot, null, 2);
  const historyBlock = history ? `Recent conversation (most recent last):\n${history}` : "Recent conversation: (none)";
  return `
You are an agent running inside Axis (a student planner). You can take actions via tools.

Rules:
- You MUST return strict JSON only matching: {"assistant_reply":"string","tool_calls":[{"name":"tool","arguments":{...}}]}.
- Only call tools from the provided tool list.
- Use the conversation history to resolve references and keep context.
- If the user request is ambiguous, ask a clarifying question in assistant_reply and leave tool_calls empty.
- Never invent task, habit, or goal IDs; use snapshot/listed IDs.
- Do NOT delete anything unless the user explicitly asked.
- Keep tool_calls minimal (0–4 is ideal).
- If the user asks to add/edit tasks, use create_task/update_task. If the user asks to manage goals, use create_goal/update_goal.
- Only rebalance_schedule if the user asks to change/rebalance the schedule.
- assistant_reply may use Markdown (bullets, **bold**, *italics*, ++underline++, $math$).

${historyBlock}

User message:
${message}

Current user snapshot:
${snapshotJson}

Available tools (JSON):
${toolsJson}
`.trim();
}

function buildAssistantFinalPrompt({ message, toolResults, history }) {
  const resultsJson = JSON.stringify(toolResults, null, 2);
  const historyBlock = history ? `Recent conversation (most recent last):\n${history}` : "Recent conversation: (none)";
  return `
Return JSON only: {"reply":"..."}.

Write a concise, helpful response to the user. Summarize what you changed (if anything), and suggest the next best step.
The reply may include Markdown (bullets, **bold**, *italics*, ++underline++, $math$). Keep JSON valid by escaping newlines as \\n.

${historyBlock}

User message:
${message}

Tool results:
${resultsJson}
`.trim();
}

function buildAssistantFinalTextPrompt({ message, toolResults, history }) {
  const resultsJson = JSON.stringify(toolResults, null, 2);
  const historyBlock = history ? `Recent conversation (most recent last):\n${history}` : "Recent conversation: (none)";
  return `
Write the final user-facing reply in Markdown (no JSON).
Be concise, but include any important outcomes:
- What you changed (tasks, habits, goals, schedule).
- If something failed, say what and what to do next.
Use Markdown for formatting. For underline, use ++text++. For formulas, use $...$ or $$...$$.

${historyBlock}

User message:
${message}

Tool results:
${resultsJson}
`.trim();
}

app.get("/api/assistant/tools", authenticateToken, async (req, res) => {
  try {
    const data = await getUserData(req.user.userId);
    const safeData = ensureAxisUserDataShape(data);
    res.json({
      provider: resolveProviderForUserData(safeData),
      supportedProviders: Array.from(SUPPORTED_LLM_PROVIDERS).sort(),
      configuredProviders: listConfiguredProviders(),
      tools: ASSISTANT_TOOLS,
    });
  } catch (err) {
    console.error("assistant tools error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.post(
  "/api/assistant/agent",
  authenticateToken,
  validateBody(assistantAgentRequestSchema),
  async (req, res) => {
    try {
      const userId = req.user.userId;
      const rawData = await getUserData(userId);
      if (!rawData) return res.status(404).json({ error: "User data not found" });

      const data = ensureAxisUserDataShape(rawData);
      const provider = resolveProviderForUserData(data);
      const snapshot = buildAssistantSnapshot(data);
      const message = req.body.message;
      const history = formatAssistantHistory(data);

      const planResult = await callLlmWithJsonRepair({
        provider,
        system:
          "You are Axis Assistant, an agent that can update the user's planner via tools (tasks, goals, habits, schedule). Return strict JSON only. Do not wrap JSON in markdown fences.",
        user: buildAssistantPlannerPrompt({ message, snapshot, history }),
        temperature: 0.15,
        maxTokens: 900,
        schema: assistantPlannerResponseSchema,
        schemaHint: '{"assistant_reply":"string","tool_calls":[{"name":"tool","arguments":{...}}]}',
      });

      if (!planResult.ok) {
        return res.status(502).json({ error: "Assistant returned invalid JSON." });
      }

      const toolCalls = planResult.data.tool_calls.slice(0, 6);
      const toolResults = [];
      let changed = false;

      for (const call of toolCalls) {
        const toolName = String(call?.name || "").trim();
        if (!toolName) continue;
        try {
          const result = await executeAssistantTool({
            name: toolName,
            args: call.arguments,
            userId,
            data,
            req,
          });
          toolResults.push({ name: toolName, ok: true, result });
          if (!["get_snapshot", "list_tasks", "get_calendar_links"].includes(toolName)) {
            changed = true;
          }
        } catch (err) {
          toolResults.push({ name: toolName, ok: false, error: err.message || String(err) });
        }
      }

      let reply = planResult.data.assistant_reply;

      if (toolCalls.length) {
        const finalResult = await callLlmWithJsonRepair({
          provider,
          system:
            "You are Axis Assistant. Produce the final user-facing message. Return strict JSON only. Do not wrap JSON in markdown fences.",
          user: buildAssistantFinalPrompt({ message, toolResults, history }),
          temperature: 0.2,
          maxTokens: 700,
          schema: assistantFinalResponseSchema,
          schemaHint: '{"reply":"string"}',
        });

        if (finalResult.ok) {
          reply = finalResult.data.reply;
        }
      }

      const historyChanged = appendAssistantHistory(data, [
        { role: "user", content: message },
        { role: "assistant", content: reply },
      ]);
      if (changed || historyChanged) {
        await saveUserData(userId, data);
      }

      const response = { reply, data };
      if (toolCalls.length) response.toolResults = toolResults;
      res.json(response);
    } catch (err) {
      console.error("assistant agent error:", err);
      res.status(500).json({ error: err.message || "Assistant failed" });
    }
  },
);

app.post(
  "/api/assistant/agent/stream",
  authenticateToken,
  validateBody(assistantAgentRequestSchema),
  async (req, res) => {
    setSseHeaders(res);
    const abort = new AbortController();
    req.on("close", () => abort.abort());

    try {
      const userId = req.user.userId;
      const rawData = await getUserData(userId);
      if (!rawData) {
        sseSend(res, "error", { error: "User data not found" });
        res.end();
        return;
      }

      const data = ensureAxisUserDataShape(rawData);
      const provider = resolveProviderForUserData(data);
      const snapshot = buildAssistantSnapshot(data);
      const message = req.body.message;
      const history = formatAssistantHistory(data);

      sseSend(res, "meta", { provider });
      sseSend(res, "status", { stage: "planning" });

      const planResult = await callLlmWithJsonRepair({
        provider,
        system:
          "You are Axis Assistant, an agent that can update the user's planner via tools (tasks, goals, habits, schedule). Return strict JSON only. Do not wrap JSON in markdown fences.",
        user: buildAssistantPlannerPrompt({ message, snapshot, history }),
        temperature: 0.15,
        maxTokens: 900,
        schema: assistantPlannerResponseSchema,
        schemaHint: '{"assistant_reply":"string","tool_calls":[{"name":"tool","arguments":{...}}]}',
      });

      if (!planResult.ok) {
        sseSend(res, "error", { error: "Assistant returned invalid JSON." });
        res.end();
        return;
      }

      const toolCalls = planResult.data.tool_calls.slice(0, 6);
      const toolResults = [];
      let changed = false;

      sseSend(res, "status", { stage: "acting", toolCalls: toolCalls.length });

      for (const call of toolCalls) {
        const toolName = String(call?.name || "").trim();
        if (!toolName) continue;
        sseSend(res, "status", { stage: "tool_start", tool: toolName });
        try {
          const result = await executeAssistantTool({
            name: toolName,
            args: call.arguments,
            userId,
            data,
            req,
          });
          toolResults.push({ name: toolName, ok: true, result });
          if (!["get_snapshot", "list_tasks", "get_calendar_links"].includes(toolName)) {
            changed = true;
          }
          sseSend(res, "status", { stage: "tool_done", tool: toolName, ok: true });
        } catch (err) {
          const error = err?.message || String(err);
          toolResults.push({ name: toolName, ok: false, error });
          sseSend(res, "status", { stage: "tool_done", tool: toolName, ok: false, error });
        }
      }

      if (!toolCalls.length) {
        const reply = planResult.data.assistant_reply;
        const historyChanged = appendAssistantHistory(data, [
          { role: "user", content: message },
          { role: "assistant", content: reply },
        ]);
        if (changed || historyChanged) {
          await saveUserData(userId, data);
        }
        sseSend(res, "token", { token: reply });
        sseSend(res, "result", { reply, toolResults, data });
        sseSend(res, "done", {});
        res.end();
        return;
      }

      sseSend(res, "status", { stage: "responding" });

      let reply = "";
      reply = await callLlmStream({
        system:
          "You are Axis Assistant. Write the final user-facing reply in Markdown only (no JSON).",
        user: buildAssistantFinalTextPrompt({ message, toolResults, history }),
        temperature: 0.25,
        maxTokens: 700,
        provider,
        onToken: (token) => sseSend(res, "token", { token }),
        signal: abort.signal,
      });

      const historyChanged = appendAssistantHistory(data, [
        { role: "user", content: message },
        { role: "assistant", content: reply },
      ]);
      if (changed || historyChanged) {
        await saveUserData(userId, data);
      }

      sseSend(res, "result", { reply, toolResults, data });
      sseSend(res, "done", {});
    } catch (err) {
      const message = err?.name === "AbortError" ? "Client disconnected." : err?.message || String(err);
      sseSend(res, "error", { error: message });
    } finally {
      res.end();
    }
  },
);

// Profile update endpoint
app.put("/api/user/profile", authenticateToken, async (req, res) => {
  try {
    const { name } = req.body;
    
    if (!name || typeof name !== "string" || name.trim().length === 0) {
      return res.status(400).json({ error: "Name is required" });
    }

    const users = await getUsers();
    const userEmail = req.user.email;
    
    if (!users[userEmail]) {
      return res.status(404).json({ error: "User not found" });
    }

    // Update user name
    users[userEmail].name = name.trim();
    users[userEmail].updatedAt = new Date().toISOString();
    await saveUsers(users);

    res.json({ 
      success: true, 
      user: { 
        id: users[userEmail].id, 
        email: userEmail, 
        name: users[userEmail].name 
      } 
    });
  } catch (err) {
    console.error("Profile update error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Password change endpoint
app.put("/api/user/password", authenticateToken, async (req, res) => {
  try {
    const { currentPassword, newPassword } = req.body;
    
    if (!currentPassword || !newPassword) {
      return res.status(400).json({ error: "Current and new passwords are required" });
    }

    if (newPassword.length < 8) {
      return res.status(400).json({ error: "New password must be at least 8 characters" });
    }

    const users = await getUsers();
    const userEmail = req.user.email;
    const user = users[userEmail];
    
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    // Google-only accounts don't have passwords
    if (!user.password && user.googleId) {
      return res.status(400).json({ error: "Cannot change password for Google-linked accounts" });
    }

    // Verify current password
    const validPassword = await bcrypt.compare(currentPassword, user.password);
    if (!validPassword) {
      return res.status(401).json({ error: "Current password is incorrect" });
    }

    // Hash and save new password
    user.password = await bcrypt.hash(newPassword, 10);
    user.updatedAt = new Date().toISOString();
    await saveUsers(users);

    res.json({ success: true, message: "Password updated successfully" });
  } catch (err) {
    console.error("Password change error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Account deletion endpoint
app.delete("/api/user/account", authenticateToken, async (req, res) => {
  try {
    const users = await getUsers();
    const userEmail = req.user.email;
    
    if (!users[userEmail]) {
      return res.status(404).json({ error: "User not found" });
    }

    const userId = users[userEmail].id;

    // Delete user data file
    const userDataPath = path.join(USER_DATA_DIR, `${userId}.json`);
    try {
      await fs.unlink(userDataPath);
    } catch (err) {
      // Ignore if file doesn't exist
      if (err.code !== "ENOENT") {
        console.error("Error deleting user data file:", err);
      }
    }

    // Delete user from users.json
    delete users[userEmail];
    await saveUsers(users);

    res.json({ success: true, message: "Account deleted successfully" });
  } catch (err) {
    console.error("Account deletion error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Get user info endpoint
app.get("/api/user/info", authenticateToken, async (req, res) => {
  try {
    const users = await getUsers();
    const user = users[req.user.email];
    
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    res.json({
      id: user.id,
      email: req.user.email,
      name: user.name,
      createdAt: user.createdAt,
      googleLinked: !!user.googleId
    });
  } catch (err) {
    console.error("Get user info error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.get("/api/ai/providers", authenticateToken, async (req, res) => {
  try {
    const data = await getUserData(req.user.userId);
    if (!data) return res.status(404).json({ error: "User data not found" });

    const safeData = ensureAxisUserDataShape(data);
    const supportedProviders = Array.from(SUPPORTED_LLM_PROVIDERS).sort();
    const configuredProviders = listConfiguredProviders();
    const defaultProvider = normalizeProvider(LLM_PROVIDER) || "deepseek";
    const selectedProvider = safeData.settings.aiProvider;
    const effectiveProvider = resolveProviderForUserData(safeData);

    res.json({
      supportedProviders,
      configuredProviders,
      defaultProvider,
      selectedProvider,
      effectiveProvider,
    });
  } catch (err) {
    console.error("Get AI providers error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ---------- AI Planning Endpoints (DeepSeek-powered) ----------

app.post("/api/ai/task-priority", async (req, res) => {
  try {
    const {
      description = "",
      category = "",
      deadlineDate = "",
      deadlineTime = "",
      durationHours = null,
      urgentHint = "",
      importantHint = "",
    } = req.body || {};

    if (!description || typeof description !== "string") {
      return res.status(400).json({ error: "Missing 'description' in request body." });
    }

    const normalizedUrgentHint = String(urgentHint || "").trim().toLowerCase();
    const normalizedImportantHint = String(importantHint || "").trim().toLowerCase();

    const userPrompt = `
Decide the Eisenhower priority for this task.
Return JSON only: {"task_priority":"Urgent & Important"|"Urgent, Not Important"|"Important, Not Urgent"|"Not Urgent & Not Important","reason":"short"}.
- Use the user's urgent/important hints as signals, but you may override if the deadline/duration strongly suggests otherwise.
Task description: ${description}
Category: ${category || "unknown"}
Deadline: ${deadlineDate || "unknown"} ${deadlineTime || ""}
Estimated duration (hours): ${durationHours ?? "unknown"}
User says urgent: ${normalizedUrgentHint || "unknown"}
User says important: ${normalizedImportantHint || "unknown"}
`.trim();

    const reply = await callLlm({
      system: "You are an AI planner. Return strict JSON only.",
      user: userPrompt,
      temperature: 0.2,
      maxTokens: 180,
      expectJSON: true,
    });

    const parsed = safeParseJSONFromText(reply) || {};
    const allowed = new Set([
      "Urgent & Important",
      "Urgent, Not Important",
      "Important, Not Urgent",
      "Not Urgent & Not Important",
    ]);

    if (!allowed.has(parsed.task_priority)) {
      return res.status(502).json({ error: "AI returned an invalid task_priority." });
    }

    res.json({ task_priority: parsed.task_priority, reason: parsed.reason || "" });
  } catch (err) {
    console.error("task-priority error:", err);
    res.status(500).json({ error: err.message || "Task priority failed" });
  }
});

app.post("/api/ai/prioritize-tasks", authenticateToken, async (req, res) => {
  try {
    const provider = await getEffectiveProviderForUserId(req.user.userId);
    const { tasks = [], profile = {}, timeBudgetHours = 6 } = req.body || {};
    const userPrompt = `
Given the tasks and user profile, rank the top tasks to do next.
Output JSON only: {"rankedTasks":[{"id":"task-id","score":0-100,"reason":"why","deadlineRisk":"low|medium|high","bucket":"do-first|schedule|delegate|drop"}]}
- Prefer tasks with earlier deadlines, higher priority, and small duration fits in ~${timeBudgetHours}h today.
- Avoid overcommitting; include at most 7 tasks.
Tasks: ${JSON.stringify(tasks).slice(0, 6000)}
Profile: ${JSON.stringify(profile).slice(0, 2000)}
`.trim();

    const reply = await callLlm({
      system: "You are an AI planner. Be concise and return strict JSON.",
      user: userPrompt,
      temperature: 0.2,
      maxTokens: 700,
      expectJSON: true,
      provider,
    });
    const parsed = safeParseJSONFromText(reply) || { rankedTasks: [] };
    res.json(parsed);
  } catch (err) {
    console.error("prioritize-tasks error:", err);
    res.status(500).json({ error: err.message || "Prioritization failed" });
  }
});

app.post("/api/ai/schedule", authenticateToken, async (req, res) => {
  try {
    const provider = await getEffectiveProviderForUserId(req.user.userId);
    const {
      tasks = [],
      fixedBlocks = [],
      productiveWindows = {},
      day = "today",
      maxHours = 10,
    } = req.body || {};

    const userPrompt = `
Build a simple schedule for ${day}.
Respect fixed blocks and avoid overlapping times.
Prefer placing high-priority tasks in productive windows when provided.
Return JSON only: {"blocks":[{"taskId":"id","start":"HH:MM","end":"HH:MM","reason":"short note"}]}
- Cap total scheduled work to about ${maxHours} hours.
Tasks: ${JSON.stringify(tasks).slice(0, 6000)}
Fixed blocks: ${JSON.stringify(fixedBlocks).slice(0, 3000)}
Productive windows: ${JSON.stringify(productiveWindows).slice(0, 1500)}
`.trim();

    const reply = await callLlm({
      system: "You are a time-blocking assistant. Return valid JSON only.",
      user: userPrompt,
      temperature: 0.25,
      maxTokens: 700,
      expectJSON: true,
      provider,
    });
    const parsed = safeParseJSONFromText(reply) || { blocks: [] };
    res.json(parsed);
  } catch (err) {
    console.error("schedule error:", err);
    res.status(500).json({ error: err.message || "Schedule generation failed" });
  }
});

app.post(
  "/api/ai/reschedule",
  authenticateToken,
  validateBody(aiRescheduleSchema),
  async (req, res) => {
    try {
      const provider = await getEffectiveProviderForUserId(req.user.userId);
      const {
        tasks = [],
        fixedBlocks = [],
        schedule = [],
        profile = {},
        horizonDays = 7,
        maxHoursPerDay = 10,
      } = req.body || {};

      const tasksBrief = tasks
        .filter((t) => t && typeof t === "object" && typeof t.id === "string")
        .filter((t) => !t.completed)
        .slice(0, 350)
        .map((t) => ({
          id: t.id,
          name: String(t.task_name || "").slice(0, 140),
          priority: String(t.task_priority || ""),
          category: String(t.task_category || ""),
          deadline: `${t.task_deadline || ""}T${t.task_deadline_time || "23:59"}`,
          durationHours: Number(t.task_duration_hours || 0) || 0,
        }));

      const fixedBrief = fixedBlocks
        .filter((b) => b && typeof b === "object" && b.start && b.end)
        .slice(0, 500)
        .map((b) => ({
          start: b.start,
          end: b.end,
          label: String(b.label || b.kind || "Fixed").slice(0, 80),
          category: String(b.category || ""),
        }));

      const scheduleBrief = schedule
        .filter((b) => b && typeof b === "object" && b.start && b.end)
        .slice(0, 500)
        .map((b) => ({
          taskId: b.taskId || "",
          start: b.start,
          end: b.end,
        }));

      const profileBrief = (() => {
        if (!profile || typeof profile !== "object") return {};
        const keep = [
          "procrastinator_type",
          "preferred_work_style",
          "preferred_study_method",
          "most_productive_time",
          "is_procrastinator",
          "has_trouble_finishing",
          "productive_windows",
        ];
        const out = {};
        keep.forEach((k) => {
          if (profile[k] !== undefined) out[k] = profile[k];
        });
        return out;
      })();

      const today = new Date();
      const todayIso = today.toISOString().slice(0, 10);

      const userPrompt = `
Rebalance the user's schedule for the next ${horizonDays} days starting ${todayIso}.
Return JSON only: {"blocks":[{"taskId":"task-id","start":"ISO-8601 UTC","end":"ISO-8601 UTC","reason":"short"}]}.

Hard rules:
- "start" and "end" MUST be ISO-8601 timestamps in UTC with a trailing "Z", e.g. "2026-01-05T14:00:00Z".
- Do not create blocks that overlap fixedBlocks.
- Do not overlap your own blocks.
- Only use taskIds from the provided tasks list.
- Keep total scheduled work per day <= ${maxHoursPerDay} hours.
- Each block must be at least 15 minutes and end after start.

Soft rules:
- Prefer scheduling higher priority and earlier deadlines first.
- Split long tasks into multiple blocks, adding small buffers when reasonable.
- Use the user's focus preferences when provided in profile.

Tasks: ${JSON.stringify(tasksBrief).slice(0, 7000)}
Fixed blocks (unavailable): ${JSON.stringify(fixedBrief).slice(0, 7000)}
Current schedule (may be ignored): ${JSON.stringify(scheduleBrief).slice(0, 6000)}
Profile: ${JSON.stringify(profileBrief).slice(0, 2000)}
`.trim();

      const reply = await callLlm({
        system: "You are a time-blocking assistant. Return strict JSON only.",
        user: userPrompt,
        temperature: 0.25,
        maxTokens: 1200,
        expectJSON: true,
        provider,
      });

      const parsed = safeParseJSONFromText(reply) || {};
      const blocks = Array.isArray(parsed.blocks) ? parsed.blocks : [];
      if (!blocks.length) {
        return res.status(502).json({ error: "AI returned no schedule blocks." });
      }

      // Basic validation (client also validates). Reject if nothing survives.
      const taskIdSet = new Set(tasksBrief.map((t) => t.id));
      const normalized = [];
      for (const b of blocks) {
        if (!b || typeof b !== "object") continue;
        if (!taskIdSet.has(b.taskId)) continue;
        const start = new Date(b.start);
        const end = new Date(b.end);
        if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) continue;
        if (end <= start) continue;
        normalized.push({
          taskId: b.taskId,
          start: start.toISOString(),
          end: end.toISOString(),
          reason: typeof b.reason === "string" ? b.reason.slice(0, 140) : "",
        });
      }

      if (!normalized.length) {
        return res.status(502).json({ error: "AI returned invalid schedule blocks." });
      }

      res.json({ blocks: normalized });
    } catch (err) {
      console.error("reschedule error:", err);
      res.status(500).json({ error: err.message || "Reschedule failed" });
    }
  },
);

app.post("/api/ai/reflection-summary", authenticateToken, async (req, res) => {
  try {
    const provider = await getEffectiveProviderForUserId(req.user.userId);
    const { reflections = [], goals = [] } = req.body || {};
    const userPrompt = `
Summarize the recent reflections and suggest a weekly focus.
Return JSON only: {"summary":"2-3 bullet sentences","focus":"one theme","habit":"one small habit","risk":"one risk to watch"}
Reflections: ${JSON.stringify(reflections).slice(0, 5000)}
Goals: ${JSON.stringify(goals).slice(0, 3000)}
`.trim();

    const reply = await callLlm({
      system: "You are a concise coach. JSON only.",
      user: userPrompt,
      temperature: 0.3,
      maxTokens: 500,
      expectJSON: true,
      provider,
    });
    const parsed = safeParseJSONFromText(reply) || {};
    res.json(parsed);
  } catch (err) {
    console.error("reflection-summary error:", err);
    res.status(500).json({ error: err.message || "Reflection analysis failed" });
  }
});

app.post("/api/ai/mood-plan", authenticateToken, async (req, res) => {
  try {
    const provider = await getEffectiveProviderForUserId(req.user.userId);
    const { mood = "neutral", energy = "medium", tasks = [] } = req.body || {};
    const userPrompt = `
Given mood "${mood}" and energy "${energy}", pick matching work styles.
Return JSON only: {"plan":"short guidance","suggestedTasks":["taskId",...],"break":"break advice"}
Tasks: ${JSON.stringify(tasks).slice(0, 3000)}
`.trim();

    const reply = await callLlm({
      system: "You are an emotion-aware study coach. JSON only.",
      user: userPrompt,
      temperature: 0.35,
      maxTokens: 400,
      expectJSON: true,
      provider,
    });
    const parsed = safeParseJSONFromText(reply) || {};
    res.json(parsed);
  } catch (err) {
    console.error("mood-plan error:", err);
    res.status(500).json({ error: err.message || "Mood plan failed" });
  }
});

app.post("/api/ai/habit", authenticateToken, async (req, res) => {
  try {
    const provider = await getEffectiveProviderForUserId(req.user.userId);
    const { goals = [], recentTasks = [] } = req.body || {};
    const userPrompt = `
Suggest one tiny daily habit that supports the goals.
Return JSON only: {"habit":"one line","when":"time suggestion","why":"short reason"}
Goals: ${JSON.stringify(goals).slice(0, 2000)}
Recent tasks: ${JSON.stringify(recentTasks).slice(0, 2000)}
`.trim();

    const reply = await callLlm({
      system: "You are a behavior change coach. JSON only.",
      user: userPrompt,
      temperature: 0.35,
      maxTokens: 400,
      expectJSON: true,
      provider,
    });
    const parsed = safeParseJSONFromText(reply) || {};
    res.json(parsed);
  } catch (err) {
    console.error("habit error:", err);
    res.status(500).json({ error: err.message || "Habit suggestion failed" });
  }
});

app.post("/api/ai/focus-tuning", authenticateToken, async (req, res) => {
  try {
    const provider = await getEffectiveProviderForUserId(req.user.userId);
    const { blocks = [], estimates = [] } = req.body || {};
    const userPrompt = `
Given recent focus blocks and estimate accuracy, suggest block length.
Return JSON only: {"lengthMinutes":25,"bufferMinutes":5,"tip":"one sentence","reason":"short"}
Blocks: ${JSON.stringify(blocks).slice(0, 4000)}
Estimates: ${JSON.stringify(estimates).slice(0, 2000)}
`.trim();

    const reply = await callLlm({
      system: "You are a focus coach. JSON only.",
      user: userPrompt,
      temperature: 0.3,
      maxTokens: 350,
      expectJSON: true,
      provider,
    });
    const parsed = safeParseJSONFromText(reply) || {};
    res.json(parsed);
  } catch (err) {
    console.error("focus-tuning error:", err);
    res.status(500).json({ error: err.message || "Focus tuning failed" });
  }
});

// Serve the existing static front-end (index.html, script.js, style.css, etc.)
// Disable caching for development to ensure latest files are always served
app.use(express.static(path.join(__dirname), {
  etag: false,
  lastModified: false,
  setHeaders: (res, filePath) => {
    // Disable caching for HTML, CSS, and JS files
    if (filePath.endsWith('.html') || filePath.endsWith('.css') || filePath.endsWith('.js')) {
      res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
      res.setHeader('Pragma', 'no-cache');
      res.setHeader('Expires', '0');
      res.setHeader('Surrogate-Control', 'no-store');
    }
  }
}));

app.post("/api/chat/stream", async (req, res) => {
  const { message, context, provider } = req.body || {};
  if (!message || typeof message !== "string") {
    return res.status(400).json({ error: "Missing 'message' in request body." });
  }

  setSseHeaders(res);

  const abort = new AbortController();
  req.on("close", () => abort.abort());

  const systemPrompt = [
    "You are Axis, a supportive, professional AI study planner.",
    "You help students prioritize tasks, manage time, and reduce procrastination.",
    "Be concrete and actionable; ask a clarifying question when needed.",
    "Avoid long essays; prefer short bullets when helpful.",
    "Use Markdown for formatting (bullets, **bold**, *italics*, ++underline++, $math$).",
  ].join(" ");

  let userContent = message;
  if (context && typeof context === "string") {
    userContent = `Context:\n${context}\n\nUser question:\n${message}`;
  }

  const resolvedProvider = resolveProviderForRequest(provider || LLM_PROVIDER || "deepseek");
  sseSend(res, "meta", { provider: resolvedProvider });

  let reply = "";
  try {
    reply = await callLlmStream({
      system: systemPrompt,
      user: userContent,
      temperature: 0.7,
      maxTokens: 512,
      provider: resolvedProvider,
      onToken: (token) => sseSend(res, "token", { token }),
      signal: abort.signal,
    });
    sseSend(res, "done", { reply });
  } catch (err) {
    const message = err?.name === "AbortError" ? "Client disconnected." : err?.message || String(err);
    sseSend(res, "error", { error: message });
  } finally {
    res.end();
  }
});

app.post("/api/chat", async (req, res) => {
  try {
    const { message, context } = req.body || {};
    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "Missing 'message' in request body." });
    }

    const systemPrompt =
      "You are Axis, a supportive, gender-neutral, professional AI study planner. " +
      "You help students prioritize tasks, manage time, combat procrastination, and protect work-life balance. " +
      "Keep answers short, concrete, and actionable. Never encourage procrastination. " +
      "Use Markdown for formatting (bullets, **bold**, *italics*, ++underline++, $math$).";

    let userContent = message;
    if (context && typeof context === "string") {
      userContent = `Context:\n${context}\n\nUser question:\n${message}`;
    }

    const reply = await callLlm({
      system: systemPrompt,
      user: userContent,
      temperature: 0.7,
      maxTokens: 512,
    });

    res.json({ reply });
  } catch (err) {
    console.error("Error in /api/chat:", err);
    res.status(502).json({ error: err.message || "Upstream AI error." });
  }
});

app.listen(PORT, () => {
  console.log(`Axis server running at http://localhost:${PORT}`);
});

#!/usr/bin/env node
/**
 * DeepSeek Free → OpenAI-совместимый прокси.
 *
 * Использует браузерную сессию DeepSeek (бесплатно) и предоставляет
 * OpenAI-совместимый REST API для любых клиентов.
 *
 * Запуск:          node server.mjs
 * Первый вход:     node server.mjs --login
 */

import http from "node:http";
import url from "node:url";
import fs from "node:fs";
import { DeepSeekClient } from "./src/client.mjs";
import {
  loginAndSaveAuth,
  readSavedAuth,
  refreshAuthFromProfile,
  clearProfileSession,
} from "./src/auth.mjs";
import { BASE_URL } from "./src/config.mjs";

// ─── Config ───────────────────────────────────────────────
const PORT = parseInt(process.argv[2] || process.env.PORT || "18632", 10);
const HOST = process.env.HOST || "0.0.0.0";

// ─── Auth state ───────────────────────────────────────────
let auth = { cookieHeader: "", token: "" };

async function initAuth(forceLogin = false) {
  if (forceLogin) {
    await clearProfileSession().catch(() => {});
    const result = await loginAndSaveAuth();
    auth.cookieHeader = result.cookieHeader;
    auth.token = result.token;
    console.log("[auth] ✅ Новый вход выполнен успешно");
    return;
  }

  // Try saved auth
  const saved = readSavedAuth();
  if (saved) {
    auth.cookieHeader = saved.cookieHeader;
    auth.token = saved.token;
    console.log("[auth] ✅ Загружена сохранённая авторизация");
    return;
  }

  // No auth — open login window
  console.log("[auth] Нет сохранённой авторизации. Открываю окно логина...");
  const result = await loginAndSaveAuth();
  auth.cookieHeader = result.cookieHeader;
  auth.token = result.token;
  console.log("[auth] ✅ Авторизация получена");
}

function createClient() {
  return new DeepSeekClient({ cookieHeader: auth.cookieHeader, token: auth.token, debug: false });
}

// ─── OpenAI → DeepSeek conversion ─────────────────────────
function messagesToPrompt(messages) {
  return messages
    .map((m) => {
      const role = m.role === "assistant" ? "Assistant" : "User";
      let content = "";
      if (typeof m.content === "string") content = m.content;
      else if (Array.isArray(m.content))
        content = m.content.filter((c) => c.type === "text").map((c) => c.text).join("\n");
      return `${role}: ${content}`;
    })
    .join("\n\n") + "\n\nAssistant:";
}

function openaiChunk(id, created, model, content, finishReason = null) {
  return `data: ${JSON.stringify({
    id, object: "chat.completion.chunk", created, model,
    choices: [{ index: 0, delta: content ? { content, role: "assistant" } : {}, logprobs: null, finish_reason: finishReason }],
  })}\n\n`;
}

function openaiDone() { return "data: [DONE]\n\n"; }

function openaiFull(id, created, model, content) {
  return JSON.stringify({
    id, object: "chat.completion", created, model,
    choices: [{ index: 0, message: { role: "assistant", content }, logprobs: null, finish_reason: "stop" }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  });
}

// ─── Completion handler ────────────────────────────────────
async function handleCompletion(reqBody) {
  const { prompt, stream, model, temperature, max_tokens } = parseReq(reqBody);
  const client = createClient();
  const sessionId = await client.createSession();

  const modelLower = (model || "").toLowerCase();
  let modelType = null;
  if (modelLower.includes("reasoner") || modelLower.includes("r1")) modelType = "deepseek-reasoner";

  if (stream) {
    return {
      type: "stream",
      run: async (onChunk, onDone, onError) => {
        try {
          const id = `chatcmpl-${Date.now()}`;
          const created = Math.floor(Date.now() / 1000);
          onChunk(openaiChunk(id, created, model, "", null));

          await client.complete({ sessionId, prompt, modelType, thinkingEnabled: false, searchEnabled: false, onText: (text) => { onChunk(openaiChunk(id, created, model, text, null)); } });

          onChunk(openaiChunk(id, created, model, "", "stop"));
          onChunk(openaiDone());
          onDone();
        } catch (e) { onError(e); }
      },
    };
  }

  let fullText = "";
  await client.complete({ sessionId, prompt, modelType, thinkingEnabled: false, searchEnabled: false, onText: (text) => { fullText += text; } });

  const id = `chatcmpl-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);
  return { type: "json", body: openaiFull(id, created, model, fullText) };
}

function parseReq(body) {
  const messages = body.messages || [];
  const stream = body.stream === true;
  const model = body.model || "deepseek-chat";
  return { prompt: messagesToPrompt(messages), stream, model };
}

// ─── HTTP Server ─────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }

  const parsed = url.parse(req.url, true);
  const pathname = parsed.pathname;

  try {
    if (req.method === "GET" && pathname === "/v1/models") {
      const models = [
        { id: "deepseek-chat", object: "model", created: Date.now(), owned_by: "deepseek" },
        { id: "deepseek-reasoner", object: "model", created: Date.now(), owned_by: "deepseek" },
        { id: "deepseek-r1", object: "model", created: Date.now(), owned_by: "deepseek" },
      ];
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ object: "list", data: models }));
      return;
    }

    if (req.method === "GET" && (pathname === "/health" || pathname === "/")) {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok", auth_loaded: !!auth.token, port: PORT, deepseek_url: BASE_URL }));
      return;
    }

    if (req.method === "POST" && pathname === "/v1/chat/completions") {
      const buffers = [];
      for await (const chunk of req) buffers.push(chunk);
      const body = JSON.parse(Buffer.concat(buffers).toString());

      const result = await handleCompletion(body);

      if (result.type === "stream") {
        res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
        let closed = false;
        req.on("close", () => { closed = true; });
        result.run(
          (chunk) => { if (!closed) res.write(chunk); },
          () => { if (!closed) res.end(); },
          (error) => {
            console.error(`[stream] ${error.message}`);
            if (!closed) { res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`); res.end(); }
          },
        );
      } else {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(result.body);
      }
      return;
    }

    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "not_found", message: `Path ${pathname} not found` }));

  } catch (error) {
    console.error(`[error] ${error.message}`);
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "internal_error", message: error.message }));
  }
});

// ─── Main ─────────────────────────────────────────────────
const args = process.argv.slice(2);

// Справка
if (args.includes("--help") || args.includes("-h")) {
  console.log(`
Использование: node server.mjs [команда] [порт]

Команды:
  (без команды)          Запуск прокси-сервера
  --login                Логин в DeepSeek через Playwright (отдельное окно)
  --connect [порт]       Подключиться к твоему Chrome через CDP (по умолч. порт 9222)
  --import <файл> <токен> Импорт cookies из JSON-файла + userToken вручную
  --manual               Показать инструкцию по ручному экспорту сессии
  --help                 Эта справка

Примеры:
  node server.mjs                # Запуск сервера
  node server.mjs --login        # Войти в DeepSeek через новое окно
  node server.mjs --connect      # Забрать сессию из твоего Chrome
  node server.mjs --import cookies.json "sk-or-...token"
  `);
  process.exit(0);
}

// Ручной импорт cookies
const importIdx = args.indexOf("--import");
if (importIdx >= 0) {
  const cookiesFile = args[importIdx + 1];
  const userToken = args[importIdx + 2];
  if (!cookiesFile || !userToken) {
    console.error("Ошибка: нужно указать файл cookies и токен.");
    console.error("  node server.mjs --import cookies.json \"<userToken>\"");
    process.exit(1);
  }
  try {
    const { importCookies } = await import("./src/auth.mjs");
    importCookies(cookiesFile, userToken);
    console.log("✅ Импорт готов. Запускай: node server.mjs");
    process.exit(0);
  } catch (e) {
    console.error(`❌ ${e.message}`);
    process.exit(1);
  }
}

// Подключение к Chrome через CDP
if (args.includes("--connect")) {
  const cdpPort = parseInt(args[args.indexOf("--connect") + 1], 10) || 9222;
  try {
    const { connectToRunningChrome } = await import("./src/auth.mjs");
    await connectToRunningChrome(cdpPort);
    console.log("✅ Подключение готово. Запускай: node server.mjs");
    process.exit(0);
  } catch (e) {
    console.error(`❌ ${e.message}`);
    process.exit(1);
  }
}

// Показать инструкцию
if (args.includes("--manual")) {
  const { printManualInstructions } = await import("./src/auth.mjs");
  printManualInstructions();
  process.exit(0);
}

// Логин через Playwright
if (args.includes("--login")) {
  initAuth(true).then(() => process.exit(0)).catch((e) => { console.error(e.message); process.exit(1); });
} else {
  server.listen(PORT, HOST, async () => {
    console.log(`
╔══════════════════════════════════════════════════╗
║     DeepSeek Free → OpenAI Proxy                ║
║══════════════════════════════════════════════════║
║  Порт:    ${String(PORT).padEnd(39)}║
║  Хост:    ${HOST.padEnd(39)}║
║══════════════════════════════════════════════════║
║  POST http://localhost:${PORT}/v1/chat/completions    ║
║  GET  http://localhost:${PORT}/v1/models               ║
║  GET  http://localhost:${PORT}/health                   ║
╚══════════════════════════════════════════════════╝
    `);

    try { await initAuth(); console.log("\n🚀 Сервер готов к работе!\n"); }
    catch (e) { console.log(`\n⚠️  Авторизация не загружена: ${e.message}`); console.log("   Выполни: node server.mjs --login\n"); }
  });
}

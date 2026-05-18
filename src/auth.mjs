import fs from "node:fs";
import path from "node:path";
import { BASE_URL, AUTH_FILE, BROWSER_PROFILE } from "./config.mjs";

// ─── Auth persistence ────────────────────────────────────────

export function normalizeToken(inputToken) {
  const token = String(inputToken || "").trim();
  if (!token) return "";
  try {
    const parsed = JSON.parse(token);
    if (typeof parsed === "string") return parsed.trim();
    if (parsed && typeof parsed.value === "string") return parsed.value.trim();
  } catch {}
  return token;
}

export function cookieHeaderFromArray(parsed) {
  if (!Array.isArray(parsed)) throw new Error("Cookie data must be a JSON array.");
  const usable = parsed.filter((cookie) => cookie?.name && "value" in cookie);
  if (!usable.some((cookie) => cookie.name === "ds_session_id"))
    throw new Error("Cookie file does not contain ds_session_id.");
  return usable.map((cookie) => `${cookie.name}=${cookie.value}`).join("; ");
}

export function readSavedAuth() {
  if (!fs.existsSync(AUTH_FILE)) return null;
  const parsed = JSON.parse(fs.readFileSync(AUTH_FILE, "utf8"));
  if (!parsed || typeof parsed !== "object") return null;
  const token = normalizeToken(parsed.userToken || parsed.token || "");
  const cookieHeader = cookieHeaderFromArray(parsed.cookies || []);
  return { token, cookieHeader };
}

export function writeSavedAuth({ cookies, userToken }) {
  fs.mkdirSync(path.dirname(AUTH_FILE), { recursive: true });
  fs.writeFileSync(AUTH_FILE, JSON.stringify({
    version: 1,
    savedAt: new Date().toISOString(),
    baseUrl: BASE_URL,
    profileDir: BROWSER_PROFILE,
    userToken,
    cookies,
  }, null, 2), { mode: 0o600 });
  try { fs.chmodSync(AUTH_FILE, 0o600); } catch {}
}

// ─── Browser login (Playwright) ──────────────────────────────

export async function loginAndSaveAuth() {
  const { chromium } = await import("playwright");

  // Launch browser
  fs.mkdirSync(BROWSER_PROFILE, { recursive: true });
  const context = await launchPersistentContext(chromium, false);
  const page = context.pages()[0] || (await context.newPage());
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });

  console.log("\n🔓 Откроется окно DeepSeek. Залогинься там любым способом.");
  console.log("   Окно закроется автоматически после успешного входа.\n");

  // Wait for successful API call with auth header
  await waitForAuthApiCall(context);

  // Capture cookies + token
  let cookies = [];
  let rawToken = null;
  try {
    cookies = await context.cookies(BASE_URL);
    rawToken = await page.evaluate(() => {
      try { return localStorage.getItem("userToken"); } catch { return null; }
    }).catch(() => null);
  } catch {
    await context.close().catch(() => {});
    throw new Error("Не удалось прочитать состояние из окна.");
  }

  const token = normalizeToken(rawToken || "");
  const hasSessionCookie = cookies.some((c) => c.name === "ds_session_id");
  if (!token) throw new Error("В localStorage нет userToken. Логин не завершён.");
  if (!hasSessionCookie) throw new Error("В куках нет ds_session_id. Логин не завершён.");

  writeSavedAuth({ cookies, userToken: rawToken });
  await context.close();

  console.log("✅ Авторизация DeepSeek сохранена!\n");
  return { token, cookieHeader: cookieHeaderFromArray(cookies) };
}

export async function refreshAuthFromProfile() {
  const saved = JSON.parse(fs.readFileSync(AUTH_FILE, "utf8"));
  const profileDir = saved.profileDir || BROWSER_PROFILE;
  const { chromium } = await import("playwright");

  const context = await launchPersistentContext(chromium, true);
  const page = context.pages()[0] || (await context.newPage());
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded" }).catch(() => {});

  const cookies = await context.cookies(BASE_URL);
  const userToken = await page.evaluate(() => localStorage.getItem("userToken")).catch(() => "");
  const token = normalizeToken(userToken);

  if (!token || !cookies.some((c) => c.name === "ds_session_id")) {
    await context.close();
    return null;
  }

  writeSavedAuth({ cookies, userToken });
  await context.close();
  return { token, cookieHeader: cookieHeaderFromArray(cookies) };
}

export async function clearProfileSession() {
  const { chromium } = await import("playwright");
  const context = await launchPersistentContext(chromium, true);
  try { await context.clearCookies(); } finally { await context.close().catch(() => {}); }
}

// ─── Option 2: Подключение к уже запущенному Chrome через CDP ──
// Пользователь запускает Chrome с --remote-debugging-port=9222,
// скрипт подключается и забирает сессию DeepSeek (если уже залогинен).

export async function connectToRunningChrome(cdpPort = 9222) {
  const { chromium } = await import("playwright");
  const cdpUrl = `http://127.0.0.1:${cdpPort}`;

  console.log(`\n🔗 Подключаюсь к Chrome на ${cdpUrl}...`);
  let browser;
  try { browser = await chromium.connectOverCDP(cdpUrl); }
  catch (e) {
    throw new Error(`Не удалось подключиться к Chrome.\n` +
      `1. Закрой весь Chrome\n` +
      `2. Запусти его заново этой командой в терминале:\n` +
      `   google-chrome --remote-debugging-port=${cdpPort}\n` +
      `   (Windows: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=${cdpPort})\n` +
      `   (macOS: open -a "Google Chrome" --args --remote-debugging-port=${cdpPort})\n`);
  }

  // Ищем уже открытую вкладку DeepSeek или создаём новую
  let page = null;
  const contexts = browser.contexts();
  for (const ctx of contexts) {
    for (const p of ctx.pages()) {
      const url = p.url();
      if (url.startsWith(BASE_URL)) { page = p; break; }
    }
    if (page) break;
  }

  if (!page) {
    const ctx = contexts[0] || await browser.newContext();
    page = await ctx.newPage();
  }

  // Загружаем DeepSeek и проверяем авторизацию
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded", timeout: 15000 });

  const rawToken = await page.evaluate(() => {
    try { return localStorage.getItem("userToken"); } catch { return null; }
  }).catch(() => null);

  const token = normalizeToken(rawToken || "");
  const cookies = await browser.contexts()[0].cookies(BASE_URL).catch(() => []);
  const hasSessionCookie = cookies.some((c) => c.name === "ds_session_id");

  if (!token) {
    console.log("ℹ️ Ты не залогинен в DeepSeek. Открываю страницу — зайди в аккаунт.");
    await page.goto(`${BASE_URL}/`, { waitUntil: "domcontentloaded" });
    console.log("   После входа нажми Enter в этом терминале.");
    await new Promise((resolve) => {
      process.stdin.once("data", resolve);
    });
    // После нажатия Enter — перечитываем
    const token2 = await page.evaluate(() => {
      try { return localStorage.getItem("userToken"); } catch { return null; }
    }).catch(() => null);
    const cookies2 = await browser.contexts()[0].cookies(BASE_URL).catch(() => []);
    if (!token2 || !cookies2.some((c) => c.name === "ds_session_id")) {
      await browser.close();
      throw new Error("Логин не подтверждён. Нет userToken или ds_session_id.");
    }
    writeSavedAuth({ cookies: cookies2, userToken: token2 });
    await browser.close();
    console.log("✅ Сессия из твоего Chrome сохранена!\n");
    return { token: normalizeToken(token2), cookieHeader: cookieHeaderFromArray(cookies2) };
  }

  writeSavedAuth({ cookies, userToken: rawToken });
  await browser.close();
  console.log("✅ Сессия DeepSeek из твоего Chrome сохранена!\n");
  return { token, cookieHeader: cookieHeaderFromArray(cookies) };
}

// ─── Option 3: Ручной импорт cookies + токена ──────────────────
// Формат cookies: массив объектов JSON с полями name, value (можно экспортировать
// через EditThisCookie или DevTools → Application → Cookies → Export).
// Токен: скопировать из DevTools → Application → Local Storage → userToken.

export function importCookies(cookiesFilePath, userTokenStr) {
  if (!fs.existsSync(cookiesFilePath)) {
    throw new Error(`Файл cookies не найден: ${cookiesFilePath}`);
  }

  const raw = fs.readFileSync(cookiesFilePath, "utf8");
  let cookies;
  try { cookies = JSON.parse(raw); }
  catch { throw new Error("Файл cookies должен быть валидным JSON."); }

  if (!Array.isArray(cookies)) {
    // Может быть объектом {cookies: [...]} — пробуем
    if (cookies.cookies && Array.isArray(cookies.cookies)) cookies = cookies.cookies;
    else throw new Error("Файл cookies должен быть массивом JSON-объектов.");
  }

  const token = normalizeToken(userTokenStr || "");
  if (!token) throw new Error("Токен не может быть пустым.");

  // Валидация
  const usable = cookies.filter((c) => c?.name && "value" in c);
  if (!usable.some((c) => c.name === "ds_session_id")) {
    throw new Error("В файле cookies нет ds_session_id. Убедись, что экспортировал куки с chat.deepseek.com");
  }

  const cookieHeader = cookieHeaderFromArray(usable);
  writeSavedAuth({ cookies: usable, userToken: token });

  console.log("✅ Cookies и токен импортированы!\n");
  return { token, cookieHeader };
}

// ─── Экспорт инструкции для ручного получения ─────────────────
export function printManualInstructions() {
  console.log(`
══════════════════════════════════════════════════
  Ручной экспорт сессии DeepSeek
══════════════════════════════════════════════════

  1. Открой Chrome и зайди на https://chat.deepseek.com
  2. Убедись что ты залогинен (должен быть интерфейс чата)
  3. Открой DevTools (F12 или Ctrl+Shift+I)
  4. Перейди на вкладку Application → Local Storage
     → https://chat.deepseek.com
  5. Найди ключ "userToken" и скопируй его значение целиком
  6. Перейди на вкладку Application → Cookies
     → https://chat.deepseek.com
  7. Экспортируй все куки в файл (или скопируй вручную)

  Формат cookies JSON:
  [
    {"name": "ds_session_id", "value": "...", "domain": "chat.deepseek.com", ...},
    {"name": "...", "value": "...", ...}
  ]

  Сохрани файл и выполни:
    node server.mjs --import cookies.json "<userToken>"
══════════════════════════════════════════════════
  `);
}

// ─── Browser helpers ─────────────────────────────────────────

async function launchPersistentContext(chromium, headless) {
  const tryLaunch = async () => {
    try { return await chromium.launchPersistentContext(BROWSER_PROFILE, { headless, viewport: null, args: ["--disable-blink-features=AutomationControlled"], channel: "chrome" }); }
    catch (e) {
      try { return await chromium.launchPersistentContext(BROWSER_PROFILE, { headless, viewport: null, args: ["--disable-blink-features=AutomationControlled"] }); }
      catch (e2) { throw new Error(`Chrome: ${e.message}. Chromium: ${e2.message}`); }
    }
  };

  try { return await tryLaunch(); }
  catch (error) {
    const msg = String(error?.message || "");
    if (msg.includes("SingletonLock")) {
      for (const f of ["SingletonLock", "SingletonCookie", "SingletonSocket"]) {
        try { fs.unlinkSync(path.join(BROWSER_PROFILE, f)); } catch {}
      }
      try { return await tryLaunch(); }
      catch (retryError) { throw new Error(`Не удалось открыть браузер: ${retryError.message}`); }
    }
    throw new Error(`Не удалось открыть браузер. Установи Google Chrome или выполни "npx playwright install chromium". ${error.message}`);
  }
}

function waitForAuthApiCall(context, { timeoutMs = 5 * 60 * 1000, settleMs = 800 } = {}) {
  return new Promise((resolve, reject) => {
    let done = false;
    const timer = setTimeout(() => {
      if (done) return;
      done = true;
      context.off("response", handler);
      reject(new Error(`Таймаут входа ${Math.round(timeoutMs / 1000)}с. Залогинься в окне DeepSeek.`));
    }, timeoutMs);

    const handler = async (response) => {
      if (done) return;
      try {
        const url = response.url();
        if (!url.includes("/api/v0/")) return;
        if (response.status() !== 200) return;
        const reqHeaders = response.request().headers();
        const authHdr = reqHeaders["authorization"] || reqHeaders["Authorization"];
        if (!authHdr || !/^Bearer\s+\S{10,}/.test(authHdr)) return;
        let body = null;
        try { body = await response.json(); } catch { return; }
        if (body && body.code !== undefined && body.code !== 0) return;

        done = true;
        clearTimeout(timer);
        context.off("response", handler);
        setTimeout(resolve, settleMs);
      } catch {}
    };

    context.on("response", handler);
  });
}

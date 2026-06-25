# DeepSeek Free API

**Бесплатный OpenAI-совместимый API через DeepSeek.**  
Без API-ключей, без регистрации, без платежей.  
Нужен только аккаунт DeepSeek (регистрация бесплатная).

## Как это работает

Сервис открывает окно браузера → ты логинишься в DeepSeek → сохраняет cookies + токен → проксирует запросы к DeepSeek API в формате OpenAI.

**Совместим с любыми инструментами:** Cursor, Continue.dev, Open Interpreter, Aider, OpenCode, Claude Code, кастомные скрипты — всё, что умеет OpenAI API.

---

## Быстрый старт

### 1. Установка

```bash
git clone <ссылка_на_репо> deepseek-free-api
cd deepseek-free-api
npm install                # установит Playwright и Chromium
```

> Нужен Node.js 18+ (скачать: https://nodejs.org)

### 2. Получить сессию DeepSeek (выбери один способ)

**Способ А — из твоего Chrome (рекомендую)**
Если ты уже залогинен в DeepSeek в своём Chrome — просто подключись:

```bash
# 1. Закрой Chrome полностью
# 2. Запусти Chrome заново с флагом remote-debugging:
google-chrome --remote-debugging-port=9222
# (Windows: "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222)
# (macOS: open -a "Google Chrome" --args --remote-debugging-port=9222)

# 3. Открой https://chat.deepseek.com и залогинься (если ещё нет)
# 4. Запусти подключение:
node server.mjs --connect
```

Скрипт подключится к твоему Chrome, найдёт сессию DeepSeek и сохранит её.

**Способ Б — логин через новое окно (Playwright)**

```bash
node server.mjs --login
```

Откроется отдельное окно браузера. Залогинься в DeepSeek любым способом. Окно закроется автоматически.

**Способ В — ручной импорт (без Playwright!)**
Если не хочешь ставить браузер или всё ломается:

```bash
node server.mjs --manual
```

Пошаговая инструкция: открой DevTools → скопируй `userToken` из Local Storage → экспортируй cookies → импортируй:

```bash
node server.mjs --import cookies.json "скопированный-токен"
```

### 3. Запуск

```bash
node server.mjs
```

Сервер на `http://localhost:18632`. Готов к работе.

### 4. SOCKS5 прокси (опционально)

Если DeepSeek заблокирован или нужен прокси — передай адрес SOCKS5-сервера:

```bash
node server.mjs --proxy 127.0.0.1:9150
node server.mjs --proxy socks5://user:pass@10.0.0.1:1080
```

Или через переменную окружения:

```bash
SOCKS5_PROXY=127.0.0.1:9150 node server.mjs
```

Прокси применяется ко всем запросам: API-вызовы, загрузка WASM, окно логина (Playwright).

---

## Куда вставлять

### OpenCode (рекомендую)

В `~/.config/opencode.yaml` или `opencode.json`:

```yaml
model: deepseek-chat
provider:
  id: deepseek-free
  url: http://localhost:18632/v1
  key: sk-dummy
```

Или через `opencode` CLI:

```bash
opencode model set deepseek-chat
opencode provider set http://localhost:18632/v1 --key sk-dummy
```

### Cursor

Settings → Models → Add Custom Model:
- **Name:** `deepseek-chat`
- **Endpoint:** `http://localhost:18632/v1`
- **Key:** любой (например `sk-dummy`)

### Continue.dev (`~/.continue/config.json`)

```json
{
  "models": [{
    "title": "DeepSeek Free",
    "provider": "openai",
    "model": "deepseek-chat",
    "apiBase": "http://localhost:18632/v1",
    "apiKey": "sk-dummy"
  }]
}
```

### Aider

```bash
aider --model openai/deepseek-chat --openai-api-base http://localhost:18632/v1 --openai-api-key sk-dummy
```

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:18632/v1
export ANTHROPIC_API_KEY=sk-dummy
claude
```

### curl (для проверки)

```bash
curl http://localhost:18632/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Привет! Как дела?"}],"stream":false}'
```

---

## Модели

| ID | Описание |
|---|---|
| `deepseek-chat` | DeepSeek V3 / V4 (обычный чат) |
| `deepseek-reasoner` | DeepSeek R1 (с рассуждением) |
| `deepseek-r1` | Алиас deepseek-reasoner |

---

## Команды

```bash
node server.mjs              # Запуск сервера
node server.mjs 8080         # На другом порту
node server.mjs --proxy 127.0.0.1:9150  # Через SOCKS5 прокси
node server.mjs --help       # Все команды

# Получение сессии:
node server.mjs --connect    # Из твоего Chrome (через CDP)
node server.mjs --login      # Через новое окно Playwright
node server.mjs --manual     # Инструкция по ручному экспорту
node server.mjs --import cookies.json "токен"  # Импорт вручную
```

---

## Если авторизация протухла

DeepSeek-сессия живёт несколько дней. Если сервер начал выдавать ошибки авторизации — просто перелогинься:

```bash
node server.mjs --login
node server.mjs
```

---

## Требования

- **Node.js** 18 или новее
- **Google Chrome** или Chromium (ставится автоматически при `npm install`)
- **Аккаунт DeepSeek** — регистрация на https://chat.deepseek.com (бесплатно, почта+пароль)

---

## Ограничения

- DeepSeek имеет лимиты на количество запросов с одной сессии (~20-30 в минуту)
- Не все фичи DeepSeek API доступны через веб-формат (поиск, файлы)
- Сессию нужно периодически обновлять (раз в несколько дней)

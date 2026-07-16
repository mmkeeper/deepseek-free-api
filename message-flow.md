# Поток сообщений в deepseek-free-api

## 1. Архитектура

Прокси-сервер на aiohttp, транслирующий OpenAI chat completions format в DeepSeek web API и обратно. Использует бесплатные веб-сессии DeepSeek (cookies + JWT), а не платный API-ключ.

```
Hermes (OpenAI client)  ←→  proxy (:18632)  ←→  chat.deepseek.com
```

---

## 2. Идентификаторы и связывание сессий

### 2.1 DeepSeek-сессия

Создаётся через `POST /api/v0/chat_session/create`. Возвращает UUID (напр. `0b384f46-c76e-488d-8e2a-a1a1919c481d`). Эта сессия хранит контекст диалога на стороне DeepSeek.

### 2.2 Хранилище сессий (`_session_store`)

In-memory dict, отображающий **хеш сообщений** → `(session_id, parent_message_id, had_tool_call, tool_calls_cache)`:

```python
_session_store: dict[str, tuple[str, int | None, bool, list | None]]
```

### 2.3 Вычисление ключей

**nkey** (полный ключ) — хеш всех system+user сообщений запроса:

```python
def _hash_messages(msgs):
    raw = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]  # 16 hex-символов

def _user_messages(msgs):
    return [{"role": m["role"], "content": m.get("content", "")}
            for m in msgs if m["role"] in ("user", "system")]

nkey = _hash_messages(_user_messages(messages))
```

**pkey** (префиксный ключ) — хеш всех сообщений **кроме последнего**:

```python
def _prefix_key(messages):
    prefix = messages[:-1] if len(messages) >= 1 else []
    return _hash_messages(_user_messages(prefix))
```

### 2.4 Связывание

При новом запросе:
1. Вычисляется `pkey` — хеш истории без последнего сообщения.
2. Поиск в `_session_store[pkey]`:
   - **Найден** → переиспользуем session_id и parent_message_id. DeepSeek продолжает диалог.
   - **Не найден** → создаём новую DeepSeek-сессию (`client.create_session()`).

После каждого завершённого completion сохраняются **два ключа**:

```python
_session_store[nkey] = (session_id, lastAssistantMessageId, had_tool_call, cached_tc)
_session_store[pkey] = (session_id, lastAssistantMessageId, had_tool_call, cached_tc)
```

Это гарантирует:
- `nkey` — точное совпадение сообщений (для повторного того же запроса).
- `pkey` — префикс (для последующих сообщений в том же диалоге).

При ошибке стрима все записи с этим `session_id` удаляются:

```python
for k, v in list(_session_store.items()):
    if v[0] == session_id:
        del _session_store[k]
```

---

## 3. Приём сообщений от Hermes

### 3.1 Endpoint

`POST /v1/chat/completions`

Парсинг тела запроса:

```python
body = await request.json()
messages = body.get("messages", [])
stream = body.get("stream", False)
model = strip_prefix(body.get("model", "deepseek-chat"))
tools = body.get("tools")
```

### 3.2 Модели

Модели рекламируются с префиксом `dsf-`: `dsf-deepseek-chat`, `dsf-deepseek-reasoner`, `dsf-deepseek-vision`. Префикс отрезается перед отправкой в DeepSeek.

Тип модели определяет `model_type`:
- `reasoner` или `r1` → `"expert"`
- `vision` → `"vision"`
- остальные → `"default"`

### 3.3 Определение тул-резолтов

Прокси проверяет, является ли **последнее сообщение** результатом вызова тула (тройной формат):

```python
# 1. Native OpenAI tool role
if last_msg.get("role") == "tool":
    is_tool_result = True

# 2. XML в content предыдущего assistant-сообщения
elif len(messages) >= 2:
    prev = messages[-2]
    if prev.get("role") == "assistant":
        if re.search(r'<tool_call\s+name=', prev.get("content") or ""):
            is_tool_result = True
        elif prev.get("tool_calls"):
            is_tool_result = True
```

Это покрывает три формата, которые может прислать Hermes.

---

## 4. Отправка запросов в DeepSeek

### 4.1 Полный цикл

```
1. Решить PoW (Proof of Work) challenge
2. POST /api/v0/chat/completion с:
   - chat_session_id (из хранилища или новая)
   - parent_message_id (для продолжения диалога)
   - prompt (текст сообщений)
   - model_type
   - thinking_enabled, search_enabled
   - x-ds-pow-response header
```

### 4.2 PoW (Proof of Work)

DeepSeek требует решения PoW-задачи для каждого completion-запроса:

1. `POST /api/v0/chat/create_pow_challenge` — получить challenge.
2. Решить алгоритм `DeepSeekHashV1` через wasmer (WASM-бинарник).
3. Отправить решение в заголовке `x-ds-pow-response` (base64-encoded JSON).

### 4.3 Построение prompt

Прокси строит prompt в разных форматах в зависимости от контекста:

**Новая сессия** — полная конвертация через `messages_to_prompt`:
```
System: ...\n\nAssistant: ...\n\nUser: ...\n\nAssistant:
```

**Повторное использование сессии** (системные сообщения + последнее user):
```python
prompt = (sys_text + "\n\n" + tool_ctx + "\n\nUser: " + last_content + "\n\nAssistant:").strip()
```

**Тул-резолт**:
```python
prompt = f"User: <tool_result id=\"{tc_id}\">{last_content}</tool_result>\n\nAssistant:"
```

---

## 5. Получение ответа от DeepSeek (SSE)

### 5.1 Парсинг SSE

Входящий стрим парсится в `src/sse.py`. DeepSeek использует JSON Patch-подобный формат с фрагментами (fragments):

| Тип события | Куда идёт |
|---|---|
| `THINK` fragment | `reasoning_content` (thinking) |
| `RESPONSE` / `TEMPLATE_RESPONSE` fragment | `content` (текст) |
| SNAPSHOT | Полное состояние — дельта считается сравнением с кешем |
| APPEND к `fragments/-1/content` | Одиночный токен к текущему фрагменту |

Дельты вычисляются вычитанием предыдущего содержимого фрагмента из нового:

```python
prev = frag_content.get(fid, "")
if fcontent.startswith(prev):
    delta = fcontent[len(prev):]
else:
    delta = fcontent
```

### 5.2 Поток внутри прокси

```
on_thinking(text) → openai_chunk(reasoning_content=text)
on_text(text)     → openai_chunk(content=text)
```

При переходе от thinking к text отправляется маркер `</think>`.

---

## 6. Определение и передача тулов

### 6.1 Парсинг тулов из ответа DeepSeek

После получения полного текста (или в конце стрима) вызывается `parse_tool_calls()`, которая пробует 9 регулярных выражений:

| # | Формат | Пример |
|---|---|---|
| 1 | `<invoke name="X">...<parameter>...</parameter></invoke>` | Стандартный invoke |
| 2 | `<tool_calls><invoke name="X">...</invoke></tool_calls>` | Обёртка tool_calls |
| 3 | `<tool_calls>{"name":"X","arguments":{...}}</tool_calls>` | JSON внутри tool_calls |
| 4 | `<X attr="val"/>` | Самозакрывающийся тег |
| 6 | `<X>{...}</X>` | JSON-контент |
| 8 | `<X:param>value</X:param>` | Колон-разделитель |
| 9 | `<tool_call name="X"><parameter name="P">V</parameter></tool_call>` | **Hermes-формат** |

### 6.2 Детекция тула в середине стрима

Прокси отслеживает начало XML-тула в реальном времени:

```python
m = re.search(r'<(?:invoke|tool_call)\s', text)
if m:
    # Отправляем весь текст до маркера как content
    before = _strip_tool_tags(text[:m.start()])
    if before:
        on_chunk(openai_chunk(...))
    # Всё остальное буферизируем
    tool_text_buf = text[m.start():]
    in_tool_call = True
```

В конце стрима:
- Если тулы найдены — отправляются как `tool_calls` chunks
- Если не найдены — текст фильтруется от XML-тегов и отправляется как обычный контент

### 6.3 Формат тулов в OpenAI-стриме

```python
def openai_tool_calls_chunk(chunk_id, created, model, tool_calls):
    formatted_calls = []
    for i, tc in enumerate(tool_calls):
        formatted_calls.append({
            "index": i,
            "id": f"call_{hash(tc['name']) % 100000:05d}",
            "type": "function",
            "function": {"name": tc["name"], "arguments": tc["arguments"]}
        })
    return json.dumps({
        "id": chunk_id, "object": "chat.completion.chunk",
        "created": created, "model": model,
        "choices": [{"index": 0,
                      "delta": {"role": "assistant", "tool_calls": formatted_calls},
                      "logprobs": None, "finish_reason": None}]
    })
```

ID тула генерируется из имени функции: `call_{hash(name) % 100000:05d}`. Это даёт стабильный ID для одного и того же тула.

### 6.4 Кеширование тулов

Когда Hermes присылает тот же самый набор сообщений (тот же `pkey`), а в прошлый раз DeepSeek вернул тулы, прокси **не обращается к DeepSeek**, а возвращает закешированные тулы:

```python
if had_tool_call and cached_tool_calls and not is_tool_result:
    return _build_cached_tool_call_response(chunk_id, created, model,
                                            cached_tool_calls, req_id)
```

Этот кейс возникает, когда Hermes (OpenAI SDK `1.109.1`) не распознаёт тулы в стриме и повторяет запрос с теми же `[system, user]` сообщениями.

---

## 7. Формирование ответа Hermes

### 7.1 Формат стрима (streaming)

```
1. Роль:              {"delta": {"role": "assistant"}}
2. Thinking:          {"delta": {"reasoning_content": "..."}}
3. <think>/</think>:  {"delta": {"content": "<think>"}}
4. Контент:          {"delta": {"content": "..."}}
5. Тул-коллы:         {"delta": {"role": "assistant", "tool_calls": [{...}]}}
6. Финал:             {"delta": {}, "finish_reason": "stop"|"tool_calls"}
7. [DONE]
```

### 7.2 Формат не-стрима (non-streaming)

```python
{
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "..." | None,
            "tool_calls": [...] | None,
        },
        "finish_reason": "stop" | "tool_calls"
    }],
    "thinking": "..." | None
}
```

Когда есть тулы, `content` = `None`, `tool_calls` = массив тулов, `finish_reason` = `"tool_calls"`.

---

## 8. Полные сценарии

### 8.1 Обычный чат (без тулов)

```
Hermes → POST /v1/chat/completions {messages: [system, user], stream: true}
  → Прокси: pkey=хеш([system, user])
  → _session_store[pkey] = MISS
  → DeepSeek: POST /api/v0/chat_session/create → session_id
  → DeepSeek: POST /api/v0/chat/completion (session_id, prompt)
  → SSE stream → текст → thinking + content
  → finish_reason=stop
  → _session_store[nkey] = (sid, msg_id, False, None)
  → _session_store[pkey] = (sid, msg_id, False, None)
```

### 8.2 Первый вызов тула

```
Hermes → POST /v1/chat/completions {messages: [system, user], tools: [...]}
  → Новая сессия
  → Prompt включает описание всех тулов
  → DeepSeek возвращает XML с <tool_call name="...">
  → Прокси парсит, отправляет tool_calls chunk
  → finish_reason=tool_calls
  → Кеш: had_tool_call=True, cached_tc=[{name, arguments}]
```

### 8.3 Повторный вызов тула (из кеша)

```
Hermes → POST /v1/chat/completions {messages: [system, user], tools: [...]}
  → pkey совпадает, had_tool_call=True, cached_tc есть, is_tool_result=False
  → DeepSeek НЕ вызывается
  → Прокси синтезирует стрим из кеша
  → Те же tool_calls, тот же finish_reason
```

### 8.4 Результат тула (после выполнения)

```
Hermes → POST /v1/chat/completions
  {messages: [system, user, assistant(w/tool_calls), tool(result)]}
  → is_tool_result = True
  → Сессия переиспользуется (parent_message_id указан)
  → Prompt = "User: <tool_result id=\"call_X\">...результат...</tool_result>\n\nAssistant:"
  → DeepSeek продолжает диалог с учётом результата
```

# Подключение локального прокси как TTS-провайдера Hermes

Прокси отдаёт TTS по нестандартному контракту (номер сообщения вместо текста),
поэтому штатное подключение «провайдер + base_url» невозможно — нужен плагин.

## Почему не конфигом

Встроенный OpenAI-провайдер Hermes (`tools/tts_tool_openai.py`) бьёт в
`POST {base_url}/audio/speech` с полем `input`. Прокси такого роута не имеет.
Настроить `tts.provider: openai` с `base_url` на прокси — гарантированный 404.

## Архитектура плагина: два хука

| Хук | Роль |
|---|---|
| middleware `llm_request` | на каждом исходящем запросе сохраняет `messages` в память процесса |
| `ctx.register_tts_provider` | в `synthesize()` игнорирует текст и шлёт сохранённые `messages` |

Почему именно `llm_request`: он вызывается ДО запроса и получает полный payload
(`api_kwargs["messages"]`), без нормализации, независимо от стриминга и чанкинга.
Это снимает обе проблемы: id сессии ловить не нужно, а нормализация текста
(`prepare_spoken_text`) на поиск сессии не влияет.

Почему НЕ `post_api_request`: он получает не сырой ответ, а sanitized-dict из
четырёх полей (`model`, `finish_reason`, `assistant_message`, `usage`).
`chat_session_id` туда не входит и попасть не может.

## Обязательное объявление в манифесте

`hermes plugins validate` требует декларировать middleware явно, иначе
`Validation failed` с «undeclared middleware registered»:

```yaml
name: deepseek-session-tts
version: 1.0.0
description: "TTS через локальный DeepSeek-прокси."
provides_middleware:
  - llm_request
```

## Точки расширения

```python
class MyTTS(TTSProvider):
    @property
    def name(self): return "deepseek-session"

    @property
    def voice_compatible(self): return True   # опт-ин в voice-bubble

    def is_available(self): return True

    def synthesize(self, text, output_path, *, voice=None, model=None,
                   speed=None, format="mp3", **extra):
        # text ИГНОРИРУЕТСЯ: слать надо сохранённые messages
        ...

def register(ctx):
    ctx.register_middleware("llm_request", on_llm_request)
    ctx.register_tts_provider(MyTTS())
```

`TTSProvider` — из `agent.tts_provider`; имя не должно совпадать со встроенным.
Возврат из `stream()` не нужен: неплагинные стримеры живут в отдельном реестре
`tools/tts_streaming.py`, и провайдер всегда пойдёт синхронным путём.

## Выбор провайдера

```bash
hermes config set tts.provider deepseek-session
```

Вложенные ключи вида `tts.deepseek-session.voice` реестр конфига НЕ признаёт —
`hermes config set` ответит «is not a recognized config key». Настройки плагина
читать самостоятельно из `config.yaml` (`load_config().get("tts")`), с дефолтами
в коде; либо править YAML вручную.

## Ловушки режима `/voice tts`

Режим непригоден для провайдера «озвучь последний ответ», по двум причинам:

1. **Пофразовый вызов.** `stream_tts_to_speaker` режет ответ по предложениям
   (`SentenceChunker`) и зовёт TTS на каждое. Плагин не попадает в реестр
   стримеров, значит пойдёт синхронный путь — N предложений дадут N запросов,
   каждый вернёт один и тот же последний ответ. Дублирование.
2. **`/voice tts` требует включённого `/voice on`.** `_voice_toggle_tts`
   возвращает ошибку 4014 «enable voice mode first», если `HERMES_VOICE != "1"`.
   Сам по себе `/voice tts` озвучку не включает.

Флаги `HERMES_VOICE` / `HERMES_VOICE_TTS` — runtime-only (`os.environ`),
в `config.yaml` не пишутся. Если `/voice tts` не даёт запроса в логах —
проверять не перезапуск, а `/voice status` и порядок команд.

## Диагностика: почему нет запроса к TTS

Смотреть `logs/agent.log` по маркерам:

```
Generating speech with plugin TTS provider 'NAME'
TTS audio saved: PATH (N bytes, provider: NAME)
```

Отсутствие обеих строк означает, что TTS просто не запускался — а не что
провайдер не подключился. Регистрация провайдера видна отдельной строкой
`Plugin 'NAME' registered TTS provider: ...` и происходит на старте процесса.

## Семантика инструмента `text_to_speech`

Аргумент `text` игнорируется: озвучивается ответ DeepSeek. Это вводит в
заблуждение — агент думает, что озвучил переданный текст. При подключении
такого провайдера стоит описать это в его документации/скилле и не ожидать,
что «озвучь вот это» сработает буквально.

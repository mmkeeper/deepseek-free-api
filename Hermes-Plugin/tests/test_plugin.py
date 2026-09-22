"""Автотесты логики плагина deepseek-session TTS.

Не требуют ни сети, ни установленного Hermes: модуль плагина загружается
с заглушкой `agent.tts_provider`, а проверяется чистая логика —
санитайзер, стабилизация нормализатора, поиск номера сообщения,
кросс-сессионный сценарий.

Запуск:
    python tests/test_plugin.py          # без pytest
    python -m pytest tests/ -v           # с pytest
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

# --- заглушка agent.tts_provider, чтобы импортировать модуль без Hermes ----

if "agent" not in sys.modules:
    agent_pkg = types.ModuleType("agent")
    agent_pkg.__path__ = []
    sys.modules["agent"] = agent_pkg

if "agent.tts_provider" not in sys.modules:
    tts_mod = types.ModuleType("agent.tts_provider")

    class TTSProvider:  # noqa: D401 - заглушка
        """Минимальная заглушка базового класса провайдера."""

        @property
        def name(self) -> str:
            return "stub"

        @property
        def voice_compatible(self) -> bool:
            return True

        def is_available(self) -> bool:
            return True

        def list_voices(self):
            return []

        def synthesize(self, text, output_path, **extra):
            raise NotImplementedError

    tts_mod.TTSProvider = TTSProvider
    sys.modules["agent.tts_provider"] = tts_mod

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_PATH = os.path.join(os.path.dirname(HERE), "deepseek-session-tts", "__init__.py")

spec = importlib.util.spec_from_file_location("deepseek_session_tts_under_test", PLUGIN_PATH)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)


# ------------------------------- утилиты теста ------------------------------

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    mark = "OK  " if condition else "FAIL"
    print("%s %s%s" % (mark, name, ("  | " + detail) if detail and not condition else ""))


def fresh():
    """Сбросить состояние плагина между тестами."""
    with plugin._lock:
        plugin._sessions.clear()
    plugin._spoken.cache_clear() if hasattr(plugin._spoken, "cache_clear") else None
    if hasattr(plugin, "_SPOKEN_CACHE"):
        plugin._SPOKEN_CACHE.clear()


# --------------------------- тесты 1: санитайзер ----------------------------

def test_sanitize():
    fresh()

    # markdown-таблица снимается целиком
    table = "## Заголовок\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nТекст после."
    out = plugin._sanitize_for_speech(table)
    check("санитайзер: таблица снята", "|" not in out, repr(out))
    check("санитайзер: заголовок снят", "##" not in out, repr(out))
    check("санитайзер: текст сохранён", "Текст после" in out, repr(out))

    # код-блок снимается
    fenced = "До.\n\n```python\nx = 1\n```\n\nПосле."
    out = plugin._sanitize_for_speech(fenced)
    check("санитайзер: код-блок снят", "x = 1" not in out, repr(out))

    # inline-код разворачивается без кавычек
    out = plugin._sanitize_for_speech("Вызови `foo()` сейчас.")
    check("санитайзер: inline-код без кавычек", "`" not in out and "foo()" in out, repr(out))

    # ссылка -> текст ссылки
    out = plugin._sanitize_for_speech("Смотри [документацию](https://example.com) тут.")
    check("санитайзер: ссылка -> текст",
          "документацию" in out and "https://" not in out, repr(out))

    # одиночный перевод строки -> пробел (мягкий перенос)
    out = plugin._sanitize_for_speech("Первая строка\nВторая строка")
    check("санитайзер: одиночный перенос -> пробел",
          out == "Первая строка Вторая строка", repr(out))

    # пустая строка (двойной перенос) -> точка
    out = plugin._sanitize_for_speech("Первая строка\n\nВторая строка")
    check("санитайзер: пустая строка -> точка",
          out == "Первая строка. Вторая строка", repr(out))

    # пустая строка после пунктуации не удваивает знак
    out = plugin._sanitize_for_speech("Конец.\n\nНачало.")
    check("санитайзер: пунктуация + пустая строка не удваивает знак",
          out == "Конец. Начало.", repr(out))

    # заголовок снимается и после него ставится точка
    out = plugin._sanitize_for_speech("## Заголовок\n\nТекст.")
    check("санитайзер: заголовок + пустая строка",
          out == "Заголовок. Текст.", repr(out))


# --------------------- тесты 2: стабилизация нормализатора ------------------

def test_stabilization():
    fresh()
    try:
        from tools.tts_text_normalize import prepare_spoken_text as sp
        have_hermes = True
    except Exception:
        have_hermes = False
    check("нормализатор Hermes доступен", have_hermes,
          "тесты стабилизации пропущены: tools.tts_text_normalize не найден")
    if not have_hermes:
        return

    # _spoken должен возвращать НЕПОДВИЖНУЮ ТОЧКУ: повторный проход не меняет
    samples = [
        "Готово.",
        "skillview(name='..')",
        "Хотя.. инструкция",
        "5 кг",
        "Ответ с многоточием... и точкой.",
        "Цена 1 234,56 руб.",
    ]
    unstable = []
    for s in samples:
        once = plugin._spoken(s)
        twice = plugin._spoken(once) if False else None  # кэш вернёт то же
        # проверяем через сам нормализатор: точка не должна сдвигаться
        fixed = sp(once)
        if fixed != once:
            unstable.append((s, once, fixed))
    check("стабилизация: результат неподвижен", not unstable, repr(unstable[:2]))


# ------------------------- тесты 3: поиск индекса ---------------------------

def test_find_message_index():
    fresh()
    msgs = [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "Первый ответ."},
        {"role": "user", "content": "ещё"},
        {"role": "assistant", "content": "Второй ответ."},
        {"role": "assistant", "content": ""},          # пустой с tool_calls
        {"role": "assistant", "content": "Третий ответ."},
    ]
    # assistant-сообщений 4 (индексы 0..3), ищем с конца
    spoken = plugin._spoken("Третий ответ.")
    idx = plugin._find_message_index(msgs, spoken)
    check("поиск: находит последний ответ (индекс 3)", idx == 3, "получено %r" % idx)

    spoken = plugin._spoken("Первый ответ.")
    idx = plugin._find_message_index(msgs, spoken)
    check("поиск: находит первый ответ (индекс 0)", idx == 0, "получено %r" % idx)

    spoken = plugin._spoken("Такого текста нет.")
    idx = plugin._find_message_index(msgs, spoken)
    check("поиск: отсутствующий текст -> None", idx is None, "получено %r" % idx)

    # дубликат: побеждает ближайший к концу
    dup = [
        {"role": "assistant", "content": "Повтор."},
        {"role": "assistant", "content": "Повтор."},
    ]
    idx = plugin._find_message_index(dup, plugin._spoken("Повтор."))
    check("поиск: дубликат -> ближайший к концу (1)", idx == 1, "получено %r" % idx)

    check("поиск: пустой текст -> None",
          plugin._find_message_index(msgs, "") is None)


# --------------------- тесты 4: снапшоты и локализация ----------------------

def test_sessions_and_locate():
    fresh()
    plugin._touch("session-A", "messages", [
        {"role": "user", "content": "вопрос A"},
        {"role": "assistant", "content": "Ответ сессии A."},
    ])
    plugin._touch("session-A", "history", [
        {"role": "user", "content": "вопрос A"},
        {"role": "assistant", "content": "Ответ сессии A."},
    ])
    plugin._touch("session-B", "messages", [
        {"role": "user", "content": "вопрос B"},
        {"role": "assistant", "content": "Ответ сессии B."},
    ])

    loc = plugin._locate(plugin._spoken("Ответ сессии A."))
    check("локализация: находит сессию A", loc is not None and loc[0] == "session-A",
          repr(loc))
    check("локализация: индекс ответа A = 0",
          loc is not None and loc[1] == 0, repr(loc))

    loc = plugin._locate(plugin._spoken("Ответ сессии B."))
    check("локализация: находит сессию B", loc is not None and loc[0] == "session-B",
          repr(loc))

    loc = plugin._locate(plugin._spoken("Чужой текст, которого нет."))
    check("локализация: чужой текст -> None", loc is None, repr(loc))

    # свежий ответ есть только в history -> (session, None)
    fresh()
    plugin._touch("session-C", "messages", [
        {"role": "user", "content": "вопрос C"},
    ])
    plugin._touch("session-C", "history", [
        {"role": "user", "content": "вопрос C"},
        {"role": "assistant", "content": "Свежий ответ C."},
    ])
    loc = plugin._locate(plugin._spoken("Свежий ответ C."))
    check("локализация: свежий в history -> (C, None)",
          loc is not None and loc[0] == "session-C" and loc[1] is None, repr(loc))

    # НЕ последняя реплика в history -> НЕ локализуется (защита от чужого)
    fresh()
    plugin._touch("session-D", "history", [
        {"role": "assistant", "content": "Старый ответ D."},
        {"role": "user", "content": "ещё вопрос"},
        {"role": "assistant", "content": "Свежий ответ D."},
    ])
    loc = plugin._locate(plugin._spoken("Старый ответ D."))
    check("локализация: не-последняя в history -> None (не читаем чужое)",
          loc is None, repr(loc))


# ------------------- тесты 5: кросс-сессионная защита -----------------------

def test_cross_session():
    fresh()
    plugin._touch("session-X", "messages", [
        {"role": "assistant", "content": "Текст сессии X."},
    ])
    plugin._touch("session-X", "history", [
        {"role": "assistant", "content": "Текст сессии X."},
    ])
    plugin._touch("session-Y", "messages", [
        {"role": "assistant", "content": "Текст сессии Y."},
    ])

    # текст X не должен находиться, если его нет; а текст Y — находиться в Y
    loc = plugin._locate(plugin._spoken("Текст сессии Y."))
    check("кросс-сессия: текст Y найден в Y",
          loc is not None and loc[0] == "session-Y", repr(loc))

    loc = plugin._locate(plugin._spoken("Текст сессии X."))
    check("кросс-сессия: текст X найден в X, не в Y",
          loc is not None and loc[0] == "session-X", repr(loc))

    # пустой session_id не сохраняется
    fresh()
    plugin._touch("", "messages", [{"role": "assistant", "content": "Безадресный."}])
    check("кросс-сессия: пустой session_id не сохраняется",
          plugin._session_order() == [], repr(plugin._session_order()))


# ---------------------- тесты 6: вытеснение по LRU --------------------------

def test_lru():
    fresh()
    for i in range(plugin._MAX_SESSIONS + 3):
        plugin._touch("s%d" % i, "messages", [
            {"role": "assistant", "content": "Ответ %d." % i},
        ])
    order = plugin._session_order()
    check("LRU: число сессий не превышает лимит",
          len(order) <= plugin._MAX_SESSIONS, "их %d" % len(order))
    check("LRU: самые старые вытеснены",
          "s0" not in order and "s1" not in order, repr(order[:3]))


# -------------------------- тесты 7: конфиг/регистр -------------------------

def test_registry_shape():
    fresh()
    provider = plugin.DeepSeekSessionTTS()
    check("провайдер: имя deepseek-session", provider.name == "deepseek-session",
          provider.name)
    check("провайдер: voice_compatible True", provider.voice_compatible is True)
    check("провайдер: is_available True", provider.is_available() is True)
    voices = provider.list_voices()
    ids = sorted(v["id"] for v in voices)
    check("провайдер: список голосов", ids == ["echo", "mira", "stella", "tide"], repr(ids))

    # ключевые функции существуют
    for fn in ("_sanitize_for_speech", "_strip_md_tables", "_md_row",
               "_spoken", "_locate", "_find_message_index"):
        check("модуль: есть %s" % fn, hasattr(plugin, fn))


# ---------------------------------- прогон ----------------------------------

def main():
    print("плагин:", PLUGIN_PATH)
    print("=" * 70)
    for fn in (test_sanitize, test_stabilization, test_find_message_index,
               test_sessions_and_locate, test_cross_session, test_lru,
               test_registry_shape):
        print("\n--- %s ---" % fn.__name__)
        fn()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 70)
    print("ИТОГО: %d/%d" % (passed, total))
    if failed:
        print("ПРОВАЛЫ:")
        for name in failed:
            print("  -", name)
        return 1
    print("ВСЕ ТЕСТЫ ПРОШЛИ")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Test parse_tool_calls: new nested-element format + regression of old formats."""
import json
import sys
sys.path.insert(0, ".")

import server
from server import parse_tool_calls, _strip_tool_tags, _mask_code_fences


def test_strip_tool_tags_keeps_code_fences():
    """Стриппинг не трогает tool-разметку внутри ```-блоков (регресс: блок пропадал в Hermes)."""
    text = """Текст до.

```xml
<tool_calls>
  <tool_call name="tool_describe">
    <parameter name="name">mcp__playwright__browser_navigate</parameter>
  </tool_call>
</tool_calls>
```

Текст после. <tool_call name="REAL"><parameter name="p">v</parameter></tool_call>"""
    out = _strip_tool_tags(text)
    assert '<tool_call name="tool_describe">' in out, "fence content must survive"
    assert "mcp__playwright__browser_navigate" in out
    assert "```xml" in out and "```" in out
    assert "<tool_call name=\"REAL\">" not in out, "tags outside fences must be stripped"
    assert "Текст после." in out and "v" in out  # inner text of real call stays
    print("  PASS: _strip_tool_tags keeps fenced blocks intact")


def test_strip_tool_tags_keeps_inline_code_spans():
    """Инлайн-упоминания в одинарных бэктиках (таблицы, проза) не вырезаются."""
    text = """Формат: `<tool_calls>`, вызов — `<tool_call name="x">`.

| Элемент | Роль |
|---|---|
| `<tool_calls>` | обёртка |
| `<parameter name="...">` | аргумент |

А это реальный вызов (должен вырезаться): <tool_call name="REAL"><parameter name="p">v</parameter></tool_call>"""
    out = _strip_tool_tags(text)
    assert "`<tool_calls>`" in out and "`<parameter name=\"...\">`" in out, "inline spans must survive"
    assert "<tool_call name=\"REAL\">" not in out
    print("  PASS: inline code spans preserved")


def test_nested_format_single():
    text = """Подумаю, какой инструмент нужен.

<tool_call>
    <name>search_files</name>
    <arguments>
      <path>C:\\Projects\\ice</path>
      <pattern>*.pdf</pattern>
      <target>files</target>
      <limit>50</limit>
    </arguments>
  </tool_call>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
    assert tcs[0]["name"] == "search_files"
    args = json.loads(tcs[0]["arguments"])
    assert args == {"path": "C:\\Projects\\ice", "pattern": "*.pdf",
                    "target": "files", "limit": "50"}, f"bad args: {args}"
    print("  PASS: nested format single call")


def test_nested_format_multiple():
    text = """<tool_call>
  <name>a</name>
  <arguments>
    <x>1</x>
  </arguments>
</tool_call>
<tool_call>
  <name>b</name>
  <arguments>
    <y>2</y>
    <z>3</z>
  </arguments>
</tool_call>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 2, f"expected 2 tool calls, got {len(tcs)}: {tcs}"
    assert [t["name"] for t in tcs] == ["a", "b"]
    assert json.loads(tcs[0]["arguments"]) == {"x": "1"}
    assert json.loads(tcs[1]["arguments"]) == {"y": "2", "z": "3"}
    print("  PASS: nested format multiple calls")


def test_nested_format_cyrillic_params():
    text = """<tool_call>
  <name>имя_инструмента</name>
  <arguments>
    <параметр1>значение</параметр1>
    <путь>C:\\temp</путь>
  </arguments>
</tool_call>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
    assert tcs[0]["name"] == "имя_инструмента"
    args = json.loads(tcs[0]["arguments"])
    assert args == {"параметр1": "значение", "путь": "C:\\temp"}, f"bad args: {args}"
    print("  PASS: cyrillic tool/param names")


def test_nested_format_multiline_value():
    text = """<tool_call>
  <name>terminal</name>
  <arguments>
    <command>echo one
echo two
echo three</command>
  </arguments>
</tool_call>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
    args = json.loads(tcs[0]["arguments"])
    assert args["command"] == "echo one\necho two\necho three", f"bad args: {args}"
    print("  PASS: multiline value")


def test_nested_format_no_arguments_wrapper():
    text = """<tool_call>
  <name>t1</name>
  <x>1</x>
  <y>2</y>
</tool_call>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
    assert json.loads(tcs[0]["arguments"]) == {"x": "1", "y": "2"}
    print("  PASS: missing <arguments> wrapper tolerated")


def test_regression_hermes_format9():
    text = """<tool_call name="search_files">
<parameter name="path">C:\\Projects\\ice</parameter>
<parameter name="pattern">*.pdf</parameter>
</tool_call>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
    assert tcs[0]["name"] == "search_files"
    assert json.loads(tcs[0]["arguments"]) == {"path": "C:\\Projects\\ice", "pattern": "*.pdf"}
    print("  PASS: regression Format 9 (hermes attr style)")


def test_regression_invoke_format1():
    text = """<invoke name="search_files">
<parameter name="pattern">*.log</parameter>
</invoke>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
    assert tcs[0]["name"] == "search_files"
    assert json.loads(tcs[0]["arguments"]) == {"pattern": "*.log"}
    print("  PASS: regression Format 1 (invoke)")


def test_extra_attributes_tolerated():
    """Модель добавляет string=\"true\" и прочие атрибуты — парсер их игнорирует.

    Запреты в промпте не остановят модель: вместо жёсткого формата лучше
    терпимо парсить любые атрибуты после name= (Format 0/1/9) и на тегах.
    """
    cases = [
        # Format 1: invoke с атрибутами на invoke и parameter
        ("""<invoke name="web_search" string="true" type="function">
<parameter name="query" string="true">t</parameter>
<parameter name="limit" string="false">3</parameter>
</invoke>""",
         "web_search", {"query": "t", "limit": "3"}),
        # Format 0: usr_tool_call с атрибутами на call и параметрах
        ("""<usr_tool_calls>
  <usr_tool_call name="vision_analyze" string="true">
    <usr_parameter name="image_url" string="true">C:\\img.png</usr_parameter>
    <usr_parameter name="question" string="true">q</usr_parameter>
  </usr_tool_call>
</usr_tool_calls>""",
         "vision_analyze", {"image_url": "C:\\img.png", "question": "q"}),
    ]
    for text, name, want in cases:
        tcs = parse_tool_calls(text)
        assert len(tcs) == 1, f"expected 1 call, got {len(tcs)}: {tcs}"
        assert tcs[0]["name"] == name, tcs
        args = json.loads(tcs[0]["arguments"])
        assert args == want, f"args: {args!r}"
    print("  PASS: extra attributes on invoke/usr_ and parameter tags ignored")


def test_json_inside_arguments():
    """FULL_SCHEMA_TOOLS показывает параметры как JSON — модель может сымитировать это."""
    text = """<tool_call>
  <name>search_files</name>
  <arguments>{"path": "docs", "pattern": "*.pdf"}</arguments>
</tool_call>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
    assert tcs[0]["name"] == "search_files", f"bogus name: {tcs[0]['name']}"
    args = json.loads(tcs[0]["arguments"])
    assert args == {"path": "docs", "pattern": "*.pdf"}, f"bad args: {args}"
    print("  PASS: JSON inside <arguments> (not Format 6 mis-parse)")


def test_wrapper_with_direct_tool_tags():
    """Гибрид: plural-обёртка + имя тула как прямой тег (без <name>)."""
    text = """<tool_calls>
<session_search>
<query>анализ системного промта уменшился</query>
</session_search>
</tool_calls>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
    assert tcs[0]["name"] == "session_search"
    args = json.loads(tcs[0]["arguments"])
    assert args == {"query": "анализ системного промта уменшился"}, f"bad args: {args}"
    print("  PASS: <tool_calls> wrapper with direct tool tags")


def test_wrapper_multiple_direct_tags():
    text = """<tool_calls>
<read_file>
<path>C:\\x</path>
</read_file>
<web_search>
<query>test query</query>
</web_search>
</tool_calls>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 2, f"expected 2 tool calls, got {len(tcs)}: {tcs}"
    assert [t["name"] for t in tcs] == ["read_file", "web_search"]
    assert json.loads(tcs[0]["arguments"]) == {"path": "C:\\x"}
    print("  PASS: wrapper with multiple direct tool tags")


def test_mask_defuses_tool_word_in_fences():
    """Фенсы: tool->t00l; инлайн-спаны: забеливаются для поиска;
    реальный вызов вне их парсится, длина сохраняется."""
    src = ("до <tool_call name=\"x\"><parameter name=\"p\">1</parameter></tool_call>"
           " `упоминание <tool_calls>`\n```xml\n<tool_calls></tool_calls>\n```\nпосле")
    out = _mask_code_fences(src)
    assert "<t00l_calls>" in out and "</t00l_calls>" in out, out
    # содержимое спана сохраняется дословно (точечный суб только гравис-тегов)
    i = out.index("`", out.index("до <tool_call"))
    span = out[i:i + 30]
    assert "`" in span and span.startswith("`упоминание <tool_calls>`"), span
    assert "<tool_call name=\"x\">" in out, "real call outside spans/fences must survive"
    assert len(out) == len(src)
    # фенснутая и инлайн разметка не парсится, реальный вызов парсится
    tcs = parse_tool_calls(src)
    assert len(tcs) == 1 and tcs[0]["name"] == "x", tcs
    print("  PASS: mask defuses fences, preserves prose mentions, keeps real calls")


def test_nested_quadruple_fence_defused():
    """Вложенные фенсы (```` вокруг ```) — один регион, вызов не исполняется
    (регресс лога REQ-77bbf30013: фантомный execute_code)."""
    server.MASK_CODE_FENCES = True
    try:
        src = ("````markdown\n```xml\n<tool_calls>\n  <tool_call name=\"execute_code\">\n"
               "    <parameter name=\"code\">print('это просто текст')</parameter>\n"
               "  </tool_call>\n</tool_calls>\n```\n````\n"
               "Это текст, а не вызов.")
        out = _mask_code_fences(src)
        assert "<t00l_call" in out and "tool" not in out.split("Это текст")[0].replace("````", "").replace("```xml", ""), out
        assert len(out) == len(src)
        assert parse_tool_calls(src) == [], f"fenced example must not parse: {parse_tool_calls(src)}"
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: nested quadruple-fence example defused")


def test_tilde_fences_defused():
    """Тильды-фенсы (~~~ и ~~~~) обезвреживаются как код-блоки."""
    server.MASK_CODE_FENCES = True
    try:
        # 1. простой ~~~
        src = "Пример:\n\n~~~\n<tool_call name=\"fake\"><parameter name=\"p\">v</parameter></tool_call>\n~~~\n"
        assert parse_tool_calls(src) == [], parse_tool_calls(src)
        out = _mask_code_fences(src)
        assert "<t00l_call" in out and len(out) == len(src)

        # 2. вложенные тильды: ~~~~ вокруг ~~~
        src = ("~~~~\n~~~xml\n<tool_calls><tool_call name=\"x\">"
               "<parameter name=\"q\">1</parameter></tool_call></tool_calls>\n~~~\n~~~~")
        assert parse_tool_calls(src) == [], parse_tool_calls(src)
        out = _mask_code_fences(src)
        assert "<t00l_calls>" in out and len(out) == len(src)

        # 3. закрытие не тем символом не закрывает регион: ~~~ открыт, ``` внутри игнорируется
        src = "~~~\n<tool_call name=\"y\"></tool_call>\n```\nещё <tool_calls>\n~~~\nхвост"
        assert parse_tool_calls(src) == [], parse_tool_calls(src)
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: tilde fences (incl. nested and type-mismatch) defused")


def test_strip_tool_tags_keeps_tilde_fences():
    """Стриппинг не трогает разметку внутри ~~~-фенсов."""
    text = "до\n\n~~~\n<tool_calls><tool_call name=\"x\"><parameter name=\"p\">v</parameter></tool_call></tool_calls>\n~~~\n\nпосле <tool_call name=\"REAL\"></tool_call>"
    out = _strip_tool_tags(text)
    assert '<tool_call name="x">' in out, "tilde-fenced content must survive"
    assert "<tool_call name=\"REAL\">" not in out
    print("  PASS: _strip_tool_tags keeps tilde fences intact")


def test_midline_backtick_run_in_param_value():
    """'```' ВНУТРИ строки значения параметра — не фенс; вызов парсится
    (регресс лога REQ-a9d47a0026: write_file с проверкой '```' в коде)."""
    server.MASK_CODE_FENCES = True
    try:
        src = ('<tool_calls>\n<tool_call name="write_file">\n'
               '<parameter name="content">import os\n'
               "if any(k in head for k in ['markdown', 'fence', '```']):\n"
               "    print(fp)\n"
               "</parameter>\n"
               '<parameter name="path">C:/tmp/x.py</parameter>\n'
               "</tool_call>\n</tool_calls>")
        tcs = parse_tool_calls(src)
        assert len(tcs) == 1 and tcs[0]["name"] == "write_file", tcs
        args = json.loads(tcs[0]["arguments"])
        assert "'```'" in args["content"], args
        assert "print(fp)" in args["content"]
        assert args["path"] == "C:/tmp/x.py"
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: mid-line backtick run in param value is not a fence")


def test_plain_code_fence_with_call_xml_executes():
    """Код-блок верхнего уровня (любая инфо-строка, здесь пустая) с XML вызова —
    РЕАЛЬНЫЙ вызов, а не пример. Контентное правило."""
    server.MASK_CODE_FENCES = True
    try:
        text = """Вот как вызывать инструменты:

```
<tool_calls>
  <tool_call name="ИМЯ_ИНСТРУМЕНТА">
    <parameter name="ПАРАМЕТР">ЗНАЧЕНИЕ</parameter>
  </tool_call>
</tool_calls>
```

Нужно ещё что-то?"""
        tcs = parse_tool_calls(text)
        assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
        assert tcs[0]["name"] == "ИМЯ_ИНСТРУМЕНТА", tcs
        assert json.loads(tcs[0]["arguments"]) == {"ПАРАМЕТР": "ЗНАЧЕНИЕ"}, tcs
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: top-level plain code fence with call XML executes")


def test_masking_disabled_parses_fenced_call():
    """При выключенной маске вызов из код-блока парсится (текущее экспериментальное поведение)."""
    assert server.MASK_CODE_FENCES is False
    text = """```
<tool_calls>
  <tool_call name="tool_describe">
    <parameter name="name">mcp__playwright__browser_navigate</parameter>
  </tool_call>
</tool_calls>
```"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
    assert tcs[0]["name"] == "tool_describe"
    print("  PASS: masking disabled -> fenced call parses")


def test_real_call_after_closed_fence():
    """Вызовы в ```-блоке верхнего уровня И сырой inline-вызов парсятся оба."""
    server.MASK_CODE_FENCES = True
    try:
        text = """Пример:

```
<tool_call name="FAKE">
<parameter name="x">1</parameter>
</tool_call>
```

А теперь по делу:

<tool_call name="search_files">
  <parameter name="path">C:\\Projects</parameter>
</tool_call>"""
        tcs = parse_tool_calls(text)
        assert len(tcs) == 2, f"expected 2 tool calls, got {len(tcs)}: {tcs}"
        assert [t["name"] for t in tcs] == ["FAKE", "search_files"], tcs
        assert json.loads(tcs[1]["arguments"]) == {"path": "C:\\Projects"}
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: fenced + inline calls both parsed (content rule)")


def test_unclosed_fence_masks_tail():
    """Незакрытый фенс маскирует весь хвост — вызовов нет."""
    server.MASK_CODE_FENCES = True
    try:
        text = "Смотри:\n\n```\n<tool_call name=\"fake\"><parameter name=\"p\">v</parameter>"
        tcs = parse_tool_calls(text)
        assert tcs == [], f"expected no tool calls, got {tcs}"
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: unclosed fence masks the tail (mask on)")


def test_zero_argument_calls():
    """Вызов без аргументов — валидный вызов (регресс: skills_list терялся, REQ-7b1f03000e)."""
    tcs = parse_tool_calls('<tool_calls>\n<tool_call name="skills_list">\n\n</tool_call>\n</tool_calls>')
    assert len(tcs) == 1 and tcs[0]["name"] == "skills_list", tcs
    assert json.loads(tcs[0]["arguments"]) == {}
    tcs = parse_tool_calls("<tool_call>\n<name>ping</name>\n</tool_call>")
    assert len(tcs) == 1 and tcs[0]["name"] == "ping", tcs
    print("  PASS: zero-argument calls parsed")


def test_single_inline_mention_breaks_nothing():
    """Одиночные упоминания tool_call в тексте не дают фантомных вызовов."""
    server.MASK_CODE_FENCES = True
    try:
        # 1. бэктикнутое одиночное упоминание открывающего тега
        tcs = parse_tool_calls("Оберните вызов в `<tool_call name=\"x\">` и дождитесь результата.")
        assert tcs == [], tcs

        # 2. бэктикнутый ПОЛНЫЙ пример с аргументами — тоже не вызов
        tcs = parse_tool_calls(
            "Формат такой: `<tool_call name=\"terminal\">"
            "<parameter name=\"command\">ls</parameter></tool_call>` — и всё.")
        assert tcs == [], tcs

        # 3. упоминание + реальный вызов в одном сообщении -> парсится только реальный
        msg = ("Пример: `<tool_call name=\"fake\"><parameter name=\"p\">v</parameter></tool_call>`.\n\n"
               "<tool_call name=\"skills_list\"></tool_call>")
        tcs = parse_tool_calls(msg)
        assert len(tcs) == 1 and tcs[0]["name"] == "skills_list", tcs

        # 4. mid-stream: маскированный контекст не матчится на инлайн-упоминании,
        #    но матчится на реальном вызове (смещения валидны)
        import re
        from server import _mask_code_fences
        buf = "Пример: `<tool_call name=\"fake\">`.\n\n<tool_call name=\"skills_list\"></tool_call>"
        ctx = _mask_code_fences(buf)
        m = re.search(r'<(?:invoke|tool_calls?)[\s>]', ctx)
        assert m is not None
        assert ctx[m.start():].startswith("<tool_call name=\"skills_list\""), ctx[m.start():m.start()+40]
        assert len(ctx) == len(buf)
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: single inline mentions produce no phantoms, real call intact")


def test_no_tool_call_in_plain_text():
    tcs = parse_tool_calls("Просто ответ без вызовов инструментов.")
    assert tcs == [], f"expected no tool calls, got {tcs}"
    print("  PASS: plain text -> no tool calls")


def test_usr_format_parses():
    """Новый usr_-префиксный формат (актуальный tool_header)."""
    text = """<usr_tool_calls>
  <usr_tool_call name="web_search">
    <usr_parameter name="query">погода в санкт-петербурге</usr_parameter>
    <usr_parameter name="limit">5</usr_parameter>
  </usr_tool_call>
</usr_tool_calls>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1, f"expected 1 tool call, got {tcs}"
    assert tcs[0]["name"] == "web_search", tcs
    assert json.loads(tcs[0]["arguments"]) == {"query": "погода в санкт-петербурге", "limit": "5"}
    print("  PASS: usr_ format single call")


def test_usr_format_multiple_calls():
    text = """<usr_tool_calls>
  <usr_tool_call name="read_file">
    <usr_parameter name="path">C:\\x.txt</usr_parameter>
  </usr_tool_call>
  <usr_tool_call name="web_search">
    <usr_parameter name="query">hello</usr_parameter>
  </usr_tool_call>
</usr_tool_calls>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 2, f"expected 2 tool calls, got {tcs}"
    assert [t["name"] for t in tcs] == ["read_file", "web_search"]
    assert json.loads(tcs[0]["arguments"]) == {"path": "C:\\x.txt"}
    print("  PASS: usr_ format multiple calls")


def test_usr_format_no_wrapper():
    """Без <usr_tool_calls>-обёртки тоже парсится (как старый Format 9)."""
    text = """<usr_tool_call name="terminal">
  <usr_parameter name="command">ls -la</usr_parameter>
</usr_tool_call>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1 and tcs[0]["name"] == "terminal", tcs
    assert json.loads(tcs[0]["arguments"]) == {"command": "ls -la"}
    print("  PASS: usr_ format without wrapper")


def test_usr_format_zero_args():
    tcs = parse_tool_calls("<usr_tool_calls>\n<usr_tool_call name=\"skills_list\">\n\n</usr_tool_call>\n</usr_tool_calls>")
    assert len(tcs) == 1 and tcs[0]["name"] == "skills_list", tcs
    assert json.loads(tcs[0]["arguments"]) == {}
    print("  PASS: usr_ zero-argument call")


def test_usr_format_in_token_tool_header_rendered():
    """_tool_calls_to_xml рендерит историю tool_calls в usr_ формате."""
    from server import _tool_calls_to_xml
    xml = _tool_calls_to_xml([{
        "id": "call_1", "type": "function",
        "function": {"name": "web_search", "arguments": json.dumps({"query": "t", "limit": 3})}
    }])
    assert '<usr_tool_call name="web_search">' in xml, xml
    assert '<usr_parameter name="query">t</usr_parameter>' in xml, xml
    assert '</usr_parameter>' in xml and '</usr_tool_call>' in xml
    print("  PASS: _tool_calls_to_xml emits usr_ tags")


def test_usr_strip_tool_tags():
    """_strip_tool_tags вырезает usr_ теги вне код-блоков, внутри - сохраняет."""
    text = """до <usr_tool_call name="x"><usr_parameter name="p">1</usr_parameter></usr_tool_call> после
```
<usr_tool_calls>
  <usr_tool_call name="y"/>
</usr_tool_calls>
```"""
    out = _strip_tool_tags(text)
    assert '<usr_tool_call name="x">' not in out, out
    assert '<usr_tool_call name="y"/>' in out, "fenced usr_ markup must survive"
    print("  PASS: _strip_tool_tags handles usr_ tags")


def test_usr_format_in_plain_fence_executes():
    """Пример usr_ формата в ```-блоке верхнего уровня — реальный вызов."""
    server.MASK_CODE_FENCES = True
    try:
        text = """Вот формат:

```
<usr_tool_calls>
  <usr_tool_call name="web_search">
    <usr_parameter name="query">пример</usr_parameter>
  </usr_tool_call>
</usr_tool_calls>
```
"""
        tcs = parse_tool_calls(text)
        assert len(tcs) == 1, f"expected 1 call, got {len(tcs)}: {tcs}"
        assert tcs[0]["name"] == "web_search", tcs
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: usr_ example in plain top-level fence now executes")



def test_plain_xml_block_executed_log_case():
    """Регресс REQ-3538dd0009: вызов, обёрнутый моделью в ```xml...```,
    исполняется (2 вызова), а не уходит клиенту текстом."""
    server.MASK_CODE_FENCES = True
    try:
        text = """Хороший вопрос.

```xml
<usr_tool_calls>
  <usr_tool_call name="search_files">
    <usr_parameter name="pattern">Deferred tool catalog</usr_parameter>
    <usr_parameter name="path">C:/Users/keeper/AppData/Local/hermes</usr_parameter>
    <usr_parameter name="output_mode">files_only</usr_parameter>
    <usr_parameter name="limit">20</usr_parameter>
  </usr_tool_call>
  <usr_tool_call name="search_files">
    <usr_parameter name="pattern">*deferred*</usr_parameter>
    <usr_parameter name="path">C:/Users/keeper/AppData/Local/hermes</usr_parameter>
    <usr_parameter name="limit">30</usr_parameter>
  </usr_tool_call>
</usr_tool_calls>
```"""
        tcs = parse_tool_calls(text)
        assert len(tcs) == 2, f"expected 2 calls, got {len(tcs)}: {tcs}"
        assert [t["name"] for t in tcs] == ["search_files", "search_files"], tcs
        # дефолтная маска по-прежнему обезвреживает разметку в таком блоке
        out = _mask_code_fences(text)
        assert "usr_t00l" in out, out
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: ```xml-wrapped call executes (REQ-3538dd0009)")


def test_usr_inline_mention_defused():
    """Инлайн-упоминание `<usr_tool_call ...>` не даёт фантомного вызова."""
    server.MASK_CODE_FENCES = True
    try:
        tcs = parse_tool_calls("Оберните вызов в `<usr_tool_calls>` и `<usr_tool_call name=\"x\">`.")
        assert tcs == [], tcs
        msg = ("Пример: `<usr_tool_call name=\"fake\"><usr_parameter name=\"p\">v</usr_parameter></usr_tool_call>`.\n\n"
               "<usr_tool_call name=\"skills_list\"></usr_tool_call>")
        tcs = parse_tool_calls(msg)
        assert len(tcs) == 1 and tcs[0]["name"] == "skills_list", tcs
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: usr_ inline mentions no phantoms, real call intact")


def test_fix_tool_desc_rewrites_format_mentions():
    from server import _fix_tool_desc
    assert _fix_tool_desc("respond with <tool_call name='x'>") == "respond with <usr_tool_call name='x'>"
    assert _fix_tool_desc("wrap the call in <tool_calls>") == "wrap the call in <usr_tool_calls>"
    assert _fix_tool_desc("no tags here") == "no tags here"
    assert _fix_tool_desc("") == ""
    print("  PASS: _fix_tool_desc rewrites tool_call -> usr_tool_call")


def test_messages_to_prompt_rewrites_tool_descriptions():
    """Описания тулов от клиента с 'tool_call' не противоречат usr_ формату."""
    from server import messages_to_prompt
    tools = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Call <tool_call name=\"web_search\"> to search the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "query; use tool_call as described"}
                },
                "required": ["q"],
            },
        },
    }]
    out = messages_to_prompt(
        [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}],
        tools,
    )
    assert "<usr_tool_call name=\"web_search\">" in out, out
    assert "use usr_tool_call as described" in out, out
    assert "<tool_call" not in out, "no raw tool_call mentions may remain"
    print("  PASS: messages_to_prompt shows tool descriptions in usr_ format")


def test_spaced_tags_parsed():
    """deepseek-flash теперь шлёт теги с пробелом после '<' — вызов должен распознаваться."""
    text = """< calls>
< invoke name="skill_view">
< parameter name="name" string="true">local-firecrawl-setup</ parameter>
</ invoke>
</ calls>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
    assert tcs[0]["name"] == "skill_view", f"bogus name: {tcs}"
    args = json.loads(tcs[0]["arguments"])
    assert args == {"name": "local-firecrawl-setup"}, f"bad args: {args}"
    print("  PASS: spaced tags parsed")


def test_spaced_tags_stripped_from_client_text():
    """Те же теги с пробелами не должны утекать в текст ответа."""
    text = """Начинаю.

< calls>
< invoke name="skill_view">
< parameter name="name" string="true">local-firecrawl-setup</ parameter>
</ invoke>
</ calls>"""
    out = _strip_tool_tags(text)
    assert "<" not in out or "calls" not in out, f"tool markup leaked: {out!r}"
    assert "Начинаю." in out
    print("  PASS: spaced tags stripped from client text")


def test_calls_wrapper_region_no_residue():
    """Обёртка <calls> ... </calls> целиком входит в регион: в before/tail не
    остаётся <calls> / </calls> (регресс REQ-1ec8b60034 — остатки в Hermes)."""
    from server import _TOOL_OPEN_RE, _tool_region_end, _strip_vyzov_blocks
    text = """Отвечаю.

<calls>
<invoke name="skill_view">
<parameter name="name" string="true">web-fetch-diagnostics</parameter>
</invoke>
</calls>
"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 1 and tcs[0]["name"] == "skill_view", tcs
    m = _TOOL_OPEN_RE.search(text)
    assert m is not None and text.startswith("<calls>", m.start()), text[m.start():m.start() + 10]
    region_end = _tool_region_end(text, m.start())
    assert text[region_end - 8:region_end] == "</calls>", text[region_end - 20:region_end]
    before = _strip_tool_tags(text[:m.start()])
    tail = _strip_tool_tags(_strip_vyzov_blocks(text[region_end:])).lstrip("\n")
    assert before == "Отвечаю.\n\n", repr(before)
    assert tail == "", repr(tail)
    print("  PASS: <calls> wrapper fully inside region — no residue")


def test_strip_tool_tags_keeps_literal_mention():
    """Литеральное упоминание `<usr_tool_calls>` в прозе — НЕ вызов и НЕ вырезается.
    (Сообщено: «что вызывать надо через <usr_tool_calls>» превращалось в «через .»)."""
    out = _strip_tool_tags("что вызывать надо через <usr_tool_calls>")
    assert "<usr_tool_calls>" in out, f"literal mention must survive: {out!r}"
    out2 = _strip_tool_tags("а потом <usr_tool_call name='x'> и <parameter name='q'>")
    assert "<usr_tool_call" in out2 and "<parameter" in out2, out2
    # полная структура вызова по-прежнему вырезается
    out3 = _strip_tool_tags("до <usr_tool_calls><usr_tool_call name=\"x\"><usr_parameter name=\"p\">1</usr_parameter></usr_tool_call></usr_tool_calls> после")
    assert "<usr_tool" not in out3, out3
    assert "до" in out3 and "после" in out3, out3
    # полная структура в ```-фенсе сохраняется (примеры формата)
    out4 = _strip_tool_tags("```\n<usr_tool_calls>\n  <usr_tool_call name=\"y\"/>\n</usr_tool_calls>\n```")
    assert "<usr_tool_call name=\"y\"/>" in out4, out4
    print("  PASS: literal <usr_tool_calls> mention preserved; complete structures stripped")


def test_dsml_marker_glued_into_tags_scrubbed():
    """Остатки маркера ||DSML||, приклеенные к тегам, чистятся до распознавания."""
    from server import _DSML_GLUE_RE
    marker = "\uff5c\uff5cDSML\uff5c\uff5c"
    src = ((f"<{marker}invoke name=\"x\">"
            f"<{marker}parameter name=\"q\">v</{marker}parameter>"
            f"</{marker}invoke>"))
    out = _DSML_GLUE_RE.sub("", src)
    assert marker not in out and "\uff5c" not in out, f"marker leaked: {out!r}"
    assert out == '<invoke name="x"><parameter name="q">v</parameter></invoke>', out
    print("  PASS: glued DSML marker scrubbed from tags")


def test_usr_param_typo_closing_tag_does_not_swallow_xml():
    """Опечатка в закрытии </us_parameter> не должна пожирать остаток XML.

    Регрессия: модель написала `</us_parameter>` вместо `</usr_parameter>`;
    ленивый regex продолжается до следующего корректного закрытия и
    захватывает значение следующего параметра в предыдущий.
    """
    text = """<usr_tool_calls>
  <usr_tool_call name="vision_analyze">
    <usr_parameter name="image_url">C:\\users\\p1.png</usr_parameter>
    <usr_parameter name="question" string="true">q1</usr_parameter>
  </usr_tool_call>
  <usr_tool_call name="vision_analyze">
    <usr_parameter name="image_url" string="true">C:\\users\\p2.png</us_parameter>
    <usr_parameter name="question" string="true">q2</usr_parameter>
  </usr_tool_call>
</usr_tool_calls>"""
    tcs = parse_tool_calls(text)
    assert len(tcs) == 2, f"expected 2 calls, got {len(tcs)}: {tcs}"
    args0 = json.loads(tcs[0]["arguments"])
    assert args0["image_url"] == "C:\\users\\p1.png", args0
    assert args0["question"] == "q1", args0
    args1 = json.loads(tcs[1]["arguments"])
    assert args1["image_url"] == "C:\\users\\p2.png", f"image_url swallowed XML: {args1!r}"
    assert args1["question"] == "q2", f"question swallowed: {args1!r}"
    assert args1["image_url"] == "C:\\users\\p2.png", args1
    print("  PASS: typo in closing tag keeps parameter values clean")


def test_vyzov_block_calls_parsed():
    """Вызовы внутри блока ```Вызов ... ``` распознаются (tool_coll_header Шаг 3)."""
    server.MASK_CODE_FENCES = True
    try:
        src = """Посмотрю файлы.

```Вызов
<usr_tool_calls>
  <usr_tool_call name="search_files">
    <usr_parameter name="path">C:\\docs</usr_parameter>
    <usr_parameter name="pattern">*.md</usr_parameter>
  </usr_tool_call>
</usr_tool_calls>
```

Готово."""
        tcs = parse_tool_calls(src)
        assert len(tcs) == 1, f"expected 1 call, got {len(tcs)}: {tcs}"
        assert tcs[0]["name"] == "search_files", tcs
        assert json.loads(tcs[0]["arguments"]) == {"path": "C:\\docs", "pattern": "*.md"}, tcs
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: calls inside ```Вызов block parsed")


def test_vyzov_block_stripped_from_narrative():
    """Блок Вызов вырезается из клиентского текста целиком (фенсы + тело)."""
    from server import _strip_vyzov_blocks
    src = """Смотрю.

```Вызов
<usr_tool_calls>
  <usr_tool_call name="search_files"><usr_parameter name="p">v</usr_parameter></usr_tool_call>
</usr_tool_calls>
```

Дальше."""
    out = _strip_vyzov_blocks(src)
    assert "<usr_tool" not in out, out
    assert "Вызов" not in out, out
    assert "```" not in out, out
    assert "Смотрю." in out and "Дальше." in out, out
    print("  PASS: Вызов block fully stripped from narrative")


def test_vyzov_block_info_case_insensitive():
    """Контентное правило: блок === вызов по XML в теле, а не по инфо-строке."""
    from server import _vyzov_block_spans
    for tag in ("Вызов", "вызов", "ВЫЗОВ", "xml", "", "markdown"):
        src = "```%s\n<usr_tool_call name=\"x\"><usr_parameter name=\"a\">1</usr_parameter></usr_tool_call>\n```" % tag
        spans = _vyzov_block_spans(src)
        assert len(spans) == 1, (tag, spans)
    # без tool-тега — НЕ вызов
    assert _vyzov_block_spans("```xml\n<json>данные</json>\n```") == []
    # тильда-фенс с XML — НЕ вызов (только бэктики)
    assert _vyzov_block_spans("~~~\n<usr_tool_call name=\"x\"/>\n~~~") == []
    print("  PASS: content-based call blocks (info string irrelevant); tilde excluded")


def test_vyzov_block_untouched_by_mask():
    """keep_vyzov=True оставляет содержимое блока Вызов нетронутым; длина сохраняется."""
    from server import _mask_code_fences
    src = "```Вызов\n<tool_call name=\"x\">call</tool_call>\n```"
    out = _mask_code_fences(src, keep_vyzov=True)
    assert "<tool_call" in out and "<t00l_call" not in out, out
    assert len(out) == len(src), "mask must preserve length"
    # без keep_vyzov — по-прежнему маскируется
    out2 = _mask_code_fences(src)
    assert "<t00l_call" in out2, out2
    print("  PASS: keep_vyzov preserves Вызов block content")


def test_demo_double_escaped_vyzov_not_executed():
    """Демонстрация вызова с двойным экранированием (```` вокруг ```Вызов) —
    это НЕ вызов: вложенный блок верхнего уровня не существуют для парсера.
    """
    server.MASK_CODE_FENCES = True
    try:
        src = """Как это делается:

````markdown
```Вызов
<usr_tool_calls>
  <usr_tool_call name="search_files">
    <usr_parameter name="path">C:\\docs</usr_parameter>
  </usr_tool_call>
</usr_tool_calls>
```
````

Продолжение."""
        out = _mask_code_fences(src, keep_vyzov=True)
        assert len(out) == len(src), "mask must preserve length"
        assert "<usr_t00l" in out or "<t00l_" in out, f"demo must be defused: {out}"
        assert parse_tool_calls(src) == [], f"demo must not execute: {parse_tool_calls(src)}"
        # реальный (верхнего уровня) вызов ниже всё равно распознаётся
        real = src + "\n```Вызов\n<usr_tool_calls>\n  <usr_tool_call name=\"search_files\">\n    <usr_parameter name=\"path\">C:\\x</usr_parameter>\n  </usr_tool_call>\n</usr_tool_calls>\n```"
        r2 = parse_tool_calls(real)
        assert len(r2) == 1 and r2[0]["name"] == "search_files", r2
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: double-escaped demo not executed; top-level call still parsed")


def test_strip_vyzov_blocks_keeps_nested_demo():
    """_strip_vyzov_blocks не вырезает демо-блок (вложенный Вызов), но вырезает
    реальный блок верхнего уровня."""
    from server import _strip_vyzov_blocks
    out = _strip_vyzov_blocks("````\n```Вызов\nx\n```\n````")
    assert "Вызов" in out and "````" in out, out
    out2 = _strip_vyzov_blocks("```Вызов\n<usr_tool_calls>x</usr_tool_calls>\n```")
    assert "Вызов" not in out2, out2
    print("  PASS: strip_vyzov_blocks is nesting-aware")


def test_find_vyzov_open_top_level_only():
    """_find_vyzov_open находит начало первого блок-вызова верхнего уровня."""
    from server import _find_vyzov_open
    # двойное экранирование (демо) — не вызов
    assert _find_vyzov_open("````\n```Вызов\n<usr_tool_calls>\n  <usr_tool_call name=\"x\"><usr_parameter name=\"a\">1</usr_parameter></usr_tool_call>\n</usr_tool_calls>\n```\n````") is None, "nested demo skipped"
    # блок Вызов, вложенный в другой блок верхнего уровня — не вызов
    assert _find_vyzov_open("```xml\n```Вызов\n<usr_tool_call name=\"x\"/>\n```\n```") is None, "nested in xml block skipped"
    # верхний уровень: блок Вызов с XML — позиция его фенса
    src = "пред.\n\n```Вызов\n<usr_tool_calls>\n  <usr_tool_call name=\"x\"><usr_parameter name=\"a\">1</usr_parameter></usr_tool_call>\n</usr_tool_calls>\n```"
    pos = _find_vyzov_open(src)
    assert pos == src.find("```Вызов"), (pos, src.find("```Вызов"))
    # верхний уровень: простой ```xml блок с XML — тоже позиция фенса
    src2 = "```xml\n<usr_tool_calls>\n  <usr_tool_call name=\"y\"><usr_parameter name=\"b\">2</usr_parameter></usr_tool_call>\n</usr_tool_calls>\n```"
    pos2 = _find_vyzov_open(src2)
    assert pos2 == src2.find("```xml"), pos2
    # верхний уровень: пустой ``` с XML — позиция фенса
    src3 = "до\n```\n<usr_tool_calls><usr_tool_call name=\"z\"><usr_parameter name=\"c\">3</usr_parameter></usr_tool_call></usr_tool_calls>\n```"
    pos3 = _find_vyzov_open(src3)
    assert pos3 == src3.find("```\n<usr_tool_calls"), pos3
    # без вызовов — None
    assert _find_vyzov_open("вот блок ```xml\n<json>данные</json>\n```") is None
    print("  PASS: _find_vyzov_open is top-level content-based")


def test_messages_to_prompt_blank_line_between_tools():
    """Пустая строка между описаниями тулов в списке."""
    from server import messages_to_prompt
    tools = [
        {"type": "function", "function": {
            "name": "t1", "description": "one",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {
            "name": "t2", "description": "two",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
    ]
    out = messages_to_prompt(
        [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}],
        tools,
    )
    i1 = out.find("  - t1: one")
    i2 = out.find("  - t2: two")
    assert i1 != -1 and i2 != -1, out
    assert "\n\n" in out[i1 + len("  - t1: one"):i2], out[i1:i2]
    print("  PASS: blank line between tool descriptions")


if __name__ == "__main__":
    tests = [
        test_nested_quadruple_fence_defused,
        test_midline_backtick_run_in_param_value,
        test_tilde_fences_defused,
        test_strip_tool_tags_keeps_code_fences,
        test_strip_tool_tags_keeps_tilde_fences,
        test_strip_tool_tags_keeps_inline_code_spans,
        test_mask_defuses_tool_word_in_fences,
        test_nested_format_single,
        test_nested_format_multiple,
        test_nested_format_cyrillic_params,
        test_nested_format_multiline_value,
        test_nested_format_no_arguments_wrapper,
        test_json_inside_arguments,
        test_wrapper_with_direct_tool_tags,
        test_wrapper_multiple_direct_tags,
        test_zero_argument_calls,
        test_single_inline_mention_breaks_nothing,
        test_plain_code_fence_with_call_xml_executes,
        test_masking_disabled_parses_fenced_call,
        test_real_call_after_closed_fence,
        test_unclosed_fence_masks_tail,
        test_regression_hermes_format9,
        test_regression_invoke_format1,
        test_extra_attributes_tolerated,
        test_no_tool_call_in_plain_text,
        test_usr_format_parses,
        test_usr_format_multiple_calls,
        test_usr_format_no_wrapper,
        test_usr_format_zero_args,
        test_usr_format_in_token_tool_header_rendered,
        test_usr_strip_tool_tags,
        test_usr_format_in_plain_fence_executes,
        test_plain_xml_block_executed_log_case,
        test_usr_inline_mention_defused,
        test_fix_tool_desc_rewrites_format_mentions,
        test_messages_to_prompt_rewrites_tool_descriptions,
        test_spaced_tags_parsed,
        test_spaced_tags_stripped_from_client_text,
        test_calls_wrapper_region_no_residue,
        test_strip_tool_tags_keeps_literal_mention,
        test_dsml_marker_glued_into_tags_scrubbed,
        test_usr_param_typo_closing_tag_does_not_swallow_xml,
        test_vyzov_block_calls_parsed,
        test_vyzov_block_stripped_from_narrative,
        test_vyzov_block_info_case_insensitive,
        test_vyzov_block_untouched_by_mask,
        test_demo_double_escaped_vyzov_not_executed,
        test_strip_vyzov_blocks_keeps_nested_demo,
        test_find_vyzov_open_top_level_only,
        test_messages_to_prompt_blank_line_between_tools,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            sys.exit(1)
    print("All tests passed.")

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


def test_example_in_code_fence_ignored():
    """Пример формата внутри ```-блока — не реальный вызов (при включённой маске)."""
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
        assert tcs == [], f"expected no tool calls, got {tcs}"
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: example inside code fence ignored (mask on)")


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
    """Реальный вызов после закрытого код-блока с примером парсится."""
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
        assert len(tcs) == 1, f"expected 1 tool call, got {len(tcs)}: {tcs}"
        assert tcs[0]["name"] == "search_files"
        assert json.loads(tcs[0]["arguments"]) == {"path": "C:\\Projects"}
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: real call after closed fence parsed (mask on)")


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


def test_usr_format_in_fence_defused():
    """Пример usr_ формата внутри ```-блока — не реальный вызов (mask on)."""
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
        assert tcs == [], f"expected no tool calls, got {tcs}"
    finally:
        server.MASK_CODE_FENCES = False
    print("  PASS: usr_ example inside fence ignored (mask on)")


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
        test_example_in_code_fence_ignored,
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
        test_usr_format_in_fence_defused,
        test_usr_inline_mention_defused,
        test_fix_tool_desc_rewrites_format_mentions,
        test_messages_to_prompt_rewrites_tool_descriptions,
        test_spaced_tags_parsed,
        test_spaced_tags_stripped_from_client_text,
        test_dsml_marker_glued_into_tags_scrubbed,
        test_usr_param_typo_closing_tag_does_not_swallow_xml,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            sys.exit(1)
    print("All tests passed.")

"""Test parse_tool_calls: new nested-element format + regression of old formats."""
import json
import sys
sys.path.insert(0, ".")

import server
from server import parse_tool_calls, _strip_tool_tags


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


def test_no_tool_call_in_plain_text():
    tcs = parse_tool_calls("Просто ответ без вызовов инструментов.")
    assert tcs == [], f"expected no tool calls, got {tcs}"
    print("  PASS: plain text -> no tool calls")


if __name__ == "__main__":
    tests = [
        test_strip_tool_tags_keeps_code_fences,
        test_strip_tool_tags_keeps_inline_code_spans,
        test_nested_format_single,
        test_nested_format_multiple,
        test_nested_format_cyrillic_params,
        test_nested_format_multiline_value,
        test_nested_format_no_arguments_wrapper,
        test_json_inside_arguments,
        test_wrapper_with_direct_tool_tags,
        test_wrapper_multiple_direct_tags,
        test_example_in_code_fence_ignored,
        test_masking_disabled_parses_fenced_call,
        test_real_call_after_closed_fence,
        test_unclosed_fence_masks_tail,
        test_regression_hermes_format9,
        test_regression_invoke_format1,
        test_no_tool_call_in_plain_text,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            sys.exit(1)
    print("All tests passed.")

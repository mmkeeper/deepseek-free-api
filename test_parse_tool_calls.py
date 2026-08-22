"""Test parse_tool_calls: new nested-element format + regression of old formats."""
import json
import sys
sys.path.insert(0, ".")

from server import parse_tool_calls


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


def test_no_tool_call_in_plain_text():
    tcs = parse_tool_calls("Просто ответ без вызовов инструментов.")
    assert tcs == [], f"expected no tool calls, got {tcs}"
    print("  PASS: plain text -> no tool calls")


if __name__ == "__main__":
    tests = [
        test_nested_format_single,
        test_nested_format_multiple,
        test_nested_format_cyrillic_params,
        test_nested_format_multiline_value,
        test_nested_format_no_arguments_wrapper,
        test_json_inside_arguments,
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

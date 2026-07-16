"""Test WASM-based PoW solver with locally cached WASM file."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import struct
import ctypes
from src.pow import DeepSeekHash

_WASM_PATH = os.path.join(os.path.dirname(__file__), "data", "sha3_wasm.wasm")
_TEST_WASM_URL = "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"


def _load_wasm():
    if not os.path.exists(_WASM_PATH):
        raise FileNotFoundError(
            f"WASM not found at {_WASM_PATH}. "
            f"Download from {_TEST_WASM_URL} and save to {_WASM_PATH}"
        )
    with open(_WASM_PATH, "rb") as f:
        return f.read()


def test_wasm_file_exists():
    assert os.path.exists(_WASM_PATH), f"WASM file missing at {_WASM_PATH}"


def test_solver_creates_instance():
    solver = DeepSeekHash.from_bytes(_load_wasm())
    assert solver is not None


def test_exports_exist():
    solver = DeepSeekHash.from_bytes(_load_wasm())
    assert solver._memory is not None, "memory export missing"
    assert solver._alloc is not None, "__wbindgen_export_0 missing"
    assert solver._stack_grow is not None, "__wbindgen_add_to_stack_pointer missing"
    assert solver._wasm_solve is not None, "wasm_solve missing"


def test_memory_read_write():
    solver = DeepSeekHash.from_bytes(_load_wasm())
    ptr, length = solver._write_string("test")
    assert length == 4

    buf = bytes(solver._memory_buf())
    written = buf[ptr:ptr + length]
    assert written == b"test", f"expected b'test', got {written}"


def test_rejects_unsupported_algorithm():
    solver = DeepSeekHash.from_bytes(_load_wasm())
    try:
        solver.calculate_hash("BAD", "", "", 0, 0)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "Unsupported algorithm" in str(e)


def test_solve_fails_with_bad_challenge():
    solver = DeepSeekHash.from_bytes(_load_wasm())
    try:
        solver.calculate_hash("DeepSeekHashV1", "", "", 999999, 0)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "status 0" in str(e)


def test_solve_with_empty_strings():
    solver = DeepSeekHash.from_bytes(_load_wasm())
    try:
        solver.calculate_hash("DeepSeekHashV1", "", "", 999999, 0)
    except ValueError as e:
        assert "status 0" in str(e)


def test_double_instantiation():
    """Creating a second instance should not interfere with the first."""
    solver1 = DeepSeekHash.from_bytes(_load_wasm())
    solver2 = DeepSeekHash.from_bytes(_load_wasm())
    assert solver1 is not solver2

    p1, l1 = solver1._write_string("aaa")
    p2, l2 = solver2._write_string("bbb")
    assert l1 == 3
    assert l2 == 3

    buf1 = bytes(solver1._memory_buf())
    buf2 = bytes(solver2._memory_buf())
    assert buf1[p1:p1+3] == b"aaa"
    assert buf2[p2:p2+3] == b"bbb"


if __name__ == "__main__":
    tests = [
        test_wasm_file_exists,
        test_solver_creates_instance,
        test_exports_exist,
        test_memory_read_write,
        test_rejects_unsupported_algorithm,
        test_solve_fails_with_bad_challenge,
        test_solve_with_empty_strings,
        test_double_instantiation,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {type(e).__name__}: {e}")
            sys.exit(1)
    print("All tests passed.")

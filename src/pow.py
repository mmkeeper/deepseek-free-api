from __future__ import annotations

import json
import struct
import ctypes
import os
from pathlib import Path

from .config import DEEPSEEK_SHA3_WASM
from .proxy import get_http_client

_wasm_solver = None
_WASM_CACHE = Path(__file__).resolve().parent.parent / "data" / "sha3_wasm.wasm"


class DeepSeekHash:
    """DeepSeekHash using wasmtime runtime."""

    def __init__(self, store, instance):
        self._store = store
        self._instance = instance
        exports = instance.exports(store)
        self._memory = exports["memory"]
        self._alloc = exports["__wbindgen_export_0"]
        self._stack_grow = exports["__wbindgen_add_to_stack_pointer"]
        self._wasm_solve = exports["wasm_solve"]

    def _memory_buf(self):
        ptr = self._memory.data_ptr(self._store)
        size = self._memory.data_len(self._store)
        return (ctypes.c_uint8 * size).from_address(ctypes.addressof(ptr.contents))

    def _write_string(self, text: str) -> tuple[int, int]:
        data = text.encode("utf-8")
        length = len(data)
        ptr = self._alloc(self._store, length, 1)
        buf = self._memory_buf()
        for i, b in enumerate(data):
            buf[ptr + i] = b
        return ptr, length

    def calculate_hash(
        self, algorithm: str, challenge: str, salt: str, difficulty: int, expire_at: int
    ) -> int:
        if algorithm != "DeepSeekHashV1":
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        prefix = f"{salt}_{expire_at}_"

        retptr = self._stack_grow(self._store, -16)
        try:
            ptr0, len0 = self._write_string(challenge)
            ptr1, len1 = self._write_string(prefix)

            self._wasm_solve(self._store, retptr, ptr0, len0, ptr1, len1, float(difficulty))

            buf = bytes(self._memory_buf())
            status = struct.unpack_from("<i", buf, retptr)[0]
            value_bytes = buf[retptr + 8: retptr + 16]
            value = struct.unpack("<d", value_bytes)[0]

            if status == 0:
                raise ValueError("PoW solver returned status 0 (failure)")
            return int(value)
        finally:
            self._stack_grow(self._store, 16)

    @classmethod
    def from_bytes(cls, wasm_bytes: bytes) -> "DeepSeekHash":
        from wasmtime import Engine, Store, Module, Instance

        engine = Engine()
        store = Store(engine)
        module = Module(engine, wasm_bytes)
        instance = Instance(store, module, [])
        return cls(store, instance)

    @classmethod
    async def create(cls, wasm_url: str) -> "DeepSeekHash":
        client = get_http_client()
        resp = await client.get(wasm_url)
        resp.raise_for_status()
        return cls.from_bytes(resp.content)


async def get_wasm_solver():
    global _wasm_solver
    if _wasm_solver is None:
        if _WASM_CACHE.exists():
            _wasm_solver = DeepSeekHash.from_bytes(_WASM_CACHE.read_bytes())
        else:
            _wasm_solver = await DeepSeekHash.create(DEEPSEEK_SHA3_WASM)
    return _wasm_solver


async def solve_pow(challenge: dict) -> int:
    algorithm = challenge.get("algorithm", "")
    if algorithm != "DeepSeekHashV1":
        raise ValueError(f"Unsupported PoW algorithm: {algorithm}")

    expire_at = challenge.get("expire_at") or challenge.get("expireAt")
    if expire_at is None:
        raise ValueError("PoW challenge is missing expire_at.")

    solver = await get_wasm_solver()
    answer = solver.calculate_hash(
        algorithm,
        challenge["challenge"],
        challenge["salt"],
        int(challenge["difficulty"]),
        int(expire_at),
    )
    return answer

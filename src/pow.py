from __future__ import annotations

import json
import struct
from io import BytesIO

from .config import DEEPSEEK_SHA3_WASM
from .proxy import get_http_client

_wasm_solver = None


class DeepSeekHash:
    def __init__(self, instance):
        self.instance = instance
        self.exports = instance.exports
        self.memory = self.exports.memory
        self._offset = 0

    @classmethod
    async def create(cls, wasm_url: str) -> "DeepSeekHash":
        from wasmer import Engine, Instance, Module, Store

        client = get_http_client()
        resp = await client.get(wasm_url)
        resp.raise_for_status()

        engine = Engine()
        store = Store(engine)
        module = Module(store, resp.content)
        instance = Instance(store, module)
        return cls(instance)

    def _write_string(self, text: str) -> tuple[int, int]:
        data = text.encode("utf-8")
        length = len(data)
        ptr = self.exports.__wbindgen_export_0(length, 1)
        mem = self.memory.uint8_view()
        mem[ptr : ptr + length] = data
        return ptr, length

    def calculate_hash(
        self, algorithm: str, challenge: str, salt: str, difficulty: int, expire_at: int
    ) -> int:
        if algorithm != "DeepSeekHashV1":
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        prefix = f"{salt}_{expire_at}_"

        retptr = self.exports.__wbindgen_add_to_stack_pointer(-16)
        try:
            ptr0, len0 = self._write_string(challenge)
            ptr1, len1 = self._write_string(prefix)

            self.exports.wasm_solve(retptr, ptr0, len0, ptr1, len1, difficulty)

            buf = self.memory.buffer
            status = struct.unpack_from("<i", buf, retptr)[0]
            value_bytes = buf[retptr + 8 : retptr + 16]
            value = struct.unpack("<d", value_bytes)[0]

            if status == 0:
                raise ValueError("PoW solver returned status 0 (failure)")
            return int(value)
        finally:
            self.exports.__wbindgen_add_to_stack_pointer(16)


async def get_wasm_solver() -> DeepSeekHash:
    global _wasm_solver
    if _wasm_solver is None:
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

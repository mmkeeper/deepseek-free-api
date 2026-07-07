import os
from pathlib import Path

BASE_URL = "https://chat.deepseek.com"
APP_VERSION = "2.0.0"
COMPLETION_PATH = "/api/v0/chat/completion"
DEEPSEEK_SHA3_WASM = (
    "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"
)

AUTH_DIR = Path.home() / ".deepseek-free-api"
AUTH_FILE = AUTH_DIR / "auth.json"
BROWSER_PROFILE = str(AUTH_DIR / "browser-profile")


def get_socks5_proxy() -> str:
    return os.environ.get("SOCKS5_PROXY", "")

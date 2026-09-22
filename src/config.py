import os
from pathlib import Path

BASE_URL = "https://chat.deepseek.com"
APP_VERSION = "2.0.0"
COMPLETION_PATH = "/api/v0/chat/completion"
STOP_STREAM_PATH = "/api/v0/chat/stop_stream"
DEEPSEEK_SHA3_WASM = (
    "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"
)

AUTH_DIR = Path.home() / ".deepseek-free-api"
AUTH_FILE = AUTH_DIR / "auth.json"
BROWSER_PROFILE = str(AUTH_DIR / "browser-profile")

TTS_PATH = "/api/v0/chat/tts"
VOICES_PATH = "/api/v0/chat/tts/voices"
VOICE_PATH = "/api/v0/chat/tts/voice"
TICKET_PATH = "/api/v0/auth/ticket"
TTS_CACHE_DIR = AUTH_DIR / "tts"


def get_socks5_proxy() -> str:
    return os.environ.get("SOCKS5_PROXY", "")

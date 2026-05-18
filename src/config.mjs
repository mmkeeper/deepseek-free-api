import os from "node:os";
import path from "node:path";

export const BASE_URL = "https://chat.deepseek.com";
export const APP_VERSION = "2.0.0";
export const COMPLETION_PATH = "/api/v0/chat/completion";
export const DEEPSEEK_SHA3_WASM =
  "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm";

export const AUTH_DIR = path.join(os.homedir(), ".deepseek-free-api");
export const AUTH_FILE = path.join(AUTH_DIR, "auth.json");
export const BROWSER_PROFILE = path.join(AUTH_DIR, "browser-profile");

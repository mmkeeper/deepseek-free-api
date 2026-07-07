from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from .config import AUTH_FILE, BASE_URL, BROWSER_PROFILE, get_socks5_proxy


# ─── Auth persistence ────────────────────────────────────────


def normalize_token(input_token: str | None) -> str:
    token = str(input_token or "").strip()
    if not token:
        return ""
    try:
        parsed = json.loads(token)
        if isinstance(parsed, str):
            return parsed.strip()
        if isinstance(parsed, dict) and isinstance(parsed.get("value"), str):
            return parsed["value"].strip()
    except (json.JSONDecodeError, ValueError):
        pass
    return token


def cookie_header_from_array(cookies: list[dict]) -> str:
    usable = [c for c in cookies if c.get("name") and "value" in c]
    if not any(c["name"] == "ds_session_id" for c in usable):
        raise ValueError("Cookie file does not contain ds_session_id.")
    return "; ".join(f"{c['name']}={c['value']}" for c in usable)


def read_saved_auth() -> dict | None:
    if not AUTH_FILE.exists():
        return None
    data = json.loads(AUTH_FILE.read_text("utf-8"))
    if not isinstance(data, dict):
        return None
    token = normalize_token(data.get("userToken") or data.get("token") or "")
    cookie_header = cookie_header_from_array(data.get("cookies") or [])
    return {"token": token, "cookieHeader": cookie_header}


def write_saved_auth(cookies: list[dict], user_token: str):
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(
        json.dumps(
            {
                "version": 1,
                "savedAt": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "baseUrl": BASE_URL,
                "profileDir": BROWSER_PROFILE,
                "userToken": user_token,
                "cookies": cookies,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    try:
        AUTH_FILE.chmod(0o600)
    except OSError:
        pass


# ─── Browser login (Playwright) ──────────────────────────────


def _parse_proxy_for_playwright(proxy_str: str) -> dict:
    s = proxy_str.replace("socks5://", "").replace("socks://", "")
    username = None
    password = None
    host_port = s

    at_idx = s.rfind("@")
    if at_idx != -1:
        auth_str = s[:at_idx]
        host_port = s[at_idx + 1 :]
        colon_idx = auth_str.find(":")
        if colon_idx != -1:
            from urllib.parse import unquote

            username = unquote(auth_str[:colon_idx])
            password = unquote(auth_str[colon_idx + 1 :])

    result: dict = {"server": f"socks5://{host_port}"}
    if username:
        result["username"] = username
    if password:
        result["password"] = password
    return result


async def _launch_persistent_context(chromium, headless: bool):
    from pathlib import Path as _P

    profile = _P(BROWSER_PROFILE)
    profile.mkdir(parents=True, exist_ok=True)

    launch_opts: dict = {
        "headless": headless,
        "viewport": None,
        "args": ["--disable-blink-features=AutomationControlled"],
    }

    proxy = get_socks5_proxy()
    if proxy:
        launch_opts["proxy"] = _parse_proxy_for_playwright(proxy)

    async def try_launch():
        try:
            return await chromium.launch_persistent_context(
                str(profile), **launch_opts, channel="chrome"
            )
        except Exception:
            return await chromium.launch_persistent_context(
                str(profile), **launch_opts
            )

    try:
        return await try_launch()
    except Exception as e:
        msg = str(e)
        if "SingletonLock" in msg:
            for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
                try:
                    (profile / name).unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                return await try_launch()
            except Exception as e2:
                raise RuntimeError(f"Не удалось открыть браузер: {e2}") from e2
        raise RuntimeError(
            f"Не удалось открыть браузер. Установи Google Chrome или выполни "
            f"'playwright install chromium'. {e}"
        ) from e


async def _wait_for_auth(context, timeout_s: int = 300, settle_s: float = 0.8):
    loop = asyncio.get_event_loop()
    done = asyncio.Event()
    result = {}

    def handler(response):
        if done.is_set():
            return
        try:
            url = response.url
            if "/api/v0/" not in url:
                return
            if response.status != 200:
                return
            req_headers = response.request.headers
            auth_hdr = req_headers.get("authorization") or req_headers.get("Authorization")
            if not auth_hdr or not auth_hdr.startswith("Bearer ") or len(auth_hdr) < 20:
                return
            body = response.json()
            if body and body.get("code") is not None and body["code"] != 0:
                return
            result["ok"] = True
            done.set()
        except Exception:
            pass

    context.on("response", handler)

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        context.remove_listener("response", handler)
        raise TimeoutError(
            f"Таймаут входа {timeout_s}с. Залогинься в окне DeepSeek."
        )

    context.remove_listener("response", handler)
    await asyncio.sleep(settle_s)


async def login_and_save_auth() -> dict:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(BASE_URL, wait_until="domcontentloaded")

        print(
            "\nОткроется окно DeepSeek. Залогинься там любым способом.\n"
            "   Окно закроется автоматически после успешного входа.\n"
        )

        await _wait_for_auth(context)

        cookies = await context.cookies(BASE_URL)
        raw_token = None
        try:
            raw_token = await page.evaluate("() => localStorage.getItem('userToken')")
        except Exception:
            pass

        token = normalize_token(raw_token or "")
        has_session = any(c["name"] == "ds_session_id" for c in cookies)
        if not token:
            await context.close()
            raise RuntimeError("В localStorage нет userToken. Логин не завершён.")
        if not has_session:
            await context.close()
            raise RuntimeError("В куках нет ds_session_id. Логин не завершён.")

        write_saved_auth(cookies, raw_token or "")
        await context.close()
        print("Авторизация DeepSeek сохранена!\n")
        return {"token": token, "cookieHeader": cookie_header_from_array(cookies)}


async def connect_to_running_chrome(cdp_port: int = 9222) -> dict:
    from playwright.async_api import async_playwright

    cdp_url = f"http://127.0.0.1:{cdp_port}"
    print(f"\nПодключаюсь к Chrome на {cdp_url}...")

    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.connect_over_cdp(cdp_url)
        except Exception:
            raise RuntimeError(
                f"Не удалось подключиться к Chrome.\n"
                f"1. Закрой весь Chrome\n"
                f"2. Запусти его заново:\n"
                f"   google-chrome --remote-debugging-port={cdp_port}\n"
                f"   (Windows: chrome.exe --remote-debugging-port={cdp_port})\n"
                f"   (macOS: open -a 'Google Chrome' --args --remote-debugging-port={cdp_port})\n"
            )

        page = None
        for ctx in browser.contexts:
            for p in ctx.pages:
                if p.url.startswith(BASE_URL):
                    page = p
                    break
            if page:
                break

        if not page:
            ctx = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = await ctx.new_page()

        await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=15000)

        raw_token = None
        try:
            raw_token = await page.evaluate("() => localStorage.getItem('userToken')")
        except Exception:
            pass

        token = normalize_token(raw_token or "")
        cookies = await browser.contexts[0].cookies(BASE_URL)
        has_session = any(c["name"] == "ds_session_id" for c in cookies)

        if not token:
            print("Ты не залогинен в DeepSeek. Открываю страницу — зайди в аккаунт.")
            await page.goto(f"{BASE_URL}/", wait_until="domcontentloaded")
            print("   После входа нажми Enter в этом терминале.")
            await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)

            raw_token2 = None
            try:
                raw_token2 = await page.evaluate(
                    "() => localStorage.getItem('userToken')"
                )
            except Exception:
                pass
            cookies2 = await browser.contexts[0].cookies(BASE_URL)
            if not raw_token2 or not any(
                c["name"] == "ds_session_id" for c in cookies2
            ):
                await browser.close()
                raise RuntimeError("Логин не подтверждён. Нет userToken или ds_session_id.")

            write_saved_auth(cookies2, raw_token2)
            await browser.close()
            print("Сессия из твоего Chrome сохранена!\n")
            return {
                "token": normalize_token(raw_token2),
                "cookieHeader": cookie_header_from_array(cookies2),
            }

        write_saved_auth(cookies, raw_token or "")
        await browser.close()
        print("Сессия DeepSeek из твоего Chrome сохранена!\n")
        return {"token": token, "cookieHeader": cookie_header_from_array(cookies)}


# ─── Manual import ───────────────────────────────────────────


def import_cookies(cookies_file_path: str, user_token_str: str) -> dict:
    p = Path(cookies_file_path)
    if not p.exists():
        raise FileNotFoundError(f"Файл cookies не найден: {cookies_file_path}")

    cookies = json.loads(p.read_text("utf-8"))

    if not isinstance(cookies, list):
        if isinstance(cookies, dict) and isinstance(cookies.get("cookies"), list):
            cookies = cookies["cookies"]
        else:
            raise ValueError("Файл cookies должен быть массивом JSON-объектов.")

    token = normalize_token(user_token_str)
    if not token:
        raise ValueError("Токен не может быть пустым.")

    usable = [c for c in cookies if c.get("name") and "value" in c]
    if not any(c["name"] == "ds_session_id" for c in usable):
        raise ValueError(
            "В файле cookies нет ds_session_id. "
            "Убедись, что экспортировал куки с chat.deepseek.com"
        )

    cookie_header = cookie_header_from_array(usable)
    write_saved_auth(usable, token)
    print("Cookies и токен импортированы!\n")
    return {"token": token, "cookieHeader": cookie_header}


def print_manual_instructions():
    print(
        """
=============================================
  Ручной экспорт сессии DeepSeek
=============================================

  1. Открой Chrome и зайди на https://chat.deepseek.com
  2. Убедись что ты залогинен (должен быть интерфейс чата)
  3. Открой DevTools (F12 или Ctrl+Shift+I)
  4. Перейди на вкладку Application -> Local Storage
     -> https://chat.deepseek.com
  5. Найди ключ "userToken" и скопируй его значение целиком
  6. Перейди на вкладку Application -> Cookies
     -> https://chat.deepseek.com
  7. Экспортируй все куки в файл (или скопируй вручную)

  Формат cookies JSON:
  [
    {"name": "ds_session_id", "value": "...", "domain": "chat.deepseek.com", ...},
    {"name": "...", "value": "...", ...}
  ]

  Сохрани файл и выполни:
    python server.py --import cookies.json "<userToken>"
=============================================
"""
    )

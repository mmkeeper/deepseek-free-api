from .config import APP_VERSION, BASE_URL


def base_headers(cookie_header: str, token: str) -> dict:
    from datetime import datetime, timezone

    tz_offset = -int(
        datetime.now(timezone.utc).astimezone().utcoffset().total_seconds()
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Content-Type": "application/json",
        "Origin": BASE_URL,
        "Referer": f"{BASE_URL}/",
        "Cookie": cookie_header,
        "X-App-Version": APP_VERSION,
        "x-client-platform": "web",
        "x-client-version": APP_VERSION,
        "x-client-locale": "ru",
        "x-client-timezone-offset": str(tz_offset),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

from __future__ import annotations

from .config import get_socks5_proxy

_client = None


def _normalize_url(proxy_str: str) -> str:
    s = proxy_str.strip()
    if s.startswith("socks5://") or s.startswith("socks://"):
        return s
    return f"socks5://{s}"


def get_http_client():
    """Return a shared httpx.AsyncClient with optional SOCKS5 proxy."""
    global _client
    if _client is not None:
        return _client

    import httpx

    proxy = get_socks5_proxy()
    proxy_url = _normalize_url(proxy) if proxy else None
    _client = httpx.AsyncClient(proxy=proxy_url, timeout=60, follow_redirects=True)
    return _client


def get_proxy_info() -> dict | None:
    proxy = get_socks5_proxy()
    if not proxy:
        return None
    from urllib.parse import urlparse

    try:
        url = urlparse(_normalize_url(proxy))
        return {
            "host": url.hostname or "",
            "port": url.port or 0,
            "hasAuth": bool(url.username or url.password),
        }
    except Exception:
        return {"host": proxy, "port": 0, "hasAuth": False}

import { Socks5ProxyAgent } from "undici";
import { getSocks5Proxy } from "./config.mjs";

let cachedDispatcher = null;

function normalizeProxyUrl(proxyStr) {
  const trimmed = proxyStr.trim();
  if (/^socks5?:\/\//.test(trimmed)) return trimmed;
  return `socks5://${trimmed}`;
}

export function getProxyDispatcher() {
  const socks5Proxy = getSocks5Proxy();
  if (!socks5Proxy) return undefined;
  if (cachedDispatcher) return cachedDispatcher;

  const url = normalizeProxyUrl(socks5Proxy);
  cachedDispatcher = new Socks5ProxyAgent(url);
  return cachedDispatcher;
}

export function proxyFetchOptions(options = {}) {
  const dispatcher = getProxyDispatcher();
  if (!dispatcher) return options;
  return { ...options, dispatcher };
}

export async function proxyFetch(url, options = {}) {
  return fetch(url, proxyFetchOptions(options));
}

export function getProxyInfo() {
  const socks5Proxy = getSocks5Proxy();
  if (!socks5Proxy) return null;
  try {
    const url = new URL(normalizeProxyUrl(socks5Proxy));
    return {
      host: url.hostname,
      port: parseInt(url.port, 10),
      hasAuth: !!(url.username || url.password),
    };
  } catch {
    return { host: socks5Proxy, port: 0, hasAuth: false };
  }
}

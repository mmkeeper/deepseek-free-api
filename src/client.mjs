import { BASE_URL, COMPLETION_PATH } from "./config.mjs";
import { baseHeaders } from "./headers.mjs";
import { solvePow } from "./pow.mjs";
import { streamSse } from "./sse.mjs";

export class DeepSeekClient {
  constructor({ cookieHeader, token, debug = false }) {
    this.cookieHeader = cookieHeader;
    this.token = token;
    this.debug = debug;
  }

  _buildHeaders() {
    return baseHeaders(this.cookieHeader, this.token);
  }

  async _request(path, { method = "GET", body } = {}) {
    const res = await fetch(`${BASE_URL}${path}`, {
      method,
      headers: this._buildHeaders(),
      body: body === undefined ? undefined : JSON.stringify(body),
    });

    const text = await res.text();
    let json;
    try { json = JSON.parse(text); } catch {
      if (res.status === 401 || res.status === 403) {
        const err = new Error(`Auth required: HTTP ${res.status}`);
        err.isAuthError = true;
        throw err;
      }
      throw new Error(`Expected JSON from ${path}, got HTTP ${res.status}: ${text.slice(0, 180)}`);
    }

    if (res.status === 401 || res.status === 403 || (json && (json.code === 40002 || json.code === 40003))) {
      const err = new Error(`Auth required: code ${json?.code ?? ""}`);
      err.isAuthError = true;
      throw err;
    }

    if (!res.ok || (json.code !== undefined && json.code !== 0)) {
      throw new Error(`DeepSeek API error at ${path}: HTTP ${res.status}, code ${json.code}, msg ${json.msg || ""}`);
    }

    return json;
  }

  async createSession() {
    const json = await this._request("/api/v0/chat_session/create", { method: "POST", body: {} });
    const session = json?.data?.biz_data?.chat_session;
    if (!session?.id) throw new Error(`Cannot read chat session id: ${JSON.stringify(json).slice(0, 300)}`);
    return session.id;
  }

  async createPowHeader(targetPath) {
    const json = await this._request("/api/v0/chat/create_pow_challenge", {
      method: "POST",
      body: { target_path: targetPath },
    });

    const challenge = json?.data?.biz_data?.challenge;
    if (!challenge) throw new Error(`Cannot read PoW challenge: ${JSON.stringify(json).slice(0, 300)}`);

    const answer = await solvePow(challenge);
    const payload = {
      algorithm: challenge.algorithm,
      challenge: challenge.challenge,
      salt: challenge.salt,
      answer,
      signature: challenge.signature,
      target_path: targetPath,
    };
    return Buffer.from(JSON.stringify(payload), "utf8").toString("base64");
  }

  async complete({ sessionId, prompt, parentMessageId = null, modelType = null, thinkingEnabled = false, searchEnabled = false, onText = null }) {
    const pow = await this.createPowHeader(COMPLETION_PATH);
    const body = {
      chat_session_id: sessionId,
      parent_message_id: parentMessageId,
      model_type: modelType,
      preempt: false,
      prompt,
      ref_file_ids: [],
      thinking_enabled: thinkingEnabled,
      search_enabled: searchEnabled,
    };

    const res = await fetch(`${BASE_URL}${COMPLETION_PATH}`, {
      method: "POST",
      headers: {
        ...this._buildHeaders(),
        "X-DS-PoW-Response": pow,
      },
      body: JSON.stringify(body),
    });

    const contentType = String(res.headers.get("content-type") || "");
    if (!res.ok || !contentType.includes("text/event-stream")) {
      const text = await res.text();
      if (res.status === 401 || res.status === 403) throw authError("completion");
      try {
        const parsed = JSON.parse(text);
        if (parsed && (parsed.code === 40002 || parsed.code === 40003)) throw authError("completion");
      } catch (e) { if (e?.isAuthError) throw e; }
      throw new Error(`Completion failed: HTTP ${res.status}: ${text.slice(0, 1000)}`);
    }

    return streamSse(res, this.debug, onText);
  }
}

function authError(context) {
  const err = new Error(`Auth required during ${context}`);
  err.isAuthError = true;
  return err;
}

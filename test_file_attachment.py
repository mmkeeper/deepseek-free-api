"""Test attachment extraction: images + general file parts (xlsx, xml, pdf).

Covers OpenAI-compatible content part shapes forwarded to the proxy:
image_url/image, file, input_file — both data URLs and http(s) fetches.
"""
import asyncio
import base64
import sys
from unittest import mock

sys.path.insert(0, ".")

import server
from server import (
    _attachment_filename,
    _extract_attachments,
    _mime_ext,
    _sanitize_filename,
    messages_to_prompt,
)


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


# ─── _sanitize_filename / _mime_ext / _attachment_filename ───

def test_sanitize_filename_strips_paths_and_controls():
    assert _sanitize_filename(r"C:\tmp\..\cell.xlsx") == "cell.xlsx"
    assert _sanitize_filename("http://x/y/foo.xml") == "foo.xml"
    assert _sanitize_filename("bad\x00name.pdf") == "badname.pdf"
    assert _sanitize_filename("  spaced  .txt") == "spaced  .txt"
    assert _sanitize_filename("") == ""


def test_mime_ext_from_data_url():
    url = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,xxx"
    assert _mime_ext(url) == ".xlsx"
    url = "data:application/pdf;base64,xxx"
    assert _mime_ext(url) == ".pdf"
    url = "data:application/octet-stream;base64,xxx"
    assert _mime_ext(url) == ".bin"


def test_attachment_filename_nested_and_flat():
    assert _attachment_filename({"type": "file", "file": {"filename": "a.xlsx"}}) == "a.xlsx"
    assert _attachment_filename({"type": "input_file", "filename": "b.pdf"}) == "b.pdf"
    assert _attachment_filename({"type": "input_file", "input_file": {"filename": "c.xml"}}) == "c.xml"
    assert _attachment_filename({"type": "file", "file": {"file_data": "data:x"}}) == ""


# ─── _extract_attachments: data URLs ───

def test_extract_text_only_returns_empty():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Привет"}]}]
    assert asyncio.run(_extract_attachments(msgs)) == []


def test_extract_string_content_returns_empty():
    msgs = [{"role": "user", "content": "Привет"}]
    assert asyncio.run(_extract_attachments(msgs)) == []


def test_extract_no_messages_returns_empty():
    assert asyncio.run(_extract_attachments([])) == []


def test_extract_image_data_url_default_name():
    url = f"data:image/png;base64,{_b64('PNGDATA')}"
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Что тут?"},
        {"type": "image_url", "image_url": {"url": url}},
    ]}]
    out = asyncio.run(_extract_attachments(msgs))
    assert len(out) == 1
    filename, data, is_image = out[0]
    assert filename == "image.png"
    assert data == b"PNGDATA"
    assert is_image is True


def test_extract_image_data_url_keeps_explicit_filename():
    url = f"data:image/png;base64,{_b64('PNG')}"
    msgs = [{"role": "user", "content": [{"type": "image", "url": url, "filename": "фото.png"}]}]
    out = asyncio.run(_extract_attachments(msgs))
    assert out == [("фото.png", b"PNG", True)]


def test_extract_file_openai_format():
    url = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{_b64('XLSX')}"
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Ячейка замкнута?"},
        {"type": "file", "file": {"filename": "cell.xlsx", "file_data": url}},
    ]}]
    out = asyncio.run(_extract_attachments(msgs))
    assert len(out) == 1
    filename, data, is_image = out[0]
    assert filename == "cell.xlsx"
    assert data == b"XLSX"
    assert is_image is False


def test_extract_input_file_top_level_keys():
    url = f"data:application/xml;base64,{_b64('<a/>')}"
    msgs = [{"role": "user", "content": [
        {"type": "input_file", "filename": "schema.xml", "file_data": url},
    ]}]
    out = asyncio.run(_extract_attachments(msgs))
    assert out == [("schema.xml", b"<a/>", False)]


def test_extract_file_without_filename_derives_ext():
    url = f"data:application/pdf;base64,{_b64('PDF')}"
    msgs = [{"role": "user", "content": [{"type": "file", "file": {"file_data": url}}]}]
    out = asyncio.run(_extract_attachments(msgs))
    assert out == [("file.pdf", b"PDF", False)]


def test_extract_data_url_with_name_param():
    url = f"data:application/xml;name=doc.xml;base64,{_b64('<a/>')}"
    msgs = [{"role": "user", "content": [{"type": "file", "file": url}]}]
    out = asyncio.run(_extract_attachments(msgs))
    assert out == [("doc.xml", b"<a/>", False)]


def test_extract_bad_base64_skipped():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "только текст"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!not-base64!!!"}},
    ]}]
    out = asyncio.run(_extract_attachments(msgs))
    assert out == []


def test_extract_unknown_types_ignored():
    msgs = [{"role": "user", "content": [
        {"type": "video", "video": {"url": "http://x/v.mp4"}},
        {"type": "audio", "audio": {"url": "http://x/a.mp3"}},
    ]}]
    assert asyncio.run(_extract_attachments(msgs)) == []


def test_extract_takes_last_message_only():
    msgs = [
        {"role": "user", "content": [{"type": "file", "file": {"filename": "old.txt", "file_data": f"data:text/plain;base64,{_b64('old')}"}}]},
        {"role": "user", "content": [{"type": "text", "text": "только текст"}]},
    ]
    assert asyncio.run(_extract_attachments(msgs)) == []


# ─── _extract_attachments: http(s) fetches ───

class _FakeResp:
    def __init__(self, data: bytes, content_type: str):
        self.data = data
        self.content = data
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        pass


class _FakeHttpClient:
    def __init__(self, resp: _FakeResp):
        self._resp = resp
        self.urls = []

    async def get(self, url):
        self.urls.append(url)
        return self._resp


def test_extract_http_image(monkeypatch):
    fake = _FakeHttpClient(_FakeResp(b"PNG", "image/png"))
    monkeypatch.setattr(server, "get_http_client", lambda: fake)
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/img.png"}}]}]
    out = asyncio.run(_extract_attachments(msgs))
    assert fake.urls == ["https://x/img.png"]
    assert out == [("image.png", b"PNG", True)]


def test_extract_http_file_with_filename(monkeypatch):
    fake = _FakeHttpClient(_FakeResp(b"PDF", "application/pdf"))
    monkeypatch.setattr(server, "get_http_client", lambda: fake)
    msgs = [{"role": "user", "content": [{"type": "file", "file": {"filename": "report.pdf", "file_data": "https://x/report.pdf"}}]}]
    out = asyncio.run(_extract_attachments(msgs))
    assert out == [("report.pdf", b"PDF", False)]


def test_extract_http_file_without_filename_uses_content_type(monkeypatch):
    fake = _FakeHttpClient(_FakeResp(b"XML", "application/xml"))
    monkeypatch.setattr(server, "get_http_client", lambda: fake)
    msgs = [{"role": "user", "content": [{"type": "input_file", "file_data": "https://x/blob"}]}]
    out = asyncio.run(_extract_attachments(msgs))
    assert out == [("file.xml", b"XML", False)]


def test_extract_mixed_image_and_file(monkeypatch):
    img = _FakeHttpClient(_FakeResp(b"PNG", "image/png"))
    monkeypatch.setattr(server, "get_http_client", lambda: img)
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64('PNG')}"}},
        {"type": "file", "file": {"filename": "cell.xlsx", "file_data": f"data:application/octet-stream;base64,{_b64('X')}"}},
    ]}]
    out = asyncio.run(_extract_attachments(msgs))
    assert out == [("image.png", b"PNG", True), ("cell.xlsx", b"X", False)]


# ─── messages_to_prompt: file markers ───

def test_messages_to_prompt_image_marker():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Что на картинке?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64('P')}"}},
    ]}]
    prompt = messages_to_prompt(msgs)
    assert "[изображение]" in prompt
    assert "Что на картинке?" in prompt


def test_messages_to_prompt_file_marker_with_name():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Разбери файл"},
        {"type": "file", "file": {"filename": "cell.xlsx", "file_data": f"data:application/octet-stream;base64,{_b64('X')}"}},
    ]}]
    prompt = messages_to_prompt(msgs)
    assert "[файл: cell.xlsx]" in prompt
    assert "Разбери файл" in prompt


def test_messages_to_prompt_file_marker_without_name():
    msgs = [{"role": "user", "content": [
        {"type": "input_file", "file_data": f"data:application/xml;base64,{_b64('<a/>')}"},
    ]}]
    prompt = messages_to_prompt(msgs)
    assert "[файл]" in prompt


# ─── fetch_files: doubling backoff until the file is ready ───

from src.client import DeepSeekClient, _FILE_READY_BACKOFF


def _client(**kwargs):
    c = DeepSeekClient("cookie", "token", debug=False)
    for k, v in kwargs.items():
        setattr(c, k, v)
    return c


def _resp(files):
    return {"data": {"biz_data": {"files": files}}}


def _fake_file(status):
    return {"id": "file-1", "status": status}


class _FakeGetClient:
    """httpx-like stub for fetch_files: serves statuses in order, last repeats."""

    def __init__(self, statuses):
        self._statuses = statuses
        self.calls = 0

    async def get(self, url, headers=None, params=None):
        self.calls += 1
        idx = min(self.calls - 1, len(self._statuses) - 1)
        return mock.Mock(json=lambda: _resp([_fake_file(self._statuses[idx])]))


def test_fetch_files_returns_immediately_when_ready():
    sleeps = []
    fake = _FakeGetClient(["SUCCESS"])
    with mock.patch("src.client.get_http_client", lambda: fake), \
         mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        out = asyncio.run(_client().fetch_files(["file-1"]))

    assert out[0]["status"] == "SUCCESS"
    assert fake.calls == 1
    assert sleeps == []


def test_fetch_files_polls_with_doubling_backoff_then_succeeds():
    sleeps = []
    fake = _FakeGetClient(["PENDING", "PENDING", "SUCCESS"])
    with mock.patch("src.client.get_http_client", lambda: fake), \
         mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        out = asyncio.run(_client().fetch_files(["file-1"]))

    assert out[0]["status"] == "SUCCESS"
    assert fake.calls == 3
    assert sleeps == [0.5, 1.0], f"expected doubling 0.5,1 got {sleeps}"


def test_fetch_files_exhausts_backoff_for_stuck_file():
    sleeps = []
    fake = _FakeGetClient(["PENDING"])
    with mock.patch("src.client.get_http_client", lambda: fake), \
         mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        out = asyncio.run(_client().fetch_files(["file-1"]))

    assert sleeps == _FILE_READY_BACKOFF, f"expected {_FILE_READY_BACKOFF} got {sleeps}"
    assert fake.calls == len(_FILE_READY_BACKOFF) + 1
    assert out[0]["status"] == "PENDING"


def test_fetch_files_stops_on_failed():
    sleeps = []
    fake = _FakeGetClient(["PENDING", "FAILED"])
    with mock.patch("src.client.get_http_client", lambda: fake), \
         mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        out = asyncio.run(_client().fetch_files(["file-1"]))

    assert out[0]["status"] == "FAILED"
    assert sleeps == [0.5]
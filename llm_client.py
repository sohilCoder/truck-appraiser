"""
LLM client — supports Gemini, Groq (OpenAI-compatible), and OpenAI.

Provider is selected automatically from whichever key is present in the
environment, or can be forced with VISION_PROVIDER=gemini|groq|openai.

Public interface used by pipeline.py:
    call_json(messages, *, temperature, max_tokens, force_json, attempts)
    image_message(prompt, data_url)
    bytes_to_data_url(raw, mime)
    LLMError
"""

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

# ------------------------------------------------------------------ config

def _env(key, default=""):
    return os.environ.get(key, default).strip()


def _detect_provider():
    forced = _env("VISION_PROVIDER").lower()
    if forced in ("gemini", "groq", "openai"):
        return forced
    if _env("GEMINI_API_KEY"):
        return "gemini"
    if _env("GROQ_API_KEY"):
        return "groq"
    if _env("OPENAI_API_KEY"):
        return "openai"
    return "gemini"          # will fail with a clear message at call time


def _default_model(provider):
    defaults = {
        "gemini": "gemini-3.6-flash",
        "groq":   "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai": "gpt-4o",
    }
    return _env("VISION_MODEL") or defaults.get(provider, "gemini-3.6-flash")


PROVIDER = _detect_provider()
MODEL    = _default_model(PROVIDER)
TIMEOUT  = int(_env("LLM_TIMEOUT") or "60")


class LLMError(Exception):
    """Raised when the LLM cannot be reached or returns an unusable response."""


# ------------------------------------------------------------------ helpers

def bytes_to_data_url(raw, mime="image/jpeg"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def image_message(prompt, data_url):
    """
    Returns a provider-agnostic message dict that call_json understands.
    The '_vision' key carries both the prompt and the image data.
    """
    return {"_vision": True, "prompt": prompt, "data_url": data_url}


def _extract_json(text):
    if not text:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth, in_str, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            escaped = (not escaped and ch == "\\")
            if not escaped and ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object")


def _post_raw(url, payload, headers):
    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:700]
        raise LLMError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Network error: {exc.reason}") from exc
    except TimeoutError:
        raise LLMError("Request timed out.")


# ------------------------------------------------------------------ Gemini

def _gemini_call(messages, temperature, max_tokens, force_json):
    key = _env("GEMINI_API_KEY")
    if not key:
        raise LLMError(
            "GEMINI_API_KEY is not set. Open .env and paste your key next to GEMINI_API_KEY=."
        )
    model = MODEL
    url   = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    contents = []
    for msg in messages:
        if msg.get("_vision"):
            header, _, b64 = msg["data_url"].partition(",")
            mime = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
            contents.append({"role": "user", "parts": [
                {"text": msg["prompt"]},
                {"inlineData": {"mimeType": mime, "data": b64}},
            ]})
        else:
            role = "model" if msg.get("role") == "assistant" else "user"
            text = msg.get("content", "")
            # merge system prompt into first user turn
            if msg.get("role") == "system":
                if contents and contents[-1]["role"] == "user":
                    contents[-1]["parts"].insert(0, {"text": text + "\n\n"})
                else:
                    contents.append({"role": "user", "parts": [{"text": text}]})
                continue
            contents.append({"role": role, "parts": [{"text": text}]})

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    if force_json:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    data = _post_raw(url, payload, {"Content-Type": "application/json"})
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        finish = ""
        try:
            finish = data["candidates"][0].get("finishReason", "")
        except Exception:
            pass
        if finish == "SAFETY":
            raise LLMError("Blocked by Gemini safety filters. Try a different image.")
        raise LLMError(f"Unexpected Gemini response: {json.dumps(data)[:300]}")


# ------------------------------------------------------------------ OpenAI-compatible (Groq + OpenAI)

def _openai_call(messages, temperature, max_tokens, force_json):
    if PROVIDER == "groq":
        key = _env("GROQ_API_KEY")
        url = "https://api.groq.com/openai/v1/chat/completions"
        if not key:
            raise LLMError("GROQ_API_KEY is not set. Open .env and paste your key.")
    else:
        key = _env("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/chat/completions"
        if not key:
            raise LLMError("OPENAI_API_KEY is not set. Open .env and paste your key.")

    oai_messages = []
    for msg in messages:
        if msg.get("_vision"):
            oai_messages.append({"role": "user", "content": [
                {"type": "text",      "text": msg["prompt"]},
                {"type": "image_url", "image_url": {"url": msg["data_url"]}},
            ]})
        else:
            oai_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    payload = {
        "model":       MODEL,
        "messages":    oai_messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    data = _post_raw(url, payload, {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
    })
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected response: {json.dumps(data)[:300]}") from exc


# ------------------------------------------------------------------ public

def call_json(messages, *, temperature=0.15, max_tokens=2000,
              force_json=False, attempts=2):
    """Call the configured LLM and return a parsed JSON dict."""
    last_err = None
    msgs     = list(messages)

    for attempt in range(attempts):
        try:
            if PROVIDER == "gemini":
                raw = _gemini_call(msgs, temperature, max_tokens, force_json)
            else:
                raw = _openai_call(msgs, temperature, max_tokens, force_json)
        except LLMError:
            if attempt == attempts - 1:
                raise
            time.sleep(1.5)
            continue

        try:
            return _extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            msgs = msgs + [
                {"role": "assistant", "content": raw[:2000]},
                {"role": "user",      "content":
                    "That was not valid JSON. Reply with the JSON object only — "
                    "no prose, no code fences."},
            ]
            temperature = 0.0

    raise LLMError(f"Model did not return parseable JSON after {attempts} attempts: {last_err}")

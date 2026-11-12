"""E3 OpenRouter vision-model client. Fail-closed. Standard library only.

- Key: parsed in-process from <repo>/.env (`jev_openrouter=`); the value is
  never printed, logged, committed, or embedded in prompts.
- Every response cached as JSONL keyed by
  sha256(model | prompt_version | demo | transition | frame-hashes).
- Resumable: cached successes are skipped; failures counted and reported.
- Fail-closed: smoke validation (5 fixed transitions, parse 5/5) must pass
  before any production pass; excessive failures abort (no B2 fallback).
"""
import base64
import concurrent.futures
import hashlib
import io
import json
import os
import threading
import time
import urllib.request

from . import config

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.6-flash"
CACHE_SUBDIR = "e3_vlm_cache"
# Provider new-account limit: 20 req/min on this model. Global pacing keeps
# all worker threads under ~17 RPM; 429s sleep until the window resets.
MIN_REQUEST_INTERVAL_S = 3.5
N429_WAITS = {"n": 0}
_GATE = threading.Lock()
_LAST_START = {"t": 0.0}


def _gate():
    with _GATE:
        dt = time.time() - _LAST_START["t"]
        if dt < MIN_REQUEST_INTERVAL_S:
            time.sleep(MIN_REQUEST_INTERVAL_S - dt)
        _LAST_START["t"] = time.time()


def load_key():
    path = os.path.join(config.REPO_ROOT, ".env")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("jev_openrouter="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if len(key) < 20:
                    raise RuntimeError("jev_openrouter value looks invalid")
                return key
    raise RuntimeError("jev_openrouter not found in repo .env")


REASONING = {"effort": "low", "exclude": True}


def _post(model, messages, temperature, response_format, max_tokens, timeout,
          key, reasoning=REASONING):
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens}
    if response_format:
        body["response_format"] = response_format
    if reasoning:
        body["reasoning"] = reasoning
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/jev-test",
                 "X-Title": "jev-e3-visual-annotation"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _ratelimit_wait(e):
    """Seconds to wait on HTTP 429 (Retry-After / reset headers, else 65s)."""
    try:
        hdrs = getattr(e, "headers", {}) or {}
        ra = hdrs.get("Retry-After") or hdrs.get("retry-after")
        if ra:
            return max(1.0, float(ra) + 1.0)
        for k in ("X-RateLimit-Reset", "x-ratelimit-reset"):
            if k in hdrs:
                reset_ms = float(hdrs[k])
                now_ms = time.time() * 1000
                if reset_ms > now_ms:
                    return (reset_ms - now_ms) / 1000.0 + 2.0
    except Exception:  # noqa: BLE001
        pass
    return 65.0


def chat(messages, temperature=0, response_format=None, max_tokens=2000,
         timeout=180, attempts=5, model=MODEL, key=None, reasoning=REASONING):
    """Single chat call. Returns (content_str, usage_dict, attempts_used).

    HTTP 429 waits out the provider window (up to 10 waits) without burning
    the error budget; other errors back off and retry.
    """
    import urllib.error
    key = key or load_key()
    last = None
    waits = 0
    for a in range(1, attempts + 1):
        _gate()
        try:
            out = _post(model, messages, temperature, response_format,
                        max_tokens, timeout, key, reasoning)
            msg = out["choices"][0]["message"]
            return msg.get("content", ""), out.get("usage", {}), a
        except urllib.error.HTTPError as e:
            last = e
            if getattr(e, "code", None) == 429 and waits < 10:
                waits += 1
                N429_WAITS["n"] += 1
                time.sleep(_ratelimit_wait(e))
                continue
            time.sleep(min(2 ** a, 15))
        except Exception as e:  # noqa: BLE001 - counted, retried, reported
            last = e
            time.sleep(min(2 ** a, 15))
    raise RuntimeError(f"OpenRouter call failed after {attempts} attempts "
                       f"({waits} rate-limit waits): {type(last).__name__}")


def frame_b64(png_bytes):
    return base64.b64encode(png_bytes).decode()


def user_content(text, png_list):
    parts = [{"type": "text", "text": text}]
    for p in png_list:
        parts.append({"type": "image_url",
                      "image_url": {"url": "data:image/png;base64," + frame_b64(p)}})
    return [{"role": "user", "content": parts}]


def cache_path(schema):
    d = os.path.join(config.DATA_ROOT, CACHE_SUBDIR)
    os.makedirs(d, exist_ok=True)
    # v1 used state-only rendering and can contain images from the wrong scene.
    # Keep it for cost/audit evidence, but never reuse it with XML-correct frames.
    return os.path.join(d, f"{schema}_v2_episode_xml.jsonl")


def record_key(model, prompt_version, demo, tids, frame_hashes):
    s = "|".join([model, prompt_version, demo, ",".join(map(str, tids)),
                  ",".join(frame_hashes)])
    return hashlib.sha256(s.encode()).hexdigest()


def load_cache(schema):
    path = cache_path(schema)
    recs = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                recs[r["key"]] = r
    return recs, path


def append_record(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_pass(schema, prompt_version, model, jobs, build_messages, parse,
             max_tokens, log, workers=2, max_fail_rate=0.05):
    """Resumable production pass over jobs.

    jobs: list of dicts {demo, tids, frames:[png_bytes], frame_hashes:[hex]}.
    build_messages(job) -> messages. parse(content) -> dict or raises.
    Returns stats dict. Raises on excessive failure (fail-closed).
    """
    recs, path = load_cache(schema)
    # Retry records that previously failed (load_cache keeps last-wins, so a
    # successful retry overwrites the failure). Only "ok" records are final.
    def _ok(j):
        r = recs.get(record_key(model, prompt_version, j["demo"], j["tids"],
                                j["frame_hashes"]))
        return r is not None and r.get("status") == "ok"
    todo = [j for j in jobs if not _ok(j)]
    log(f"[vlm:{schema}] cached {len(recs)} records, {len(todo)} to request")
    stats = {"requests": 0, "ok": 0, "parse_fail": 0, "http_fail": 0,
             "retries": 0, "prompt_tokens": 0, "completion_tokens": 0}
    key = load_key()

    def one(job):
        rkey = record_key(model, prompt_version, job["demo"], job["tids"],
                          job["frame_hashes"])
        msgs = build_messages(job)
        try:
            content, usage, att = chat(msgs, temperature=0,
                                       response_format={"type": "json_object"},
                                       max_tokens=max_tokens, model=model, key=key)
        except Exception:  # noqa: BLE001
            return (rkey, job, None, "http_fail", {}, 3)
        try:
            parsed = parse(content)
        except Exception:  # noqa: BLE001
            return (rkey, job, content, "parse_fail", usage, att)
        return (rkey, job, {"parsed": parsed, "raw": content}, "ok", usage, att)

    N429_WAITS["n"] = 0
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, j): j for j in todo}
        for fut in concurrent.futures.as_completed(futs):
            rkey, job, payload, status, usage, att = fut.result()
            stats["requests"] += 1
            stats["retries"] += max(0, att - 1)
            stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            rec = {"key": rkey, "model": model, "prompt_version": prompt_version,
                   "demo": job["demo"], "tids": job["tids"],
                   "frame_hashes": job["frame_hashes"], "status": status,
                   "payload": payload, "usage": usage,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            append_record(path, rec)
            recs[rkey] = rec
            stats[status] += 1
            done += 1
            if done % 100 == 0:
                log(f"[vlm:{schema}] {done}/{len(todo)} "
                    f"(ok={stats['ok']} parse_fail={stats['parse_fail']} "
                    f"http_fail={stats['http_fail']})")
    # recount cleanly from cache for this pass's jobs
    def _ok2(j):
        return recs.get(record_key(model, prompt_version, j["demo"], j["tids"],
                                   j["frame_hashes"]), {}).get("status") == "ok"
    ok = sum(1 for j in jobs if _ok2(j))
    fail = len(jobs) - ok
    stats["jobs"] = len(jobs)
    stats["ok_total"] = ok
    stats["rate_limit_waits"] = N429_WAITS["n"]
    if jobs and fail / len(jobs) > max_fail_rate:
        raise RuntimeError(
            f"VLM pass {schema} failure rate {fail}/{len(jobs)} exceeds "
            f"{max_fail_rate}: FAIL CLOSED, no fallback")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    stats["cache_sha256"] = h.hexdigest()
    stats["cache_path"] = path
    # OpenRouter-listed price for the frozen model ($/token)
    stats["est_cost_usd"] = (stats["prompt_tokens"] * 0.75e-6
                             + stats["completion_tokens"] * 3.75e-6)
    log(f"[vlm:{schema}] done: {ok}/{len(jobs)} ok, tokens "
        f"{stats['prompt_tokens']}/{stats['completion_tokens']}, "
        f"est cost ${stats['est_cost_usd']:.2f}, cache {stats['cache_sha256'][:12]}")
    return stats

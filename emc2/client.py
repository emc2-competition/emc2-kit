"""Talk to the vLLM server: streamed completions (token IDs + timing) and /metrics counters."""

import http.client
import json
import time
import urllib.parse
import urllib.request

ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
DRAFTS = "vllm:spec_decode_num_drafts_total"


def complete(url, prompt_ids, max_tokens, stop_token_ids, timeout):
    """Greedy, streamed completion. Returns {"token_ids", "ttft_s", "total_s", "tpot_s"}.

    TTFT: time from sending the request to receiving the first generated token.
    TPOT: time per output token after the first, (total - TTFT) / (tokens - 1), the
    standard vLLM benchmark definition. None when fewer than 2 tokens were generated.
    """
    u = urllib.parse.urlparse(url)
    body = json.dumps({
        "model": "target",
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "return_token_ids": True,
        "stop_token_ids": stop_token_ids,
    })
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    t0 = time.monotonic()
    ttft = None
    ids = []
    try:
        conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices", []):
                new = choice.get("token_ids") or []
                if new and ttft is None:
                    ttft = time.monotonic() - t0
                ids.extend(new)
    finally:
        conn.close()
    total = time.monotonic() - t0
    ttft = ttft if ttft is not None else total
    tpot = (total - ttft) / (len(ids) - 1) if len(ids) > 1 else None
    return {"token_ids": ids, "ttft_s": ttft, "total_s": total, "tpot_s": tpot}


def spec_counters(url):
    """(accepted draft tokens, drafting rounds) so far, summed over all label sets."""
    with urllib.request.urlopen(url + "/metrics", timeout=30) as r:
        text = r.read().decode()
    acc = drafts = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name == ACCEPTED:
            acc += float(line.rsplit(" ", 1)[1])
        elif name == DRAFTS:
            drafts += float(line.rsplit(" ", 1)[1])
    return acc, drafts

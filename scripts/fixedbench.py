#!/usr/bin/env python3
"""Fixed greedy workload for kernel A/Bs: temperature 0, ignore_eos, fixed prompts and lengths, so every variant decodes
(nearly) the same text and acceptance is (nearly) the same. Reports tok/s per concurrency and the server's accepted
tokens per step. usage: fixedbench.py LABEL [C=1,16] [TOKENS=768] [REPS=1]"""
import asyncio, json, os, sys, time, urllib.request
import aiohttp

URL = os.environ.get("OPT_URL", "http://127.0.0.1:8000")
RESULTS = os.environ.get("OPT_RESULTS", "results")
PROMPTS = [
    "Write a Python module that parses ISO-8601 durations into a dataclass, with validation, then explain how it works.",
    "Implement an LRU cache with per-entry TTL in Go, safe for concurrent use, then explain the locking strategy.",
    "Write a Rust function that merges overlapping time intervals in place, with unit tests, then explain it.",
    "Write a short story about a lighthouse keeper who discovers that the light is being watched from the sea.",
    "Explain how public-key cryptography works, with a worked RSA example using small numbers.",
    "Write a TypeScript implementation of a typed event bus with wildcard topics and once-listeners, with JSDoc.",
    "Describe the causes and consequences of the 2008 financial crisis in a structured essay.",
    "Write a SQL schema for a library system and five example queries with explanations.",
]


def metric(name):
    txt = urllib.request.urlopen(URL + "/metrics").read().decode()
    tot = 0.0
    for line in txt.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            tot += float(line.rsplit(" ", 1)[1])
    return tot


async def one(sess, prompt, n):
    body = {"model": "Qwen/Qwen3.8-Flash-Next", "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
            "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    async with sess.post(URL + "/v1/chat/completions", json=body) as r:
        d = await r.json()
    return d["usage"]["completion_tokens"]


async def level(c, n):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:
        await one(sess, "Say hello.", 8)                    # warm
        d0, a0 = metric("vllm:spec_decode_num_drafts_total"), metric("vllm:spec_decode_num_accepted_tokens_total")
        t0 = time.time()
        if c == 1:
            toks = 0
            for p in PROMPTS[:3]:
                toks += await one(sess, p, n)
        else:
            res = await asyncio.gather(*[one(sess, PROMPTS[i % len(PROMPTS)], n) for i in range(c)])
            toks = sum(res)
        dt = time.time() - t0
        d1, a1 = metric("vllm:spec_decode_num_drafts_total"), metric("vllm:spec_decode_num_accepted_tokens_total")
    acc = 1 + (a1 - a0) / max(1, d1 - d0)
    return toks / dt, acc, dt


async def main():
    label = sys.argv[1]
    cs = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,16").split(",")]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 768
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    out = {"label": label, "t": time.time(), "n": n, "levels": {}}
    for c in cs:
        vals = []
        for r in range(reps):
            tps, acc, dt = await level(c, n)
            vals.append((round(tps, 2), round(acc, 3), round(dt, 2)))
        out["levels"][c] = vals
        print(f"{label}  c={c:2d}  " + "  ".join(f"{t:7.2f} tok/s (acc {a:.2f}, {d:.1f}s)" for t, a, d in vals), flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    open(f"{RESULTS}/fixedbench.jsonl", "a").write(json.dumps(out) + "\n")


asyncio.run(main())

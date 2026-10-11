#!/usr/bin/env python3
"""Teacher-forced next-token distributions (top-20 prompt logprobs) on a fixed corpus.
  klprobe.py capture OUT.json          capture on the running server
  klprobe.py compare REF.json NEW.json  KL(ref || new) per token (on ref's top-20, renormalized), NLL, top-1 agreement
"""
import json, math, os, sys, urllib.request

CORPUS = os.environ.get("KL_CORPUS", "kl_corpus.json")   # built on first capture from KL_SOURCES, then reused
URL = os.environ.get("OPT_URL", "http://127.0.0.1:8000")


def build_corpus():
    parts = []
    srcs = os.environ["KL_SOURCES"].split(",")   # comma-separated text files, each > 16 kB (prose and code)
    for p in srcs:
        t = open(p, errors="ignore").read()
        for off in (3000, 12000):
            if len(t) > off + 4000:
                parts.append((os.path.basename(p) + f"@{off}", t[off:off + 4000]))
    json.dump(parts, open(CORPUS, "w"))
    return parts


def capture(out):
    try:
        parts = json.load(open(CORPUS))
    except FileNotFoundError:
        parts = build_corpus()
    res = {}
    for name, text in parts:
        body = {"model": "Qwen/Qwen3.8-Flash-Next", "prompt": text, "max_tokens": 1, "temperature": 0,
                "prompt_logprobs": 20}
        req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        pl = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["prompt_logprobs"]
        rows = []
        for e in pl[1:]:
            # {token_id: {logprob, rank, decoded_token}}: the actual token (any rank) + the top-20
            rows.append({k: v["logprob"] for k, v in e.items()} | {"_act": [k for k, v in e.items() if True][0]})
        res[name] = rows
    json.dump(res, open(out, "w"))
    print("captured", sum(len(v) for v in res.values()), "positions")


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    kl_sum, n, nll_a, nll_b, agree = 0.0, 0, 0.0, 0.0, 0
    worst = []
    for name in A:
        for ra, rb in zip(A[name], B[name]):
            act = ra["_act"]
            la = {k: v for k, v in ra.items() if k != "_act"}
            lb = {k: v for k, v in rb.items() if k != "_act"}
            common = [k for k in la if k in lb]
            if not common:
                continue
            pa = [math.exp(la[k]) for k in common]
            pb = [math.exp(lb[k]) for k in common]
            sa, sb = sum(pa), sum(pb)
            kl = sum(x / sa * math.log((x / sa) / max(y / sb, 1e-30)) for x, y in zip(pa, pb))
            kl_sum += kl; n += 1
            nll_a -= ra.get(act, la.get(act, 0.0)); nll_b -= rb.get(act, lb.get(act, 0.0))
            agree += max(la, key=la.get) == max(lb, key=lb.get)
            worst.append(kl)
    worst.sort()
    print(f"positions {n}  KL mean {kl_sum/n:.5f}  p99 {worst[int(0.99*(n-1))]:.4f}  max {worst[-1]:.3f}  "
          f"NLL {nll_a/n:.4f} -> {nll_b/n:.4f} (Δ {(nll_b-nll_a)/n:+.4f})  top-1 agreement {100*agree/n:.2f}%")


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])

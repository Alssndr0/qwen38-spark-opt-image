#!/usr/bin/env python3
"""Task-level accuracy on problems with exactly computable answers (no datasets needed offline). v2 (harder).
Greedy, thinking off, 16 concurrent requests. Three kinds, 100 each (fixed seed, identical across runs):
  arith  multi-step word problems with integer answers ("Answer: <n>" on the last line)
  date   day-of-week / days-between arithmetic from given dates
  json   produce a JSON object with given keys and computed values; scored on validity + exact values
usage: taskeval.py LABEL   -> prints accuracy per kind, appends to $OPT_RESULTS/taskeval.jsonl, saves answers"""
import datetime, json, os, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("OPT_URL", "http://127.0.0.1:8000")
RESULTS = os.environ.get("OPT_RESULTS", "results")
rnd = random.Random(1234)
NAMES = ["Alice", "Marco", "Yuki", "Priya", "Omar", "Lena", "Diego", "Sara"]
ITEMS = ["apples", "books", "tickets", "boxes", "pens", "cookies", "stamps", "marbles"]


def arith():
    n, it = rnd.choice(NAMES), rnd.choice(ITEMS)
    a, b, c, d = rnd.randint(12, 90), rnd.randint(2, 9), rnd.randint(3, 40), rnd.randint(2, 6)
    total = (a * b - c) // d if (a * b - c) % d == 0 else None
    while total is None:
        c = rnd.randint(3, 40)
        total = (a * b - c) // d if (a * b - c) % d == 0 else None
    q = (f"{n} buys {b} packs of {it} with {a} {it} in each pack. {n} gives away {c} {it}, then shares the rest "
         f"equally among {d} friends. How many {it} does each friend get? Work it out briefly, then end with a line "
         f"'Answer: <number>'.")
    return q, str(total)


def date():
    start = datetime.date(2020, 1, 1) + datetime.timedelta(days=rnd.randint(0, 2500))
    k = rnd.randint(20, 400)
    end = start + datetime.timedelta(days=k)
    q = (f"What day of the week is {k} days after {start.strftime('%A, %d %B %Y')}? Reason briefly, then end with a "
         f"line 'Answer: <weekday>'.")
    return q, end.strftime("%A")


def jsontask():
    w, h, p = rnd.randint(3, 40), rnd.randint(3, 40), rnd.randint(5, 95)
    q = (f"A rectangle is {w} cm wide and {h} cm tall. Return ONLY a JSON object (no code fence, no prose) with keys "
         f"\"area\" (integer), \"perimeter\" (integer) and \"discounted\" ({p}% off a price of 200, as a number).")
    return q, json.dumps({"area": w * h, "perimeter": 2 * (w + h), "discounted": round(200 * (1 - p / 100), 2)})


def ware():
    A, B, C, D = rnd.randint(23, 180), rnd.randint(12, 96), rnd.randint(11, 140), rnd.randint(7, 64)
    tot = A * B + C * D
    E = rnd.randint(tot // 5, tot // 2)
    F = rnd.randint(7, 48)
    q = (f"A warehouse holds {A} crates with {B} units each. It receives {C} more crates with {D} units each, then "
         f"ships out {E} units. The remaining units are packed into boxes of {F}. How many FULL boxes can be packed? "
         f"Show the steps briefly, then end with a line 'Answer: <number>'.")
    return q, str((tot - E) // F)


def days():
    d1 = datetime.date(2001, 1, 1) + datetime.timedelta(days=rnd.randint(0, 6000))
    d2 = d1 + datetime.timedelta(days=rnd.randint(400, 5500))
    q = (f"How many days are there from {d1.strftime('%d %B %Y')} to {d2.strftime('%d %B %Y')} (count the days "
         f"between them, so the first date does not count and the last one does)? Show the steps briefly, then end "
         f"with a line 'Answer: <number>'.")
    return q, str((d2 - d1).days)


def mult():
    a, b = rnd.randint(1000, 9999), rnd.randint(100, 999)
    return (f"Compute {a} × {b}. Show the steps briefly, then end with a line 'Answer: <number>' (digits only).",
            str(a * b))


def stats():
    xs = [rnd.randint(10, 999) for _ in range(rnd.randint(7, 11))]
    q = (f"For the list {xs}, return ONLY a JSON object (no code fence, no prose) with keys \"sum\" (integer), "
         f"\"mean\" (rounded to 2 decimals), \"max\" (integer) and \"sorted\" (the list in ascending order).")
    return q, json.dumps({"sum": sum(xs), "mean": round(sum(xs) / len(xs), 2), "max": max(xs), "sorted": sorted(xs)})


KINDS = ("ware", "days", "mult", "stats")
TASKS = [("ware", ware()) for _ in range(100)] + [("days", days()) for _ in range(100)] + \
        [("mult", mult()) for _ in range(100)] + [("stats", stats()) for _ in range(100)]


def ask(t):
    kind, (q, gold) = t
    body = {"model": "Qwen/Qwen3.8-Flash-Next", "messages": [{"role": "user", "content": q}], "max_tokens": 1200,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"] or ""
    if kind == "stats":
        try:
            o = json.loads(out.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
            g = json.loads(gold)
            ok = (o.get("sum") == g["sum"] and o.get("max") == g["max"] and o.get("sorted") == g["sorted"]
                  and abs(float(o.get("mean", -1)) - g["mean"]) < 0.006)
        except Exception:
            ok = False
    else:
        m = re.findall(r"Answer:\s*\**\s*([0-9][0-9,]*)", out)
        ok = bool(m) and m[-1].replace(",", "") == gold
    return kind, ok, out


t0 = time.time()
with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(ask, TASKS))
acc = {k: sum(ok for kk, ok, _ in res if kk == k) for k in KINDS}
label = sys.argv[1] if len(sys.argv) > 1 else ""
print(f"{label}: " + "  ".join(f"{k} {acc[k]}/100" for k in KINDS) + f"  total {sum(acc.values())}/400  "
      f"({time.time() - t0:.0f}s)", flush=True)
os.makedirs(RESULTS, exist_ok=True)
json.dump([o for _, _, o in res], open(f"{RESULTS}/taskeval-{label}.json", "w"))
with open(f"{RESULTS}/taskeval.jsonl", "a") as f:
    f.write(json.dumps({"label": label, "t": time.time(), **acc}) + "\n")

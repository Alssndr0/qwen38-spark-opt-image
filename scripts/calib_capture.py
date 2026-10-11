#!/usr/bin/env python3
"""Calibration capture for GPTQ on a running serve booted with OPT_CALIB_DIR=/cache/<dir> (see README).
env: CALIB_OUT  the same directory as seen from the host (e.g. <recipe>/cache/calib), required
     CALIB_SRC  a directory with the vLLM package source (copied out of the image, see README), for the code texts
     CALIB_MORE=1  the larger prompt set (GPTQ v2)
1. With the recorders disarmed, let the model write varied text (languages, code, maths, thinking on).
2. Arm the recorders (touch GO) and prefill those texts plus code files that are NOT in the KL corpus, so every
   target linear sees ~4096 sampled rows of real activations.
Held out on purpose: the KL corpus (ref/kl_corpus.json) is never used here."""
import json, os, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("OPT_URL", "http://127.0.0.1:8000")
OUT = os.environ["CALIB_OUT"]
SRC = os.environ.get("CALIB_SRC", "vllm-src")
PROMPTS = [
    ("Spiega in italiano, in modo dettagliato, come funziona la fotosintesi e perché è importante per il clima.", False),
    ("Écris un essai en français sur l'histoire de l'imprimerie et son influence sur la Renaissance.", False),
    ("Erkläre auf Deutsch ausführlich, wie ein Verbrennungsmotor funktioniert, mit Beispielen.", False),
    ("Escribe en español una guía práctica para aprender a programar en Python desde cero.", False),
    ("用中文详细介绍长城的历史、建造过程以及它在中国文化中的意义。", False),
    ("日本語で、日本の四季とそれぞれの季節の行事について詳しく説明してください。", False),
    ("Write a detailed technical blog post about how B-trees work in databases, with examples.", False),
    ("Write a Rust program that implements a thread pool with graceful shutdown, then explain it.", False),
    ("Implement a red-black tree in C++ with insertion and deletion, with comments.", False),
    ("Solve step by step: a tank fills in 6 hours with pipe A and 9 hours with pipe B; pipe C drains it in 12 hours. "
     "How long with all three open? Then generalize the method.", True),
    ("Prove that there are infinitely many primes, then discuss the prime number theorem informally.", True),
    ("Plan a two-week itinerary through Japan for a family with two children, with budget estimates.", False),
    ("Write a short story about a robot who learns to paint, in the style of a fairy tale.", False),
    ("Explain the differences between TCP and QUIC, including congestion control and head-of-line blocking.", True),
    ("Write SQL queries for a retail analytics dashboard: cohort retention, basket analysis, and RFM segments.", False),
    ("Debug this: a React component re-renders infinitely after adding useEffect with an object dependency. "
     "Explain the cause and three fixes with code.", True),
]
if os.environ.get("CALIB_MORE"):
    PROMPTS += [
        ("Scrivi una lettera formale in italiano per chiedere un rimborso a una compagnia aerea, poi riassumila.", False),
        ("Escreva em português um artigo sobre energia solar no Brasil, com prós e contras.", False),
        ("Напиши по-русски подробный рассказ о путешествии по Транссибирской магистрали.", False),
        ("한국어로 한국 음식의 역사와 대표적인 요리 다섯 가지를 자세히 설명해 주세요.", False),
        ("اكتب باللغة العربية مقالاً عن تاريخ علم الفلك عند العرب.", False),
        ("Schrijf in het Nederlands een uitleg over hoe dijken en polders werken.", False),
        ("Write a Go HTTP server with middleware for logging, auth and rate limiting, with tests.", False),
        ("Write a Python script that parses Apache logs, aggregates by status and endpoint, and plots results.", False),
        ("Return only a JSON array of 8 fictional products with id, name, price, tags, stock, in valid JSON.", False),
        ("Translate this into French, German and Spanish, then explain any idioms: 'It's raining cats and dogs, "
         "so let's call it a day and hit the sack early.'", False),
        ("Explain the transformer architecture in depth: attention, positional encoding, layer norm, training.", True),
        ("Compute the eigenvalues of [[2,1],[1,3]] step by step, then explain what they mean geometrically.", True),
        ("Write a bash script that backs up a directory with rotation and checksums, with comments.", False),
        ("Summarize the causes of World War I and compare historians' interpretations.", False),
        ("You are a customer-support agent. A user says their order arrived damaged. Write the whole conversation.", False),
        ("Write a LaTeX document with a theorem, a proof, a table and a figure placeholder about graph coloring.", False),
    ]
FILES = ["v1/worker/gpu_model_runner.py", "engine/arg_utils.py", "v1/engine/core.py", "config/model.py",
         "v1/attention/backends/flash_attn.py", "v1/core/kv_cache_manager.py"]


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def gen(p):
    text, think = p
    r = post("/v1/chat/completions", {"model": "Qwen/Qwen3.8-Flash-Next", "messages": [{"role": "user", "content": text}],
                                      "max_tokens": 900, "temperature": 0.8,
                                      "chat_template_kwargs": {"enable_thinking": think}})
    m = r["choices"][0]["message"]
    return text + "\n\n" + (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")


os.makedirs(OUT, exist_ok=True)
if os.path.exists(OUT + "/GO"):
    os.remove(OUT + "/GO")
with ThreadPoolExecutor(8) as ex:
    texts = list(ex.map(gen, PROMPTS))
print("generated", len(texts), "texts,", sum(len(t) for t in texts), "chars", flush=True)
rnd = random.Random(0)
for f in FILES:
    if not os.path.exists(os.path.join(SRC, f)):
        print("skip missing", f, flush=True)
        continue
    t = open(os.path.join(SRC, f), errors="ignore").read()
    for _ in range(3 if os.environ.get("CALIB_MORE") else 2):
        o = rnd.randrange(0, max(1, len(t) - 6000))
        texts.append(t[o:o + 6000])
rnd.shuffle(texts)
open(OUT + "/GO", "w").close()
import time
time.sleep(2.5)                                  # recorders re-check the flag at most once a second
for i, t in enumerate(texts):                     # one at a time: each prefill is its own eager call
    post("/v1/completions", {"model": "Qwen/Qwen3.8-Flash-Next", "prompt": t, "max_tokens": 1, "temperature": 0})
os.remove(OUT + "/GO")
n = len([x for x in os.listdir(OUT) if x.endswith(".pt")])
print("prefilled", len(texts), "texts; recorded files:", n, flush=True)

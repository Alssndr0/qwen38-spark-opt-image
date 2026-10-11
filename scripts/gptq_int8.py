"""Offline GPTQ for the target's INT8 projections (serve stopped; run inside the image, where opt_fp8 is importable).
Reads OPT_CALIB_DIR recordings (<module>.W.pt = weight in vLLM's merged layout, <module>.<k>.pt = sampled input
rows), and for every module:
  H = X^T X over 80 % of the rows; static group scales = opt_fp8.quant_int8 on the original W (exactly what the
  serve recomputes); GPTQ with act-order processing (storage order unchanged, so Marlin needs no g_idx).
Reports the output error on the held-out 20 % of rows for round-to-nearest vs GPTQ, per family, and writes
{module: int8 codes [N, K]} for OPT_TARGET_QCODES.
usage: gptq_int8.py CALIB_DIR OUT.pt [group=128] [damp=0.01] [W_DIR=CALIB_DIR]   (W_DIR: where the bf16 <module>.W.pt
files are, for a second pass whose rows were recorded on the quantized model)"""
import glob, os, re, sys, time
from collections import defaultdict
import torch
import opt_fp8

CAL, OUT = sys.argv[1], sys.argv[2]
GROUP = int(sys.argv[3]) if len(sys.argv) > 3 else 128
DAMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
WDIR = sys.argv[5] if len(sys.argv) > 5 else CAL
dev = "cuda"
FAM = [("attn_qkv", r"qkv_proj"), ("attn_o", r"o_proj"), ("indexer", r"indexer"), ("hc_down", r"input_mix_weight_down"),
       ("hc_up", r"input_mix_weight_up"), ("shared", r"shared_expert"), ("ple", r"ple")]


def fam_of(k):
    for f, rx in FAM:
        if re.search(rx, k):
            return f
    return "other"


@torch.no_grad()
def gptq(W, H, sc, block=128):
    """W [N, K] float, H [K, K], sc [N, K] per-column scales (fixed). Returns int codes [N, K]."""
    N, K = W.shape
    W = W.clone()
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    perm = torch.argsort(torch.diag(H), descending=True)
    W, H, sc = W[:, perm], H[perm][:, perm], sc[:, perm]
    H += DAMP * torch.mean(torch.diag(H)) * torch.eye(K, device=W.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w, s = W1[:, i], sc[:, i1 + i]
            q = torch.clamp(torch.round(w / s), -128, 127)
            Q[:, i1 + i] = q
            e = (w - q * s) / Hi[i, i]
            W1[:, i:] -= e.unsqueeze(1) * Hi[i, i:].unsqueeze(0)
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    inv = torch.argsort(perm)
    return Q[:, inv]


names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(WDIR, "*.W.pt")))
print(f"{len(names)} modules, group {GROUP}, damp {DAMP}", flush=True)
codes, stats = {}, defaultdict(lambda: [0.0, 0.0, 0, 0])
t0 = time.time()
for j, name in enumerate(names):
    W = torch.load(os.path.join(WDIR, name + ".W.pt")).to(dev)
    chunks = sorted(glob.glob(os.path.join(CAL, name + ".[0-9]*.pt")), key=lambda p: int(p.rsplit(".", 2)[1]))
    if not chunks:
        print("  no rows for", name, flush=True)
        continue
    X = torch.cat([torch.load(c) for c in chunks]).to(dev).float()
    g = torch.Generator(device="cpu").manual_seed(0)
    idx = torch.randperm(X.shape[0], generator=g).to(dev)
    nf = int(0.8 * X.shape[0])
    Xf, Xh = X[idx[:nf]], X[idx[nf:]]
    H = Xf.T @ Xf / nf
    N, K = W.shape
    q_rtn, s, gg = opt_fp8.quant_int8(W, GROUP)
    G = K if gg == -1 else gg
    sc = s.float().repeat_interleave(G, dim=1)                       # [N, K]
    q_g = gptq(W.float(), H, sc)
    ref = Xh @ W.float().T
    e_rtn = float((Xh @ (q_rtn.float() * sc).T - ref).norm() / ref.norm())
    e_g = float((Xh @ (q_g * sc).T - ref).norm() / ref.norm())
    f = fam_of(name)
    st = stats[f]; st[0] += e_rtn ** 2; st[1] += e_g ** 2; st[2] += 1; st[3] += N * K
    codes[name] = q_g.to(torch.int8).cpu()
    if j % 40 == 0 or e_g > e_rtn:
        print(f"  [{j:3d}] {name} {tuple(W.shape)} rows {X.shape[0]}: held-out rel.err RTN {e_rtn:.5f} GPTQ {e_g:.5f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    del W, X, Xf, Xh, H, q_g, ref
    torch.cuda.empty_cache()
torch.save(codes, OUT)
print("saved", len(codes), "modules to", OUT, f"({time.time() - t0:.0f}s)", flush=True)
print("held-out output error per family (RMS over modules): RTN -> GPTQ", flush=True)
for f, (a, b, n, ne) in stats.items():
    print(f"  {f:9s} {int(n):3d} modules {ne*2/1e6:7.0f} MB bf16: {(a/n)**0.5:.5f} -> {(b/n)**0.5:.5f}  ({(a/b)**0.5:.2f}x)",
          flush=True)

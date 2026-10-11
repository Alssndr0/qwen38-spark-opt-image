"""OPT: two-stage draft head for the MTP drafter (output-exact: rejection sampling keeps the target's distribution).

The drafter reads the full NVFP4 lm_head (248320 x 2560, ~357 MB) on every draft pass. Here a draft pass instead
  1. projects the hidden state onto a basis P [H, r] fitted on drafter hidden states (scripts/fit_draft_basis.py),
  2. scores the whole vocabulary with U = W P [V, r] (fp8 rows with a per-row scale, ~64 MB at r = 256),
  3. keeps the top-k candidates and computes their exact logits from the raw NVFP4 rows (k x 1.4 KB),
  4. returns full-vocabulary logits with -inf outside the candidates.
Offline on 2048 held-out drafter states (rank 256, k 2048): 99.91 % of the draft distribution's mass and 100 % of
the argmaxes fall inside the candidates. The draft q is renormalized over them; a target token outside cannot be
drafted (acceptance only).

Enable: OPT_DRAFT_HEAD_BASIS=/path/basis.pt (a dict {"P": [H, r] float tensor}); OPT_DRAFT_HEAD_K (default 2048).
OPT_DRAFT_HEAD_SCORER=fp4 stores U as NVFP4 (Marlin W4A16, 32 MB at r = 256) instead of FP8: same held-out coverage
(99.634 % multilingual, scripts/draft_head_fp4.py), scorer 284 -> 154 us, draft pass 482 -> 377 us at M = 1.
"""
import json
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

log = init_logger("vllm.opt_draft_head")
_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _quant_nvfp4(U):
    """U [N, K] float -> (packed u8 [N, K/2], fp8 group scales [N, K/16], global scale); dequant = e2m1 * s * g."""
    N, K = U.shape
    G = U.view(N, K // 16, 16)
    amax = G.abs().amax(-1)
    g = float(amax.max()) / (448.0 * 6.0)
    s = (amax / 6.0 / g).clamp(min=2 ** -9, max=448.0).to(torch.float8_e4m3fn)
    x = G / (s.float() * g).unsqueeze(-1)
    mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=U.device)
    idx = torch.bucketize(x.abs().clamp(max=6.0), mid)
    code = (idx | ((x < 0) & (idx > 0)).to(idx.dtype) * 8).to(torch.uint8).view(N, K)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), s.contiguous(), g


@triton.jit
def _cand_logits_kernel(h_ptr, cand_ptr, pk_ptr, sc_ptr, lut_ptr, out_ptr, gscale, K, stride_h, stride_c, stride_o,
                        H: tl.constexpr, BLOCK_C: tl.constexpr):
    """out[m, j] = h[m] . dequant(W[cand[m, j]]) for one row m and BLOCK_C candidates per program."""
    m = tl.program_id(0)
    jb = tl.program_id(1)
    offs_c = jb * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < K
    rows = tl.load(cand_ptr + m * stride_c + offs_c, mask=mask_c, other=0).to(tl.int64)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for g in range(0, H // 16):                       # one NVFP4 group of 16 values (8 packed bytes, 1 fp8 scale)
        offs_b = tl.arange(0, 8)
        pk = tl.load(pk_ptr + rows[:, None] * (H // 2) + g * 8 + offs_b[None, :], mask=mask_c[:, None], other=0)
        lo = tl.load(lut_ptr + (pk & 0x0F).to(tl.int32))
        hi = tl.load(lut_ptr + (pk >> 4).to(tl.int32))
        sc = tl.load(sc_ptr + rows * (H // 16) + g, mask=mask_c, other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        h_lo = tl.load(h_ptr + m * stride_h + g * 16 + 2 * offs_b).to(tl.float32)
        h_hi = tl.load(h_ptr + m * stride_h + g * 16 + 2 * offs_b + 1).to(tl.float32)
        acc += tl.sum(lo * h_lo[None, :] + hi * h_hi[None, :], axis=1) * sc
    tl.store(out_ptr + m * stride_o + offs_c, acc * gscale, mask=mask_c)


@triton.jit
def _cand_logits_int4_kernel(h_ptr, cand_ptr, pk_ptr, s_ptr, out_ptr, zero, K, stride_h, stride_c, stride_o,
                             H: tl.constexpr, G: tl.constexpr, BLOCK_C: tl.constexpr):
    """INT4 (AutoGPTQ symmetric, repacked row-major): out[m, j] = h[m] . ((code - zero) * scale[group])."""
    m = tl.program_id(0)
    jb = tl.program_id(1)
    offs_c = jb * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < K
    rows = tl.load(cand_ptr + m * stride_c + offs_c, mask=mask_c, other=0).to(tl.int64)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for g in range(0, H // 16):                       # 16 values (8 packed bytes) at a time; G is a multiple of 16
        offs_b = tl.arange(0, 8)
        pk = tl.load(pk_ptr + rows[:, None] * (H // 2) + g * 8 + offs_b[None, :], mask=mask_c[:, None], other=0)
        lo = (pk & 0x0F).to(tl.float32) - zero
        hi = (pk >> 4).to(tl.float32) - zero
        sc = tl.load(s_ptr + rows * (H // G) + (g * 16) // G, mask=mask_c, other=0).to(tl.float32)
        h_lo = tl.load(h_ptr + m * stride_h + g * 16 + 2 * offs_b).to(tl.float32)
        h_hi = tl.load(h_ptr + m * stride_h + g * 16 + 2 * offs_b + 1).to(tl.float32)
        acc += tl.sum(lo * h_lo[None, :] + hi * h_hi[None, :], axis=1) * sc
    tl.store(out_ptr + m * stride_o + offs_c, acc, mask=mask_c)


class DraftHead:
    def __init__(self, model_dir: str, basis_path: str, device, k: int):
        from safetensors import safe_open
        wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        if "lm_head.qweight" in wm:
            # INT4-AutoRound: AutoGPTQ v1 layout (qweight [H/8, V] int32, 8 codes along H per word, low nibble
            # first; scales [H/G, V] fp16; symmetric, stored zero = zero - 1). Repacked row-major for the candidate
            # kernel: pk [V, H/2] u8 (even column in the low nibble), s4 [V, H/G] fp16.
            self.fmt = "int4"
            f = safe_open(os.path.join(model_dir, wm["lm_head.qweight"]), framework="pt", device="cpu")
            qw = f.get_tensor("lm_head.qweight").to(device)
            sc = f.get_tensor("lm_head.scales").to(device)
            qz = f.get_tensor("lm_head.qzeros")
            zn = int((qz.view(-1)[0].item() & 0xF)) + 1
            pat = sum((zn - 1) << (4 * j) for j in range(8))
            pat = pat - (1 << 32) if pat >= (1 << 31) else pat
            assert bool((qz == pat).all()), "draft head: int4 zero points are not uniform (asymmetric checkpoint)"
            self.zero = float(zn)
            H8, V = qw.shape
            self.H, self.V = H8 * 8, V
            self.G = self.H // sc.shape[0]
            shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
            self.pk = torch.empty(V, self.H // 2, dtype=torch.uint8, device=device)
            for a in range(0, V, 16384):
                b = min(a + 16384, V)
                codes = ((qw[:, a:b].unsqueeze(1) >> shifts[None, :, None]) & 0xF).reshape(self.H, b - a).t()
                self.pk[a:b] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
            self.s4 = sc.t().contiguous()                                                           # [V, H/G] fp16
            del qw, sc
        else:
            self.fmt = "nvfp4"
            f = safe_open(os.path.join(model_dir, wm["lm_head.weight"]), framework="pt", device="cpu")
            self.pk = f.get_tensor("lm_head.weight").to(device)                                     # [V, H/2] u8
            self.sc = f.get_tensor("lm_head.weight_scale").view(torch.uint8).to(device).view(torch.float8_e4m3fn)
            self.g = float(f.get_tensor("lm_head.weight_scale_2").float())
            self.V, self.H = self.pk.shape[0], self.pk.shape[1] * 2
            self.lut = torch.tensor(_LUT, dtype=torch.float32, device=device)
        P = torch.load(basis_path)["P"].float().to(device)                                          # [H, r]
        self.P = P.to(torch.bfloat16)
        r = P.shape[1]
        # scorer U = W P [V, r] as a Marlin weight: FP8 per-row (W8A16) or NVFP4 (W4A16); one GEMV reads it per pass
        self.fp4 = os.environ.get("OPT_DRAFT_HEAD_SCORER", "fp8") == "fp4"
        U = torch.empty(self.V, r, dtype=torch.float32, device=device)
        for a in range(0, self.V, 16384):
            U[a:a + 16384] = self.dequant(a, min(a + 16384, self.V)) @ P
        holder = torch.nn.Module()
        holder.output_size_per_partition, holder.input_size_per_partition = self.V, r
        if self.fp4:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin
            pk4, s4, g4 = _quant_nvfp4(U)
            holder.weight = torch.nn.Parameter(pk4, requires_grad=False)
            holder.weight_scale = torch.nn.Parameter(s4, requires_grad=False)
            holder.weight_global_scale = torch.nn.Parameter(torch.tensor(g4, dtype=torch.float32, device=device),
                                                            requires_grad=False)
            holder.params_dtype = torch.bfloat16
            prepare_fp4_layer_for_marlin(holder)
        else:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
            sa = U.abs().amax(1).clamp(min=1e-12) / 448.0
            holder.weight = torch.nn.Parameter((U / sa[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn),
                                               requires_grad=False)
            holder.weight_scale = torch.nn.Parameter(sa, requires_grad=False)
            holder.orig_dtype = torch.bfloat16
            prepare_fp8_layer_for_marlin(holder, size_k_first=False)
        del U
        self.scorer = holder
        self.r = r
        self.k = k
        self.buf = None
        log.info("opt draft head: %s head, rank %d, %d candidates, %s scorer %.0f MB + raw head %.0f MB", self.fmt,
                 r, k, "nvfp4" if self.fp4 else "fp8", self.V * r / (2e6 if self.fp4 else 1e6),
                 (self.pk.numel() + (self.s4.numel() * 2 if self.fmt == "int4" else self.sc.numel())) / 1e6)

    def dequant(self, a: int, b: int) -> torch.Tensor:
        """Rows a..b of the vocabulary head as float32 [b - a, H]."""
        p = self.pk[a:b]
        codes = torch.stack(((p & 0x0F), (p >> 4)), dim=-1).view(p.shape[0], -1)
        if self.fmt == "int4":
            w = (codes.float() - self.zero).view(b - a, -1, self.G) * self.s4[a:b].float().unsqueeze(-1)
        else:
            w = self.lut[codes.long()].view(b - a, -1, 16) * self.sc[a:b].float().unsqueeze(-1) * self.g
        return w.reshape(b - a, -1)

    def logits(self, h: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        n = h.shape[0]
        z = h.to(torch.bfloat16) @ self.P                                                            # [n, r]
        sc = self.scorer
        if self.fp4:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
            scores = apply_fp4_marlin_linear(z, sc.weight, sc.weight_scale, sc.weight_global_scale, sc.workspace,
                                             self.V, self.r)                                             # [n, V]
        else:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
            scores = apply_fp8_marlin_linear(z, sc.weight, sc.weight_scale, sc.workspace, self.V, self.r, None)
        cand = scores.topk(self.k, dim=-1, sorted=False).indices.to(torch.int32).contiguous()      # [n, k]
        cl = torch.empty(n, self.k, dtype=torch.float32, device=h.device)
        hc = h.contiguous()
        BLOCK_C = 64
        if self.fmt == "int4":
            _cand_logits_int4_kernel[(n, triton.cdiv(self.k, BLOCK_C))](
                hc, cand, self.pk, self.s4, cl, self.zero, self.k, hc.stride(0), cand.stride(0), cl.stride(0),
                H=self.H, G=self.G, BLOCK_C=BLOCK_C)
        else:
            _cand_logits_kernel[(n, triton.cdiv(self.k, BLOCK_C))](
                hc, cand, self.pk, self.sc.view(torch.uint8), self.lut, cl, self.g, self.k, hc.stride(0),
                cand.stride(0), cl.stride(0), H=self.H, BLOCK_C=BLOCK_C)
        if scale != 1.0:
            cl = cl * scale
        if self.buf is None or self.buf.shape[0] < n:
            self.buf = torch.empty(max(n, 128), self.V, dtype=torch.bfloat16, device=h.device)
        out = self.buf[:n]
        out.fill_(float("-inf"))
        out.scatter_(1, cand.long(), cl.to(out.dtype))
        return out

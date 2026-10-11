"""OPT: convert bf16 linear layers to FP8 (e4m3, one scale per output channel) served by vLLM's Marlin FP8 W8A16
kernel. Decode on GB10 is memory-bound, so halving the weight bytes of the large bf16 projections cuts their time.

Used for the MTP drafter (exact: the verifier's rejection sampling keeps the target's distribution whatever the
draft weights are) and, optionally, for the target's bf16 projections (changes the model's numerics slightly).

fmt="int8" stores the same bytes as symmetric INT8 with one bf16 scale per `group` input values (or per row with
group=-1), served by Marlin's uint8b128 W8A16 kernel. INT8 with small groups is several times more accurate than
FP8 e4m3, whose 3 mantissa bits cost ~3.6 % RMS relative error on every weight whatever the scaling.
"""
import torch
from torch import nn

from vllm.logger import init_logger

log = init_logger("vllm.opt_fp8")


class _Fp8MarlinMethod:
    """Stand-in quant method: weight already repacked for Marlin FP8; process_weights_after_loading is a no-op."""

    def __init__(self, n, k):
        self.n, self.k = n, k

    def process_weights_after_loading(self, layer):
        return

    def apply(self, layer, x, bias=None):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
        return apply_fp8_marlin_linear(x, layer.weight, layer.weight_scale, layer.workspace, self.n, self.k, bias)


class _Int8MarlinMethod:
    """Stand-in quant method: INT8 (uint8b128) weight already packed for Marlin GPTQ; no-op after loading."""

    def __init__(self, n, k):
        self.n, self.k = n, k

    def process_weights_after_loading(self, layer):
        return

    def apply(self, layer, x, bias=None):
        from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear
        from vllm.scalar_type import scalar_types
        return apply_gptq_marlin_linear(x, layer.weight, layer.weight_scale, layer.weight_zp, layer.workspace,
                                        scalar_types.uint8b128, self.n, self.k, bias=bias)


def int8_group(K, group):
    """Largest Marlin-supported group <= `group` that divides K (-1 = per row)."""
    if group <= 0:
        return -1
    for g in (group, 128, 64, 32):
        if g <= group and K % g == 0:
            return g
    return -1


def quant_int8(w, group, q=None):
    """w [N, K] -> (q [N, K] int in [-128, 127], s [N, K/g] bf16, g). Scales are rounded to bf16 BEFORE the codes
    are chosen (the kernel multiplies by the bf16 scale), all in fp32. `q` = codes chosen elsewhere (e.g. GPTQ)."""
    N, K = w.shape
    g = int8_group(K, group)
    G = K if g == -1 else g
    x = w.float().view(N, K // G, G)
    s = (x.abs().amax(-1) / 127.0).clamp(min=1e-10).to(torch.bfloat16)
    if q is None:
        q = (x / s.float().unsqueeze(-1)).round().clamp(-128, 127).view(N, K)
    return q.to(torch.int32), s, g


def pack_int8_marlin(q, s, g):
    """codes q [N, K] in [-128, 127], scales s [N, K/G] -> Marlin GPTQ uint8b128 weight + permuted scales."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_padded_nk, marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm, marlin_weights
    N, K = q.shape
    # Marlin needs (n % 64, k % 128) or (n % 128, k % 64): zero-pad codes and scales (padded scales are 0, so the
    # padded region contributes nothing); apply_gptq_marlin_linear pads x and slices the output back.
    pn, pk = marlin_padded_nk(N, K, g)
    qk = torch.nn.functional.pad(q + 128, (0, pk - K, 0, pn - N), value=128).t().contiguous()   # [pk, pn] uint
    st = s.t().contiguous()                                                                    # [K/G, N]
    rows = pk // g if g > 0 else 1
    st = torch.nn.functional.pad(st, (0, pn - N, 0, rows - st.shape[0]))
    packed = marlin_weights(qk, pk, pn, 8, get_weight_perm(8))
    sp = marlin_permute_scales(st, pk, pn, g)
    return packed, sp


def _bf16_linear(mod):
    from vllm.model_executor.layers.linear import LinearBase
    w = getattr(mod, "weight", None)
    return (isinstance(mod, LinearBase) and isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2
            and not isinstance(getattr(mod, "quant_method", None), (_Fp8MarlinMethod, _Int8MarlinMethod)))


def convert(model: nn.Module, include=None, exclude=("router", "lm_head", "embed"),
            exclude_suffix=(".gate", "shared_expert_gate"), min_bytes=1 << 20, tag="", fmt="fp8", group=-1,
            codes=None) -> int:
    """Convert every bf16 LinearBase in `model` whose name matches `include` (substrings; None = all) and none of
    `exclude`, with at least `min_bytes` of weights. Returns the number of bytes saved.
    fmt "fp8" (e4m3 per row) or "int8" (per `group` input values); `codes` = {module name: int8 codes [N, K]}
    computed offline (GPTQ) to use instead of round-to-nearest."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
    if fmt == "int8":
        return _convert_int8(model, include, exclude, exclude_suffix, min_bytes, tag, group, codes or {})

    saved, done, errs = 0, 0, []
    for name, mod in list(model.named_modules()):
        if not _bf16_linear(mod):
            continue
        if include is not None and not any(s in name for s in include):
            continue
        if any(s in name for s in exclude) or name.endswith(tuple(exclude_suffix)):
            continue
        w = mod.weight.data
        N, K = w.shape
        if w.numel() * 2 < min_bytes or K % 16:
            continue
        amax = w.float().abs().amax(dim=1)
        scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))   # all-zero (padding) rows: any scale
        q = (w.float() / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        # one-time accuracy check on a random activation (relative error of the output)
        x = torch.randn(4, K, device=w.device, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w).float()
        holder = nn.Module()
        holder.weight = nn.Parameter(q, requires_grad=False)
        holder.weight_scale = nn.Parameter(scale.float(), requires_grad=False)
        holder.output_size_per_partition, holder.input_size_per_partition = N, K
        holder.orig_dtype = torch.bfloat16
        meth = _Fp8MarlinMethod(N, K)
        try:
            prepare_fp8_layer_for_marlin(holder, size_k_first=False)
            out = meth.apply(holder, x).float()
            err = ((out - ref).norm() / ref.norm().clamp(min=1e-12)).item()
        except Exception as e:  # noqa: BLE001  (a shape Marlin cannot take: keep bf16)
            log.warning("opt fp8: %s %s not converted (%r)", name, tuple(w.shape), e)
            continue
        errs.append(err)
        if not err < 0.05:
            log.warning("opt fp8: %s %s rel.err %.4f — kept bf16", name, tuple(w.shape), err)
            continue
        mod.weight = holder.weight
        mod.weight_scale = holder.weight_scale
        mod.workspace = holder.workspace
        mod.quant_method = meth
        saved += N * K
        done += 1
        log.debug("opt fp8%s: %s %s rel.err %.4f", tag, name, (N, K), err)
    torch.cuda.empty_cache()
    log.info("opt fp8%s: %d linears converted, %.1f MB of weights saved, output rel.err max %.4f mean %.4f", tag, done,
             saved / 1e6, max(errs) if errs else 0.0, sum(errs) / len(errs) if errs else 0.0)
    return saved


def _convert_int8(model, include, exclude, exclude_suffix, min_bytes, tag, group, codes):
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    saved, done, errs, gptq = 0, 0, [], 0
    ws = None
    for name, mod in list(model.named_modules()):
        if not _bf16_linear(mod):
            continue
        if include is not None and not any(s in name for s in include):
            continue
        if any(s in name for s in exclude) or name.endswith(tuple(exclude_suffix)):
            continue
        w = mod.weight.data
        N, K = w.shape
        if w.numel() * 2 < min_bytes or K % 16 or not torch.isfinite(w).all():
            if not torch.isfinite(w).all():
                log.warning("opt int8: %s %s has non-finite weights — kept bf16", name, tuple(w.shape))
            continue
        cq = codes.get(name)
        q, s, g = quant_int8(w, group, q=cq.to(w.device) if cq is not None else None)
        gptq += cq is not None
        if ws is None:
            ws = marlin_make_workspace_new(w.device)
        x = torch.randn(4, K, device=w.device, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w).float()
        holder = nn.Module()
        meth = _Int8MarlinMethod(N, K)
        try:
            packed, sp = pack_int8_marlin(q, s, g)
            holder.weight = nn.Parameter(packed, requires_grad=False)
            holder.weight_scale = nn.Parameter(sp, requires_grad=False)
            holder.weight_zp = torch.empty(0, dtype=torch.int, device=w.device)
            holder.workspace = ws
            out = meth.apply(holder, x).float()
            err = ((out - ref).norm() / ref.norm().clamp(min=1e-12)).item()
        except Exception as e:  # noqa: BLE001
            log.warning("opt int8: %s %s not converted (%r)", name, tuple(w.shape), e)
            continue
        errs.append(err)
        if not err < 0.05:
            log.warning("opt int8: %s %s rel.err %.4f — kept bf16", name, tuple(w.shape), err)
            continue
        mod.weight = holder.weight
        mod.weight_scale = holder.weight_scale
        mod.weight_zp = holder.weight_zp
        mod.workspace = ws
        mod.quant_method = meth
        saved += N * K - s.numel() * 2
        done += 1
        log.debug("opt int8%s: %s %s g%d rel.err %.5f", tag, name, (N, K), g, err)
    torch.cuda.empty_cache()
    log.info("opt int8%s: %d linears converted (group %d, %d with offline codes), %.1f MB saved, output rel.err "
             "max %.5f mean %.5f", tag, done, group, gptq, saved / 1e6, max(errs) if errs else 0.0,
             sum(errs) / len(errs) if errs else 0.0)
    return saved


class _Recorder:
    """Calibration (OPT_CALIB_DIR): wraps a module's quant_method and saves randomly sampled input rows of eager
    (prefill) calls to OUT/<module>.<n>.pt for offline GPTQ. Armed only while OUT/GO exists (so the startup profile
    run's dummy tokens are not recorded); each call's rows go straight to disk (host memory is tight)."""
    _armed = [False, 0.0]

    def __init__(self, inner, name, out, rows, per_call):
        self.inner, self.name, self.out, self.rows, self.per_call = inner, name, out, rows, per_call
        self.n, self.k = 0, 0

    @classmethod
    def armed(cls, out):
        import os, time
        now = time.monotonic()
        if now - cls._armed[1] > 1.0:
            cls._armed[0], cls._armed[1] = os.path.exists(os.path.join(out, "GO")), now
        return cls._armed[0]

    def process_weights_after_loading(self, layer):
        return self.inner.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        if self.n < self.rows and x.shape[0] >= 64 and not torch.cuda.is_current_stream_capturing() \
                and self.armed(self.out):
            x2 = x.reshape(-1, x.shape[-1])
            take = min(self.per_call, self.rows - self.n, x2.shape[0])
            idx = torch.randperm(x2.shape[0], device=x2.device)[:take]
            torch.save(x2.index_select(0, idx).to("cpu"), f"{self.out}/{self.name}.{self.k}.pt")
            self.n += take
            self.k += 1
        return self.inner.apply(layer, x, bias)


def install_recorders(model, out, include=None, exclude=("router", "lm_head", "embed", "visual"),
                      exclude_suffix=(".gate", "shared_expert_gate"), min_bytes=1 << 20, rows=4096, per_call=160):
    import os
    os.makedirs(out, exist_ok=True)
    n = 0
    for name, mod in list(model.named_modules()):
        if not _bf16_linear(mod):
            continue
        if include is not None and not any(s in name for s in include):
            continue
        if any(s in name for s in exclude) or name.endswith(tuple(exclude_suffix)):
            continue
        if mod.weight.numel() * 2 < min_bytes:
            continue
        mod.quant_method = _Recorder(mod.quant_method, name, out, rows, per_call)
        torch.save(mod.weight.data.to("cpu"), f"{out}/{name}.W.pt")      # the weight in vLLM's merged layout
        n += 1
    log.info("opt calib: recording inputs of %d linears into %s (armed while %s/GO exists)", n, out, out)
    return n

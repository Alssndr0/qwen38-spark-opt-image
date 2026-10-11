# Decode optimizations for myllmbox/qwen38-flash-next-vllm:v5.2 (see ../README.md).
# Every speed change is opt-in by environment variable or by mtp_depth mode "opt"; with none of them set this image
# serves like v5.2. Always on: the three output-exact fixes (patches rejection_sampler, recoverssm_gdn, routed_experts).
FROM myllmbox/qwen38-flash-next-vllm:v5.2
ARG SITE=/usr/local/lib/python3.12/dist-packages
COPY patches/ /opt/opt-patches/
RUN set -e; cd ${SITE}; for p in /opt/opt-patches/*.patch; do patch -s -p1 --no-backup-if-mismatch < "$p"; done
COPY files/opt_fp8.py files/opt_draft_head.py ${SITE}/
COPY files/opt_ple_async.c /opt/mbx/src/opt_ple_async.c
RUN gcc -O2 -Wall -shared -fPIC -o /opt/mbx/lib/libopt_ple_async.so /opt/mbx/src/opt_ple_async.c \
        -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -lcuda -ldl -lpthread \
 && python3 -m py_compile ${SITE}/mbx_depth.py ${SITE}/opt_fp8.py ${SITE}/opt_draft_head.py \
        ${SITE}/vllm/models/qwen4_exp/nvidia/ngram_embedding.py ${SITE}/vllm/models/qwen4_exp/nvidia/mtp.py \
        ${SITE}/vllm/models/qwen4_exp/nvidia/model.py ${SITE}/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py \
        ${SITE}/vllm/model_executor/layers/mamba/gdn/recoverssm_gdn.py \
        ${SITE}/vllm/model_executor/layers/fused_moe/routed_experts.py

// opt_ple_async: non-blocking PLE table reads for the MBX NVMe table reader.
//
// Every CUDA call is made by the main (Python) thread, as ordinary stream work:
//   submit:  D2H ids -> pinned ids buffer; cuStreamWriteValue32(ids_flag = seq)            (compute stream)
//   gather:  cuStreamWaitValue32(data_flag >= seq); H2D pinned rows -> output buffers      (compute stream)
// The worker thread makes NO CUDA calls: it polls ids_flag in pinned host memory, reads the rows through
// libmbx_ple_nvme (mbx_t_read) into pinned buffers in request order, fences, and stores data_flag = seq.
// The GPU therefore only ever waits on a plain CPU store, never on another CUDA API call (a first design that let
// the worker enqueue its own copies deadlocked when the main thread entered an implicitly synchronizing driver call,
// e.g. cuModuleLoadData for a freshly JIT-compiled Triton kernel, while the GPU was parked on the wait).
//
// build: gcc -O2 -shared -fPIC -o libopt_ple_async.so opt_ple_async.c -I/usr/local/cuda/include -lcuda -ldl -lpthread
#include <cuda.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RING 16

typedef int (*read_fn)(int, const void *, int64_t, void *, void *, int32_t, int32_t, int);

struct job {
    int n;
    uint32_t seq;
};

static struct {
    int ok;
    int64_t *pin_ids;                 // written by the GPU (D2H), read by the worker
    uint8_t *pin_pk, *pin_sc;         // written by the worker, read by the GPU (H2D)
    volatile uint32_t *ids_flag;      // written by the GPU (cuStreamWriteValue32), polled by the worker
    volatile uint32_t *data_flag;     // written by the worker, waited on by the GPU (cuStreamWaitValue32)
    int64_t *uniq, *order;
    int32_t *inv;
    uint8_t *tmp_pk, *tmp_sc;
    int fd, cp, cs, threads, max_n;
    read_fn rd;
    pthread_t th;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct job q[RING];
    int head, tail, count;
    volatile int failed;
    volatile long jobs_done;
    volatile int stage;               // 0 idle, 1 waiting for ids, 2 reading, 3 publishing
    volatile uint32_t cur_seq;
    volatile double stage_t;
    double read_us_total;
} G;

static double now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static const int64_t *g_sort_keys;
static int cmp_idx(const void *a, const void *b) {
    int64_t x = g_sort_keys[*(const int64_t *)a], y = g_sort_keys[*(const int64_t *)b];
    return (x > y) - (x < y);
}

static int seq_ge(uint32_t a, uint32_t b) { return (int32_t)(a - b) >= 0; }

static void do_job(struct job *j) {
    int n = j->n;
    G.cur_seq = j->seq;
    G.stage = 1;
    G.stage_t = now_us();
    // wait until the GPU has copied this job's ids into pinned memory (spin briefly, then yield)
    int spins = 0;
    while (!seq_ge(*G.ids_flag, j->seq)) {
        if (++spins > 200) {
            struct timespec t = {0, 20000};   // 20 us: the GPU may be up to a step behind the CPU
            nanosleep(&t, NULL);
        }
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    G.stage = 2;
    double t0 = now_us();
    G.stage_t = t0;
    for (int i = 0; i < n; i++) G.order[i] = i;
    g_sort_keys = G.pin_ids;
    qsort(G.order, n, sizeof(int64_t), cmp_idx);
    int nu = 0;
    for (int i = 0; i < n; i++) {
        int64_t id = G.pin_ids[G.order[i]];
        if (nu == 0 || G.uniq[nu - 1] != id) G.uniq[nu++] = id;
        G.inv[G.order[i]] = nu - 1;
    }
    int rc = G.rd(G.fd, G.uniq, nu, G.tmp_pk, G.tmp_sc, G.cp, G.cs, G.threads);
    G.stage = 3;
    if (rc) {  // never leave the GPU waiting: zero rows, flag the failure for the Python side
        G.failed = 1;
        memset(G.pin_pk, 0, (size_t)n * G.cp);
        if (G.cs) memset(G.pin_sc, 0, (size_t)n * G.cs);
    } else {
        for (int i = 0; i < n; i++) {
            memcpy(G.pin_pk + (size_t)i * G.cp, G.tmp_pk + (size_t)G.inv[i] * G.cp, G.cp);
            if (G.cs) memcpy(G.pin_sc + (size_t)i * G.cs, G.tmp_sc + (size_t)G.inv[i] * G.cs, G.cs);
        }
    }
    G.read_us_total += now_us() - t0;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    *G.data_flag = j->seq;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    G.jobs_done++;
    G.stage = 0;
}

static void *worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&G.mu);
        while (G.count == 0) pthread_cond_wait(&G.cv, &G.mu);
        struct job j = G.q[G.head];
        pthread_mutex_unlock(&G.mu);
        do_job(&j);
        pthread_mutex_lock(&G.mu);
        G.head = (G.head + 1) % RING;
        G.count--;
        pthread_cond_broadcast(&G.cv);
        pthread_mutex_unlock(&G.mu);
    }
    return NULL;
}

static void *watchdog(void *arg) {
    (void)arg;
    for (;;) {
        struct timespec t = {1, 0};
        nanosleep(&t, NULL);
        if (G.stage && now_us() - G.stage_t > 5e6) {
            fprintf(stderr, "OPT PLE async WATCHDOG: job seq %u in stage %d for %.1f s (ids_flag %u, data_flag %u, "
                    "queued %d, done %ld)\n", G.cur_seq, G.stage, (now_us() - G.stage_t) / 1e6, *G.ids_flag,
                    *G.data_flag, G.count, G.jobs_done);
            fflush(stderr);
        }
    }
    return NULL;
}

// Buffers are pinned host memory allocated by the caller (torch pin_memory=True: mapped, so the GPU can copy to/from
// them and wait on the flags). Called once, before any job.
int opt_ple_init(const char *lib_path, int fd, int cp, int cs, int max_n, int threads, uint64_t pin_ids,
                 uint64_t pin_pk, uint64_t pin_sc, uint64_t ids_flag, uint64_t data_flag) {
    memset(&G, 0, sizeof(G));
    void *h = dlopen(lib_path, RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen(lib_path, RTLD_NOW);
    if (!h) return 1;
    G.rd = (read_fn)dlsym(h, "mbx_t_read");
    if (!G.rd) return 2;
    G.fd = fd; G.cp = cp; G.cs = cs; G.max_n = max_n; G.threads = threads;
    G.pin_ids = (int64_t *)(uintptr_t)pin_ids;
    G.pin_pk = (uint8_t *)(uintptr_t)pin_pk;
    G.pin_sc = (uint8_t *)(uintptr_t)pin_sc;
    G.ids_flag = (volatile uint32_t *)(uintptr_t)ids_flag;
    G.data_flag = (volatile uint32_t *)(uintptr_t)data_flag;
    G.uniq = malloc((size_t)max_n * 8); G.inv = malloc((size_t)max_n * 4); G.order = malloc((size_t)max_n * 8);
    G.tmp_pk = malloc((size_t)max_n * cp); G.tmp_sc = malloc((size_t)max_n * (cs ? cs : 1));
    if (!G.uniq || !G.inv || !G.order || !G.tmp_pk || !G.tmp_sc) return 7;
    pthread_mutex_init(&G.mu, NULL);
    pthread_cond_init(&G.cv, NULL);
    if (pthread_create(&G.th, NULL, worker, NULL)) return 8;
    pthread_t wd;
    pthread_create(&wd, NULL, watchdog, NULL);
    G.ok = 1;
    return 0;
}

// Main thread, after it enqueued the ids D2H copy on `stream`: publish ids_flag = seq from the stream, queue the job.
int opt_ple_submit(uint64_t stream, int n, uint32_t seq) {
    if (!G.ok || n > G.max_n) return 1;
    if (cuStreamWriteValue32((CUstream)(uintptr_t)stream, (CUdeviceptr)(uintptr_t)G.ids_flag, seq,
                             CU_STREAM_WRITE_VALUE_DEFAULT) != CUDA_SUCCESS)
        return 2;
    pthread_mutex_lock(&G.mu);
    while (G.count == RING) pthread_cond_wait(&G.cv, &G.mu);
    G.q[G.tail] = (struct job){n, seq};
    G.tail = (G.tail + 1) % RING;
    G.count++;
    pthread_cond_broadcast(&G.cv);
    pthread_mutex_unlock(&G.mu);
    return 0;
}

// Main thread: make the stream wait (on the device) for the job's rows.
int opt_ple_wait(uint64_t stream, uint32_t seq) {
    if (!G.ok) return 1;
    return cuStreamWaitValue32((CUstream)(uintptr_t)stream, (CUdeviceptr)(uintptr_t)G.data_flag, seq,
                               CU_STREAM_WAIT_VALUE_GEQ) != CUDA_SUCCESS;
}

// Block until every queued job is done (before the synchronous reader touches the table library on this thread).
void opt_ple_drain(void) {
    if (!G.ok) return;
    pthread_mutex_lock(&G.mu);
    while (G.count > 0) pthread_cond_wait(&G.cv, &G.mu);
    pthread_mutex_unlock(&G.mu);
}

int opt_ple_failed(void) { return G.failed; }
long opt_ple_jobs(void) { return G.jobs_done; }
double opt_ple_read_us(void) { return G.read_us_total; }

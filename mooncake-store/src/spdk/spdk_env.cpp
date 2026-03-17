#include "spdk/spdk_env.h"

#include <glog/logging.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

extern "C" {
#include "spdk/bdev.h"
#include "spdk/cpuset.h"
#include "spdk/env.h"
#include "spdk/event.h"
#include "spdk/log.h"
#include "spdk/thread.h"
#include "bdev/malloc/bdev_malloc.h"
}

#define SPDK_THREAD(p) (static_cast<struct spdk_thread *>(p))
#define SPDK_DESC(p) (static_cast<struct spdk_bdev_desc *>(p))
#define SPDK_CHAN(p) (static_cast<struct spdk_io_channel *>(p))

// ---------------------------------------------------------------------------
// File-local helpers
// ---------------------------------------------------------------------------
void bdev_init_complete_cb(void *, int) {}

static void bdev_event_cb(enum spdk_bdev_event_type type,
                           struct spdk_bdev *, void *) {
    LOG(INFO) << "SpdkEnv: bdev event type=" << static_cast<int>(type);
}

// ---------------------------------------------------------------------------
// submit_single_io — core I/O submission logic, always runs on a reactor.
// ---------------------------------------------------------------------------
static void submit_single_io(mooncake::SpdkIoRequest *req) {
    auto &env = mooncake::SpdkEnv::Instance();

    spdk_bdev_io_completion_cb done = [](struct spdk_bdev_io *bio, bool ok,
                                         void *arg) {
        auto *r = static_cast<mooncake::SpdkIoRequest *>(arg);
        spdk_bdev_free_io(bio);
        r->success = ok;
        r->completed.store(true, std::memory_order_release);
    };

    if (req->op == mooncake::SpdkIoRequest::WRITE) {
        if (req->src_iov && req->src_iovcnt > 0) {
            char *dst = static_cast<char *>(req->buf);
            size_t copied = 0;
            for (int i = 0; i < req->src_iovcnt; ++i) {
                std::memcpy(dst, req->src_iov[i].iov_base,
                            req->src_iov[i].iov_len);
                dst += req->src_iov[i].iov_len;
                copied += req->src_iov[i].iov_len;
            }
            if (req->nbytes > copied)
                std::memset(dst, 0, req->nbytes - copied);
        } else if (req->src_data) {
            std::memcpy(req->buf, req->src_data, req->src_len);
            if (req->nbytes > req->src_len)
                std::memset(static_cast<char *>(req->buf) + req->src_len, 0,
                            req->nbytes - req->src_len);
        }
    }

    auto *ch = SPDK_CHAN(req->_io_channel);
    int rc;
    if (req->op == mooncake::SpdkIoRequest::WRITE) {
        rc = spdk_bdev_write(SPDK_DESC(env.bdev_desc_), ch,
                             req->buf, req->offset, req->nbytes, done, req);
    } else {
        rc = spdk_bdev_read(SPDK_DESC(env.bdev_desc_), ch,
                            req->buf, req->offset, req->nbytes, done, req);
    }
    if (rc != 0) {
        LOG(ERROR) << "SpdkEnv: bdev I/O submit failed rc=" << rc;
        req->success = false;
        req->completed.store(true, std::memory_order_release);
    }
}

// ---------------------------------------------------------------------------
// execute_io_cb — single request callback (for SubmitIoAsync).
// ---------------------------------------------------------------------------
void execute_io_cb(void *ctx) {
    submit_single_io(static_cast<mooncake::SpdkIoRequest *>(ctx));
}

// ---------------------------------------------------------------------------
// execute_io_batch_cb — batch callback: walk linked list, submit each I/O.
// ---------------------------------------------------------------------------
void execute_io_batch_cb(void *ctx) {
    auto *req = static_cast<mooncake::SpdkIoRequest *>(ctx);
    while (req) {
        auto *next = req->_next_batch;
        submit_single_io(req);
        req = next;
    }
}

// ---------------------------------------------------------------------------
// init_reactor_cb — spdk_event callback, runs on target reactor's OS thread.
// Creates an spdk_thread pinned to this core and obtains an io_channel.
// ---------------------------------------------------------------------------
void init_reactor_cb(void *arg1, void *arg2) {
    auto *env = static_cast<mooncake::SpdkEnv *>(arg1);
    int idx = static_cast<int>(reinterpret_cast<intptr_t>(arg2));

    struct spdk_cpuset cpumask;
    spdk_cpuset_zero(&cpumask);
    spdk_cpuset_set_cpu(&cpumask, spdk_env_get_current_core(), true);

    char name[32];
    std::snprintf(name, sizeof(name), "mc_io_%d", idx);
    struct spdk_thread *th = spdk_thread_create(name, &cpumask);
    if (!th) {
        LOG(ERROR) << "SpdkEnv: spdk_thread_create failed for reactor " << idx;
        signal_init_done(env, -1);
        return;
    }
    spdk_set_thread(th);

    auto *ch = spdk_bdev_get_io_channel(SPDK_DESC(env->bdev_desc_));
    if (!ch) {
        LOG(ERROR) << "SpdkEnv: io_channel failed for reactor " << idx;
        signal_init_done(env, -1);
        return;
    }

    env->reactors_[idx].spdk_thread = th;
    env->reactors_[idx].io_channel = ch;
    env->reactors_[idx].core_id = spdk_env_get_current_core();

    LOG(INFO) << "SpdkEnv: reactor " << idx
              << " ready on core " << spdk_env_get_current_core();

    int ready = env->reactors_ready_.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (ready == env->num_reactors_) {
        signal_init_done(env, 0);
    }
}

// ---------------------------------------------------------------------------
// cleanup_reactor_cb — releases io_channel + destroys spdk_thread on each
// non-master reactor. Runs via spdk_event on the target reactor.
// ---------------------------------------------------------------------------
void cleanup_reactor_cb(void *arg1, void *arg2) {
    auto *env = static_cast<mooncake::SpdkEnv *>(arg1);
    int idx = static_cast<int>(reinterpret_cast<intptr_t>(arg2));

    auto &r = env->reactors_[idx];
    if (r.io_channel) {
        spdk_set_thread(SPDK_THREAD(r.spdk_thread));
        spdk_put_io_channel(SPDK_CHAN(r.io_channel));
        r.io_channel = nullptr;
    }
    // For non-master reactors, mark the thread for exit.
    // Don't destroy — the reactor framework cleans up during spdk_app_fini.
    // Reactor 0's app thread is framework-owned and must not be exited here.
    if (idx > 0 && r.spdk_thread) {
        spdk_set_thread(SPDK_THREAD(r.spdk_thread));
        spdk_thread_exit(SPDK_THREAD(r.spdk_thread));
        r.spdk_thread = nullptr;
    }

    int done = env->reactors_ready_.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (done == 0) {
        if (env->bdev_desc_) {
            spdk_bdev_close(SPDK_DESC(env->bdev_desc_));
            env->bdev_desc_ = nullptr;
        }
        spdk_app_stop(0);
    }
}

// ---------------------------------------------------------------------------
// signal_init_done — wake the main thread waiting in Init().
// ---------------------------------------------------------------------------
void signal_init_done(mooncake::SpdkEnv *env, int rc) {
    env->init_result_ = rc;
    {
        std::lock_guard<std::mutex> lk(env->init_mutex_);
        env->init_complete_ = true;
    }
    env->init_cv_.notify_one();
    if (rc != 0) spdk_app_stop(-1);
}

// ---------------------------------------------------------------------------
// app_start_cb — runs on reactor 0 (master). Opens bdev, sets up reactor 0's
// channel, then sends events to other reactors to create their channels.
// ---------------------------------------------------------------------------
void app_start_cb(void *ctx) {
    auto *env = static_cast<mooncake::SpdkEnv *>(ctx);
    auto &cfg = env->config_;

    // Open bdev (shared across all reactors).
    struct spdk_bdev_desc *desc = nullptr;
    int rc = spdk_bdev_open_ext(cfg.bdev_name.c_str(), true,
                                bdev_event_cb, nullptr, &desc);
    if (rc != 0) {
        LOG(ERROR) << "SpdkEnv: spdk_bdev_open_ext('" << cfg.bdev_name
                   << "') failed rc=" << rc;
        signal_init_done(env, rc);
        return;
    }
    env->bdev_desc_ = desc;

    struct spdk_bdev *bd = spdk_bdev_desc_get_bdev(desc);
    env->bdev_ = bd;
    env->block_size_ = spdk_bdev_get_block_size(bd);
    env->bdev_size_ =
        static_cast<uint64_t>(spdk_bdev_get_num_blocks(bd)) * env->block_size_;

    LOG(INFO) << "SpdkEnv: bdev '" << cfg.bdev_name
              << "' opened — block_size=" << env->block_size_
              << " total_size=" << env->bdev_size_;

    // Enumerate cores and allocate reactor contexts.
    std::vector<uint32_t> cores;
    uint32_t core;
    SPDK_ENV_FOREACH_CORE(core) { cores.push_back(core); }
    env->num_reactors_ = static_cast<int>(cores.size());
    env->reactors_.resize(env->num_reactors_);
    env->reactors_ready_.store(0, std::memory_order_relaxed);

    LOG(INFO) << "SpdkEnv: " << env->num_reactors_ << " reactor core(s) available";

    if (env->num_reactors_ == 0) {
        signal_init_done(env, -1);
        return;
    }

    // Reactor 0 (master): set up channel directly — we're already on it.
    {
        struct spdk_io_channel *ch = spdk_bdev_get_io_channel(desc);
        if (!ch) {
            LOG(ERROR) << "SpdkEnv: io_channel failed for master reactor";
            signal_init_done(env, -1);
            return;
        }
        env->reactors_[0].spdk_thread = spdk_get_thread();
        env->reactors_[0].io_channel = ch;
        env->reactors_[0].core_id = cores[0];

        LOG(INFO) << "SpdkEnv: reactor 0 ready on core " << cores[0];

        int ready = env->reactors_ready_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (ready == env->num_reactors_) {
            signal_init_done(env, 0);
            return;
        }
    }

    // Send events to other reactors to create their spdk_thread + io_channel.
    for (int i = 1; i < env->num_reactors_; ++i) {
        struct spdk_event *ev = spdk_event_allocate(
            cores[i], init_reactor_cb, env,
            reinterpret_cast<void *>(static_cast<intptr_t>(i)));
        spdk_event_call(ev);
    }
}

void open_bdev_cb(void *) { /* unused, kept for link compatibility */ }

// ===========================================================================
namespace mooncake {

SpdkEnv &SpdkEnv::Instance() {
    static SpdkEnv instance;
    return instance;
}

SpdkEnv::~SpdkEnv() {
    if (initialized_.load(std::memory_order_acquire)) {
        Shutdown();
    }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
int SpdkEnv::Init(const SpdkEnvConfig &config) {
    if (initialized_.load(std::memory_order_acquire)) {
        LOG(WARNING) << "SpdkEnv: already initialized";
        return 0;
    }

    config_ = config;
    init_complete_ = false;
    init_result_ = -1;

    std::string json_path;
    if (config_.use_malloc_bdev) {
        json_path = "/tmp/mooncake_spdk_bdev.json";
        FILE *f = fopen(json_path.c_str(), "w");
        if (f) {
            fprintf(f,
                "{\n"
                "  \"subsystems\": [{\n"
                "    \"subsystem\": \"bdev\",\n"
                "    \"config\": [{\n"
                "      \"method\": \"bdev_malloc_create\",\n"
                "      \"params\": {\n"
                "        \"name\": \"%s\",\n"
                "        \"num_blocks\": %" PRIu64 ",\n"
                "        \"block_size\": %u\n"
                "      }\n"
                "    }]\n"
                "  }]\n"
                "}\n",
                config_.bdev_name.c_str(),
                config_.malloc_num_blocks,
                config_.malloc_block_size);
            fclose(f);
        }
    }

    reactor_thread_ = std::thread([this, json_path]() {
        struct spdk_app_opts opts = {};
        spdk_app_opts_init(&opts, sizeof(opts));
        opts.name = config_.name.c_str();
        opts.shutdown_cb = nullptr;
        opts.reactor_mask = config_.reactor_mask.c_str();
        if (!json_path.empty()) {
            opts.json_config_file = json_path.c_str();
        }

        int rc = spdk_app_start(&opts, app_start_cb, this);
        if (rc != 0) {
            LOG(ERROR) << "SpdkEnv: spdk_app_start returned rc=" << rc;
        }
        spdk_app_fini();

        if (!init_complete_) {
            std::lock_guard<std::mutex> lk(init_mutex_);
            init_result_ = rc;
            init_complete_ = true;
            init_cv_.notify_one();
        }
    });

    {
        std::unique_lock<std::mutex> lk(init_mutex_);
        init_cv_.wait(lk, [this] { return init_complete_; });
    }

    if (init_result_ != 0) {
        LOG(ERROR) << "SpdkEnv: init failed rc=" << init_result_;
        if (reactor_thread_.joinable()) reactor_thread_.join();
        return init_result_;
    }

    initialized_.store(true, std::memory_order_release);
    LOG(INFO) << "SpdkEnv: initialization complete — "
              << num_reactors_ << " reactor(s)";
    return 0;
}

// ---------------------------------------------------------------------------
// Shutdown — release io_channels on all reactors, close bdev, stop app.
// ---------------------------------------------------------------------------
void SpdkEnv::CleanupOnSpdkThread(void *ctx) {
    auto *self = static_cast<SpdkEnv *>(ctx);
    if (self->bdev_desc_) {
        spdk_bdev_close(SPDK_DESC(self->bdev_desc_));
        self->bdev_desc_ = nullptr;
    }
    spdk_app_stop(0);
}

void SpdkEnv::Shutdown() {
    if (!initialized_.load(std::memory_order_acquire)) return;
    initialized_.store(false, std::memory_order_release);

    // Enumerate reactor cores so we can send events to each one.
    // Build a local copy of core IDs since reactors_ will be modified.
    std::vector<uint32_t> cores(num_reactors_);
    for (int i = 0; i < num_reactors_; ++i)
        cores[i] = reactors_[i].core_id;

    // Reset the barrier counter: we'll count down from num_reactors_.
    reactors_ready_.store(num_reactors_, std::memory_order_release);

    for (int i = 0; i < num_reactors_; ++i) {
        struct spdk_event *ev = spdk_event_allocate(
            cores[i], cleanup_reactor_cb, this,
            reinterpret_cast<void *>(static_cast<intptr_t>(i)));
        spdk_event_call(ev);
    }

    if (reactor_thread_.joinable()) reactor_thread_.join();
    LOG(INFO) << "SpdkEnv: shutdown complete";
}

void SpdkEnv::CleanupThreadLocalCtx() { /* no per-thread state */ }

// ---------------------------------------------------------------------------
// I/O submission — per-IO round-robin across reactors.
// ---------------------------------------------------------------------------
int SpdkEnv::SubmitIoAsync(SpdkIoRequest *req) {
    if (!initialized_.load(std::memory_order_acquire) || num_reactors_ == 0)
        return -EINVAL;

    int rid = static_cast<int>(
        next_reactor_.fetch_add(1, std::memory_order_relaxed) %
        static_cast<uint64_t>(num_reactors_));
    auto &r = reactors_[rid];

    req->_io_channel = r.io_channel;
    req->completed.store(false, std::memory_order_release);
    req->success = false;

    return spdk_thread_send_msg(SPDK_THREAD(r.spdk_thread),
                                execute_io_cb, req);
}

int SpdkEnv::SubmitIoBatchAsync(SpdkIoRequest **reqs, int count) {
    if (!initialized_.load(std::memory_order_acquire) || num_reactors_ == 0)
        return -EINVAL;
    if (count <= 0) return 0;

    constexpr int kMaxReactors = 64;
    SpdkIoRequest *heads[kMaxReactors] = {};
    SpdkIoRequest *tails[kMaxReactors] = {};

    uint64_t base = next_reactor_.fetch_add(
        static_cast<uint64_t>(count), std::memory_order_relaxed);

    for (int i = 0; i < count; ++i) {
        int rid = static_cast<int>((base + static_cast<uint64_t>(i)) %
                                   static_cast<uint64_t>(num_reactors_));
        auto *req = reqs[i];
        req->_io_channel = reactors_[rid].io_channel;
        req->completed.store(false, std::memory_order_release);
        req->success = false;
        req->_next_batch = nullptr;

        if (!heads[rid]) {
            heads[rid] = tails[rid] = req;
        } else {
            tails[rid]->_next_batch = req;
            tails[rid] = req;
        }
    }

    for (int r = 0; r < num_reactors_; ++r) {
        if (!heads[r]) continue;
        spdk_thread_send_msg(SPDK_THREAD(reactors_[r].spdk_thread),
                             execute_io_batch_cb, heads[r]);
    }
    return 0;
}

int SpdkEnv::PollIo() {
    return 0;
}

void SpdkEnv::SubmitIo(SpdkIoRequest *req) {
    int rc = SubmitIoAsync(req);
    if (rc != 0) {
        req->success = false;
        req->completed.store(true, std::memory_order_release);
        return;
    }
    while (!req->completed.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#else
        std::this_thread::yield();
#endif
    }
}

// ---------------------------------------------------------------------------
// DMA buffer helpers
// ---------------------------------------------------------------------------
void *SpdkEnv::DmaMalloc(size_t size, size_t align) {
    return spdk_dma_malloc(size, align, nullptr);
}

void SpdkEnv::DmaFree(void *buf) { spdk_dma_free(buf); }

}  // namespace mooncake

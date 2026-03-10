#include "spdk/spdk_env.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstring>

extern "C" {
#include "spdk/bdev.h"
#include "spdk/env.h"
#include "spdk/log.h"
#include "spdk/thread.h"
#include "bdev/malloc/bdev_malloc.h"
}

// Convenience casts — the header stores SPDK handles as void* to avoid
// leaking C enum/struct types into the C++ namespace.
#define SPDK_THREAD(p) (static_cast<struct spdk_thread *>(p))
#define SPDK_DESC(p) (static_cast<struct spdk_bdev_desc *>(p))
#define SPDK_CHAN(p) (static_cast<struct spdk_io_channel *>(p))
#define SPDK_BDEV(p) (static_cast<struct spdk_bdev *>(p))

// ---------------------------------------------------------------------------
// File-local SPDK callbacks (pure C linkage, outside mooncake namespace)
// ---------------------------------------------------------------------------
void bdev_init_complete_cb(void *ctx, int rc) {
    auto *env = static_cast<mooncake::SpdkEnv *>(ctx);
    env->bdev_init_rc_ = rc;
    env->bdev_init_done_.store(true, std::memory_order_release);
}

static void bdev_event_cb(enum spdk_bdev_event_type type,  // NOLINT
                           struct spdk_bdev * /*bdev*/, void * /*ctx*/) {
    LOG(INFO) << "SpdkEnv: bdev event type=" << static_cast<int>(type);
}

void execute_io_cb(void *ctx) {
    auto *req = static_cast<mooncake::SpdkIoRequest *>(ctx);
    auto &env = mooncake::SpdkEnv::Instance();

    spdk_bdev_io_completion_cb done = [](struct spdk_bdev_io *bio, bool ok,
                                         void *arg) {
        auto *r = static_cast<mooncake::SpdkIoRequest *>(arg);
        spdk_bdev_free_io(bio);
        {
            std::lock_guard<std::mutex> lk(r->mtx);
            r->success = ok;
            r->completed = true;
        }
        r->cv.notify_one();
    };

    int rc;
    if (req->op == mooncake::SpdkIoRequest::WRITE) {
        rc = spdk_bdev_write(SPDK_DESC(env.bdev_desc_),
                             SPDK_CHAN(env.io_channel_), req->buf, req->offset,
                             req->nbytes, done, req);
    } else {
        rc = spdk_bdev_read(SPDK_DESC(env.bdev_desc_),
                            SPDK_CHAN(env.io_channel_), req->buf, req->offset,
                            req->nbytes, done, req);
    }
    if (rc != 0) {
        LOG(ERROR) << "SpdkEnv: bdev "
                   << (req->op == mooncake::SpdkIoRequest::WRITE ? "write"
                                                                  : "read")
                   << " submit failed rc=" << rc;
        std::lock_guard<std::mutex> lk(req->mtx);
        req->success = false;
        req->completed = true;
        req->cv.notify_one();
    }
}

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
// Reactor thread
// ---------------------------------------------------------------------------
void SpdkEnv::ReactorLoop() {
    struct spdk_thread *thr = spdk_thread_create("mooncake_io", nullptr);
    if (!thr) {
        LOG(ERROR) << "SpdkEnv: spdk_thread_create failed";
        std::lock_guard<std::mutex> lk(init_mutex_);
        init_result_ = -1;
        init_complete_ = true;
        init_cv_.notify_one();
        return;
    }
    spdk_thread_ = thr;
    spdk_set_thread(thr);

    int rc = InitOnSpdkThread();
    {
        std::lock_guard<std::mutex> lk(init_mutex_);
        init_result_ = rc;
        init_complete_ = true;
    }
    init_cv_.notify_one();

    if (rc != 0) {
        spdk_thread_exit(thr);
        while (!spdk_thread_is_exited(thr)) spdk_thread_poll(thr, 0, 0);
        spdk_thread_destroy(thr);
        spdk_thread_ = nullptr;
        return;
    }

    while (!should_stop_.load(std::memory_order_acquire)) {
        spdk_thread_poll(thr, 0, 0);
    }

    if (io_channel_) {
        spdk_put_io_channel(SPDK_CHAN(io_channel_));
        io_channel_ = nullptr;
    }
    if (bdev_desc_) {
        spdk_bdev_close(SPDK_DESC(bdev_desc_));
        bdev_desc_ = nullptr;
    }

    std::atomic<bool> fin{false};
    spdk_bdev_finish(
        [](void *ctx) {
            static_cast<std::atomic<bool> *>(ctx)->store(
                true, std::memory_order_release);
        },
        &fin);
    while (!fin.load(std::memory_order_acquire)) spdk_thread_poll(thr, 0, 0);

    spdk_thread_exit(thr);
    while (!spdk_thread_is_exited(thr)) spdk_thread_poll(thr, 0, 0);
    spdk_thread_destroy(thr);
    spdk_thread_ = nullptr;
}

// ---------------------------------------------------------------------------
// Bdev init (runs on SPDK thread)
// ---------------------------------------------------------------------------
int SpdkEnv::InitOnSpdkThread() {
    spdk_bdev_initialize(bdev_init_complete_cb, this);

    auto *thr = SPDK_THREAD(spdk_thread_);
    while (!bdev_init_done_.load(std::memory_order_acquire)) {
        spdk_thread_poll(thr, 0, 0);
    }
    if (bdev_init_rc_ != 0) {
        LOG(ERROR) << "SpdkEnv: spdk_bdev_initialize failed rc="
                   << bdev_init_rc_;
        return bdev_init_rc_;
    }

    if (config_.use_malloc_bdev) {
        struct malloc_bdev_opts mopts = {};
        mopts.name = const_cast<char *>(config_.bdev_name.c_str());
        mopts.num_blocks = config_.malloc_num_blocks;
        mopts.block_size = config_.malloc_block_size;

        struct spdk_bdev *mbdev = nullptr;
        int mrc = create_malloc_disk(&mbdev, &mopts);
        if (mrc != 0) {
            LOG(ERROR) << "SpdkEnv: create_malloc_disk failed rc=" << mrc;
            return mrc;
        }
        LOG(INFO) << "SpdkEnv: created Malloc bdev '" << config_.bdev_name
                  << "' (" << config_.malloc_num_blocks << " x "
                  << config_.malloc_block_size << " bytes)";
    }

    struct spdk_bdev_desc *desc = nullptr;
    int rc = spdk_bdev_open_ext(config_.bdev_name.c_str(), true, bdev_event_cb,
                                nullptr, &desc);
    if (rc != 0) {
        LOG(ERROR) << "SpdkEnv: spdk_bdev_open_ext('" << config_.bdev_name
                   << "') failed rc=" << rc;
        return rc;
    }
    bdev_desc_ = desc;

    struct spdk_bdev *bd = spdk_bdev_desc_get_bdev(desc);
    bdev_ = bd;
    block_size_ = spdk_bdev_get_block_size(bd);
    bdev_size_ =
        static_cast<uint64_t>(spdk_bdev_get_num_blocks(bd)) * block_size_;

    struct spdk_io_channel *ch = spdk_bdev_get_io_channel(desc);
    if (!ch) {
        LOG(ERROR) << "SpdkEnv: spdk_bdev_get_io_channel failed";
        spdk_bdev_close(desc);
        bdev_desc_ = nullptr;
        return -1;
    }
    io_channel_ = ch;

    LOG(INFO) << "SpdkEnv: bdev '" << config_.bdev_name
              << "' opened — block_size=" << block_size_
              << " total_size=" << bdev_size_;
    return 0;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
int SpdkEnv::Init(const SpdkEnvConfig &config) {
    if (initialized_.load(std::memory_order_acquire)) {
        LOG(WARNING) << "SpdkEnv: already initialized";
        return 0;
    }

    config_ = config;

    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.name = config_.name.c_str();
    opts.shm_id = config_.shm_id;
    opts.mem_size = -1;

    LOG(INFO) << "SpdkEnv: env_opts mem_size=" << opts.mem_size
              << " shm_id=" << opts.shm_id;

    int rc = spdk_env_init(&opts);
    if (rc != 0) {
        LOG(ERROR) << "SpdkEnv: spdk_env_init failed rc=" << rc;
        return rc;
    }

    void *probe = spdk_dma_malloc(4096, 4096, nullptr);
    LOG(INFO) << "SpdkEnv: DMA probe alloc "
              << (probe ? "OK" : "FAILED — DPDK has no hugepage memory");
    if (probe) spdk_dma_free(probe);

    rc = spdk_thread_lib_init_ext(nullptr, nullptr, nullptr, 0, 8192);
    if (rc != 0) {
        LOG(ERROR) << "SpdkEnv: spdk_thread_lib_init_ext failed rc=" << rc;
        return rc;
    }

    init_complete_ = false;
    reactor_thread_ = std::thread(&SpdkEnv::ReactorLoop, this);

    {
        std::unique_lock<std::mutex> lk(init_mutex_);
        init_cv_.wait(lk, [this] { return init_complete_; });
    }

    if (init_result_ != 0) {
        LOG(ERROR) << "SpdkEnv: reactor init failed rc=" << init_result_;
        should_stop_.store(true, std::memory_order_release);
        if (reactor_thread_.joinable()) reactor_thread_.join();
        return init_result_;
    }

    initialized_.store(true, std::memory_order_release);
    LOG(INFO) << "SpdkEnv: initialization complete";
    return 0;
}

void SpdkEnv::Shutdown() {
    if (!initialized_.load(std::memory_order_acquire)) return;

    should_stop_.store(true, std::memory_order_release);
    if (reactor_thread_.joinable()) reactor_thread_.join();

    spdk_thread_lib_fini();
    initialized_.store(false, std::memory_order_release);
    LOG(INFO) << "SpdkEnv: shutdown complete";
}

// ---------------------------------------------------------------------------
// I/O submission
// ---------------------------------------------------------------------------
void SpdkEnv::SubmitIo(SpdkIoRequest *req) {
    if (!initialized_.load(std::memory_order_acquire) || !spdk_thread_) {
        std::lock_guard<std::mutex> lk(req->mtx);
        req->success = false;
        req->completed = true;
        return;
    }
    req->completed = false;
    req->success = false;
    spdk_thread_send_msg(SPDK_THREAD(spdk_thread_), execute_io_cb, req);

    std::unique_lock<std::mutex> lk(req->mtx);
    req->cv.wait(lk, [req] { return req->completed; });
}

// ---------------------------------------------------------------------------
// DMA buffer helpers
// ---------------------------------------------------------------------------
void *SpdkEnv::DmaMalloc(size_t size, size_t align) {
    return spdk_dma_malloc(size, align, nullptr);
}

void SpdkEnv::DmaFree(void *buf) { spdk_dma_free(buf); }

}  // namespace mooncake

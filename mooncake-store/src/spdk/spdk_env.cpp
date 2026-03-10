#include "spdk/spdk_env.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>

extern "C" {
#include "spdk/bdev.h"
#include "spdk/env.h"
#include "spdk/log.h"
#include "spdk/thread.h"
#include "bdev/malloc/bdev_malloc.h"
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
// Reactor thread: owns the single spdk_thread and polls it continuously.
// ---------------------------------------------------------------------------
void SpdkEnv::ReactorLoop() {
    spdk_thread_ = spdk_thread_create("mooncake_io", nullptr);
    if (!spdk_thread_) {
        LOG(ERROR) << "SpdkEnv: spdk_thread_create failed";
        std::lock_guard<std::mutex> lk(init_mutex_);
        init_result_ = -1;
        init_complete_ = true;
        init_cv_.notify_one();
        return;
    }
    spdk_set_thread(spdk_thread_);

    int rc = InitOnSpdkThread();
    {
        std::lock_guard<std::mutex> lk(init_mutex_);
        init_result_ = rc;
        init_complete_ = true;
    }
    init_cv_.notify_one();

    if (rc != 0) {
        spdk_thread_exit(spdk_thread_);
        while (!spdk_thread_is_exited(spdk_thread_)) {
            spdk_thread_poll(spdk_thread_, 0, 0);
        }
        spdk_thread_destroy(spdk_thread_);
        spdk_thread_ = nullptr;
        return;
    }

    while (!should_stop_.load(std::memory_order_acquire)) {
        spdk_thread_poll(spdk_thread_, 0, 0);
    }

    if (io_channel_) {
        spdk_put_io_channel(io_channel_);
        io_channel_ = nullptr;
    }
    if (bdev_desc_) {
        spdk_bdev_close(bdev_desc_);
        bdev_desc_ = nullptr;
    }

    std::atomic<bool> bdev_finish_done{false};
    spdk_bdev_finish(
        [](void *ctx) {
            static_cast<std::atomic<bool> *>(ctx)->store(
                true, std::memory_order_release);
        },
        &bdev_finish_done);
    while (!bdev_finish_done.load(std::memory_order_acquire)) {
        spdk_thread_poll(spdk_thread_, 0, 0);
    }

    spdk_thread_exit(spdk_thread_);
    while (!spdk_thread_is_exited(spdk_thread_)) {
        spdk_thread_poll(spdk_thread_, 0, 0);
    }
    spdk_thread_destroy(spdk_thread_);
    spdk_thread_ = nullptr;
}

// ---------------------------------------------------------------------------
// Bdev subsystem initialization (runs on SPDK thread).
// ---------------------------------------------------------------------------
void SpdkEnv::BdevInitComplete(void *ctx, int rc) {
    auto *env = static_cast<SpdkEnv *>(ctx);
    env->bdev_init_rc_ = rc;
    env->bdev_init_done_.store(true, std::memory_order_release);
}

void SpdkEnv::BdevEventCb(enum spdk_bdev_event_type type,
                           struct spdk_bdev * /*bdev*/, void * /*ctx*/) {
    LOG(INFO) << "SpdkEnv: bdev event type=" << static_cast<int>(type);
}

int SpdkEnv::InitOnSpdkThread() {
    spdk_bdev_initialize(BdevInitComplete, this);

    while (!bdev_init_done_.load(std::memory_order_acquire)) {
        spdk_thread_poll(spdk_thread_, 0, 0);
    }
    if (bdev_init_rc_ != 0) {
        LOG(ERROR) << "SpdkEnv: spdk_bdev_initialize failed rc="
                   << bdev_init_rc_;
        return bdev_init_rc_;
    }

    if (config_.use_malloc_bdev) {
        struct malloc_bdev_opts mopts;
        malloc_bdev_opts_init(&mopts);
        snprintf(mopts.name, sizeof(mopts.name), "%s",
                 config_.bdev_name.c_str());
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

    int rc =
        spdk_bdev_open_ext(config_.bdev_name.c_str(), true, BdevEventCb,
                           nullptr, &bdev_desc_);
    if (rc != 0) {
        LOG(ERROR) << "SpdkEnv: spdk_bdev_open_ext('" << config_.bdev_name
                   << "') failed rc=" << rc;
        return rc;
    }

    bdev_ = spdk_bdev_desc_get_bdev(bdev_desc_);
    block_size_ = spdk_bdev_get_block_size(bdev_);
    bdev_size_ =
        static_cast<uint64_t>(spdk_bdev_get_num_blocks(bdev_)) * block_size_;

    io_channel_ = spdk_bdev_get_io_channel(bdev_desc_);
    if (!io_channel_) {
        LOG(ERROR) << "SpdkEnv: spdk_bdev_get_io_channel failed";
        spdk_bdev_close(bdev_desc_);
        bdev_desc_ = nullptr;
        return -1;
    }

    LOG(INFO) << "SpdkEnv: bdev '" << config_.bdev_name << "' opened — block_size="
              << block_size_ << " total_size=" << bdev_size_;
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

    int rc = spdk_env_init(&opts);
    if (rc != 0) {
        LOG(ERROR) << "SpdkEnv: spdk_env_init failed rc=" << rc;
        return rc;
    }

    rc = spdk_thread_lib_init(nullptr, 0);
    if (rc != 0) {
        LOG(ERROR) << "SpdkEnv: spdk_thread_lib_init failed rc=" << rc;
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
// I/O submission: any thread → SPDK thread → block until done.
// ---------------------------------------------------------------------------
void SpdkEnv::ExecuteIoOnSpdkThread(void *ctx) {
    auto *req = static_cast<SpdkIoRequest *>(ctx);
    auto &env = SpdkEnv::Instance();

    spdk_bdev_io_completion_cb cb = IoComplete;
    int rc;
    if (req->op == SpdkIoRequest::WRITE) {
        rc = spdk_bdev_write(env.bdev_desc_, env.io_channel_, req->buf,
                             req->offset, req->nbytes, cb, req);
    } else {
        rc = spdk_bdev_read(env.bdev_desc_, env.io_channel_, req->buf,
                            req->offset, req->nbytes, cb, req);
    }
    if (rc != 0) {
        LOG(ERROR) << "SpdkEnv: bdev "
                   << (req->op == SpdkIoRequest::WRITE ? "write" : "read")
                   << " submit failed rc=" << rc;
        std::lock_guard<std::mutex> lk(req->mtx);
        req->success = false;
        req->completed = true;
        req->cv.notify_one();
    }
}

void SpdkEnv::IoComplete(struct spdk_bdev_io *bdev_io, bool success,
                          void *ctx) {
    auto *req = static_cast<SpdkIoRequest *>(ctx);
    spdk_bdev_free_io(bdev_io);
    {
        std::lock_guard<std::mutex> lk(req->mtx);
        req->success = success;
        req->completed = true;
    }
    req->cv.notify_one();
}

void SpdkEnv::SubmitIo(SpdkIoRequest *req) {
    if (!initialized_.load(std::memory_order_acquire) || !spdk_thread_) {
        std::lock_guard<std::mutex> lk(req->mtx);
        req->success = false;
        req->completed = true;
        return;
    }
    req->completed = false;
    req->success = false;
    spdk_thread_send_msg(spdk_thread_, ExecuteIoOnSpdkThread, req);

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

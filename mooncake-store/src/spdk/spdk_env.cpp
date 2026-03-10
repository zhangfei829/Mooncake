#include "spdk/spdk_env.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstring>

extern "C" {
#include "spdk/bdev.h"
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
// File-local callbacks (C linkage)
// ---------------------------------------------------------------------------
void bdev_init_complete_cb(void *, int) {}

static void bdev_event_cb(enum spdk_bdev_event_type type,
                           struct spdk_bdev *, void *) {
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
        LOG(ERROR) << "SpdkEnv: bdev I/O submit failed rc=" << rc;
        std::lock_guard<std::mutex> lk(req->mtx);
        req->success = false;
        req->completed = true;
        req->cv.notify_one();
    }
}

// ---------------------------------------------------------------------------
// spdk_app_start callback — runs on the SPDK reactor thread.
// Opens bdev, gets io_channel, signals the main thread.
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

void open_bdev_cb(void *ctx) {
    auto *env = static_cast<mooncake::SpdkEnv *>(ctx);
    auto &cfg = env->config_;

    struct spdk_bdev_desc *desc = nullptr;
    int rc = spdk_bdev_open_ext(cfg.bdev_name.c_str(), true,
                                bdev_event_cb, nullptr, &desc);
    if (rc != 0) {
        LOG(ERROR) << "SpdkEnv: spdk_bdev_open_ext failed rc=" << rc;
        signal_init_done(env, rc);
        return;
    }
    env->bdev_desc_ = desc;

    struct spdk_bdev *bd = spdk_bdev_desc_get_bdev(desc);
    env->bdev_ = bd;
    env->block_size_ = spdk_bdev_get_block_size(bd);
    env->bdev_size_ =
        static_cast<uint64_t>(spdk_bdev_get_num_blocks(bd)) * env->block_size_;

    struct spdk_io_channel *ch = spdk_bdev_get_io_channel(desc);
    if (!ch) {
        LOG(ERROR) << "SpdkEnv: spdk_bdev_get_io_channel failed";
        spdk_bdev_close(desc);
        env->bdev_desc_ = nullptr;
        signal_init_done(env, -1);
        return;
    }
    env->io_channel_ = ch;
    env->spdk_thread_ = spdk_get_thread();

    LOG(INFO) << "SpdkEnv: bdev '" << cfg.bdev_name
              << "' opened — block_size=" << env->block_size_
              << " total_size=" << env->bdev_size_;
    signal_init_done(env, 0);
}

void app_start_cb(void *ctx) {
    auto *env = static_cast<mooncake::SpdkEnv *>(ctx);
    auto &cfg = env->config_;

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

    struct spdk_io_channel *ch = spdk_bdev_get_io_channel(desc);
    if (!ch) {
        LOG(ERROR) << "SpdkEnv: spdk_bdev_get_io_channel failed";
        spdk_bdev_close(desc);
        env->bdev_desc_ = nullptr;
        signal_init_done(env, -1);
        return;
    }
    env->io_channel_ = ch;
    env->spdk_thread_ = spdk_get_thread();

    LOG(INFO) << "SpdkEnv: bdev '" << cfg.bdev_name
              << "' opened — block_size=" << env->block_size_
              << " total_size=" << env->bdev_size_;
    signal_init_done(env, 0);
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
// Public API
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
                "        \"num_blocks\": %lu,\n"
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
    LOG(INFO) << "SpdkEnv: initialization complete";
    return 0;
}

void SpdkEnv::Shutdown() {
    if (!initialized_.load(std::memory_order_acquire)) return;
    initialized_.store(false, std::memory_order_release);

    if (io_channel_) {
        spdk_put_io_channel(SPDK_CHAN(io_channel_));
        io_channel_ = nullptr;
    }
    if (bdev_desc_) {
        spdk_bdev_close(SPDK_DESC(bdev_desc_));
        bdev_desc_ = nullptr;
    }

    spdk_app_stop(0);
    if (reactor_thread_.joinable()) reactor_thread_.join();

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

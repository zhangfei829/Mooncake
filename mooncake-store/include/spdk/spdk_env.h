#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

void bdev_init_complete_cb(void *ctx, int rc);
void execute_io_cb(void *ctx);

namespace mooncake {

struct SpdkEnvConfig {
    std::string name = "mooncake_spdk";
    std::string bdev_name;
    int shm_id = -1;

    bool use_malloc_bdev = false;
    uint64_t malloc_num_blocks = 131072;
    uint32_t malloc_block_size = 4096;
};

struct SpdkIoRequest {
    enum Op { READ, WRITE };
    Op op;
    void *buf;
    uint64_t offset;
    uint64_t nbytes;

    std::mutex mtx;
    std::condition_variable cv;
    bool completed = false;
    bool success = false;
};

/// Singleton managing the SPDK environment lifecycle.
///
/// Initializes SPDK in embedded mode (no spdk_app_start): env + thread lib +
/// bdev subsystem, all driven by a dedicated reactor thread running
/// spdk_thread_poll. Any Mooncake thread can submit I/O through SubmitIo()
/// which bridges the async SPDK world back to a blocking call.
class SpdkEnv {
   public:
    static SpdkEnv &Instance();

    int Init(const SpdkEnvConfig &config);
    void Shutdown();

    bool IsInitialized() const {
        return initialized_.load(std::memory_order_acquire);
    }

    void SubmitIo(SpdkIoRequest *req);

    uint32_t GetBlockSize() const { return block_size_; }
    uint64_t GetBdevSize() const { return bdev_size_; }

    void *DmaMalloc(size_t size, size_t align = 4096);
    void DmaFree(void *buf);

   private:
    SpdkEnv() = default;
    ~SpdkEnv();
    SpdkEnv(const SpdkEnv &) = delete;
    SpdkEnv &operator=(const SpdkEnv &) = delete;

    void ReactorLoop();
    int InitOnSpdkThread();

    friend void ::bdev_init_complete_cb(void *, int);
    friend void ::execute_io_cb(void *);

    std::atomic<bool> initialized_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> bdev_init_done_{false};
    int bdev_init_rc_ = -1;

    std::thread reactor_thread_;

    // Opaque SPDK handles — stored as void* to avoid leaking C types
    // into C++ headers. Cast back in the .cpp file.
    void *spdk_thread_ = nullptr;
    void *bdev_ = nullptr;
    void *bdev_desc_ = nullptr;
    void *io_channel_ = nullptr;

    uint32_t block_size_ = 0;
    uint64_t bdev_size_ = 0;
    SpdkEnvConfig config_;

    std::mutex init_mutex_;
    std::condition_variable init_cv_;
    bool init_complete_ = false;
    int init_result_ = -1;
};

}  // namespace mooncake

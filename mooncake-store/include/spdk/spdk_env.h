#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/uio.h>

namespace mooncake { class SpdkEnv; }

void bdev_init_complete_cb(void *ctx, int rc);
void execute_io_cb(void *ctx);
void execute_io_batch_cb(void *ctx);
void app_start_cb(void *ctx);
void open_bdev_cb(void *ctx);
void signal_init_done(mooncake::SpdkEnv *env, int rc);
void init_reactor_cb(void *arg1, void *arg2);
void cleanup_reactor_cb(void *arg1, void *arg2);

namespace mooncake {

struct SpdkEnvConfig {
    std::string name = "mooncake_spdk";
    std::string bdev_name;
    int shm_id = -1;

    bool use_malloc_bdev = false;
    uint64_t malloc_num_blocks = 131072;
    uint32_t malloc_block_size = 4096;

    // Real NVMe: PCI BDF address (e.g. "0000:47:00.0").
    // When non-empty and use_malloc_bdev==false, an NVMe bdev is attached.
    // The resulting bdev name is "<nvme_ctrl_name>n1".
    std::string nvme_pci_addr;
    std::string nvme_ctrl_name = "NVMe0";

    std::string reactor_mask = "0x1";

    // Limit DPDK hugepage memory (MB). 0 = use all available hugepages.
    int mem_size_mb = 0;

    // Chunked DMA+memcpy pipeline for large values.
    // pipeline_chunk_kb: double-buffer chunk size (KB). 0 = use default (4096).
    // pipeline_threshold_kb: minimum aligned size to enable pipeline (KB).
    //   0 = use default (2×chunk, ensures at least 2 chunks for overlap).
    int pipeline_chunk_kb = 0;
    int pipeline_threshold_kb = 0;
};

struct SpdkIoRequest {
    enum Op { READ, WRITE };
    Op op;
    void *buf;
    uint64_t offset;
    uint64_t nbytes;

    std::atomic<bool> completed{false};
    bool success = false;

    void *_io_channel = nullptr;

    const void *src_data = nullptr;
    uint64_t src_len = 0;
    const struct iovec *src_iov = nullptr;
    int src_iovcnt = 0;

    const struct iovec *dst_iov = nullptr;
    int dst_iovcnt = 0;
    size_t dst_skip = 0;

    SpdkIoRequest *_next_batch = nullptr;
};

struct ReactorCtx {
    void *spdk_thread = nullptr;
    void *io_channel = nullptr;
    uint32_t core_id = 0;
};

class SpdkEnv {
   public:
    static SpdkEnv &Instance();

    int Init(const SpdkEnvConfig &config);
    void Shutdown();

    bool IsInitialized() const {
        return initialized_.load(std::memory_order_acquire);
    }

    void SubmitIo(SpdkIoRequest *req);
    int SubmitIoAsync(SpdkIoRequest *req);
    int SubmitIoBatchAsync(SpdkIoRequest **reqs, int count);
    int PollIo();
    void CleanupThreadLocalCtx();

    uint32_t GetBlockSize() const { return block_size_; }
    uint64_t GetBdevSize() const { return bdev_size_; }
    int GetNumReactors() const { return num_reactors_; }

    size_t GetPipelineChunk() const;
    size_t GetPipelineThreshold() const;
    static void SetPipelineParams(size_t threshold, size_t chunk);

    void *DmaMalloc(size_t size, size_t align = 4096);
    void DmaFree(void *buf);

    void *DmaPoolAlloc(size_t needed, size_t align = 4096);
    void DmaPoolFree(void *buf, size_t size);

    int DmaPoolAllocBatch(void **out_bufs, size_t needed, int count,
                          size_t align = 4096);
    void DmaPoolFreeBatch(void *const *bufs, size_t size, int count);
    void DmaPoolPrewarm(size_t buf_size, int count, size_t align = 4096);
    void DmaPoolDrain();

   private:
    SpdkEnv() = default;
    ~SpdkEnv();
    SpdkEnv(const SpdkEnv &) = delete;
    SpdkEnv &operator=(const SpdkEnv &) = delete;

    static void CleanupOnSpdkThread(void *ctx);

    friend void ::bdev_init_complete_cb(void *, int);
    friend void ::execute_io_cb(void *);
    friend void ::execute_io_batch_cb(void *);
    friend void ::app_start_cb(void *);
    friend void ::open_bdev_cb(void *);
    friend void ::signal_init_done(mooncake::SpdkEnv *, int);
    friend void ::init_reactor_cb(void *, void *);
    friend void ::cleanup_reactor_cb(void *, void *);

    std::atomic<bool> initialized_{false};

    std::thread reactor_thread_;

    void *bdev_ = nullptr;
    void *bdev_desc_ = nullptr;

    std::vector<ReactorCtx> reactors_;
    int num_reactors_ = 0;
    std::atomic<uint64_t> next_reactor_{0};
    std::atomic<int> reactors_ready_{0};

    uint32_t block_size_ = 0;
    uint64_t bdev_size_ = 0;
    SpdkEnvConfig config_;

    std::mutex init_mutex_;
    std::condition_variable init_cv_;
    bool init_complete_ = false;
    int init_result_ = -1;

    struct DmaPoolEntry { void *buf; size_t size; };
    std::mutex dma_pool_mutex_;
    std::vector<DmaPoolEntry> dma_pool_;
};

}  // namespace mooncake

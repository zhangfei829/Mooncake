// SPDK vs PosixFile performance benchmark.
//
// Measures: sequential bandwidth, random IOPS, per-op latency (p50/p99/p999),
//           and end-to-end OffsetAllocatorStorageBackend throughput.
//
// Default mode uses SPDK malloc bdev (pure memory), PosixFile on /dev/shm.
// For real NVMe, set --spdk_bdev_name and --posix_path to the same device.
//
// Requires: USE_SPDK build, hugepages, root.
// Run:  sudo ./spdk_bench [flags]

#ifdef USE_SPDK

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <linux/falloc.h>
#include <sys/uio.h>
#include <unistd.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "file_interface.h"
#include "spdk/spdk_env.h"
#include "storage_backend.h"

extern "C" {
#include "spdk/bdev.h"
#include "spdk/thread.h"
}

using namespace mooncake;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Flags
// ============================================================================

DEFINE_string(posix_path, "/dev/shm/spdk_bench.dat",
              "PosixFile data path (use /dev/shm for memory baseline, "
              "or a real block device for disk comparison)");
DEFINE_string(spdk_bdev_name, "BenchMalloc0",
              "SPDK bdev name (default: auto-created malloc bdev)");
DEFINE_uint64(spdk_malloc_mb, 1024,
              "Size of SPDK malloc bdev in MB (default: 1024)");
DEFINE_int32(iterations, 3,
             "Iterations per data point (min/max trimmed, report mean of rest)");
DEFINE_string(test, "all",
              "Which test to run: file_seq, file_rand, backend, all");
DEFINE_bool(verify, true, "Verify data correctness on first iteration");
DEFINE_uint64(backend_keys, 2000,
              "Number of keys for backend throughput test");
DEFINE_uint64(backend_value_kb, 128,
              "Value size in KB for backend test (default: 128)");
DEFINE_int32(threads, 1,
             "Thread count for Posix file-level I/O and backend tests (default: 1). "
             "Match to SPDK cores for fair comparison.");
DEFINE_int32(iodepth, 128,
             "I/O queue depth for SPDK async benchmarks (default: 128)");
DEFINE_int32(cores, 2,
             "Number of SPDK reactor cores (default: 2, sufficient for single NVMe)");
DEFINE_bool(profile, false,
            "Enable per-phase timing breakdown (pass --v=1 for backend detail)");
DEFINE_string(nvme_pci_addr, "",
              "NVMe PCI BDF address for real disk (e.g. 0000:47:00.0). "
              "When set, uses real NVMe bdev instead of malloc bdev.");
DEFINE_bool(posix_direct, false,
            "Use O_DIRECT for Posix file I/O (bypasses page cache, "
            "fairer comparison vs SPDK on real disk)");
DEFINE_string(posix_backend_dir, "",
              "Directory for Posix backend tests (default: /tmp). "
              "Set to a mount point on real NVMe for fair disk comparison.");
DEFINE_int32(mem_size_mb, 0,
             "Limit DPDK hugepage memory in MB (0 = use all available). "
             "Useful on machines with limited RAM.");
DEFINE_int32(pipeline_chunk_kb, 0,
             "Chunked DMA+memcpy pipeline chunk size in KB (0=default 4096). "
             "Use --test=pipeline_tune to find optimal for your hardware.");
DEFINE_int32(pipeline_threshold_kb, 0,
             "Min aligned entry size (KB) to enable pipeline (0=default 4096).");

// ============================================================================
// Helpers
// ============================================================================

static std::string BuildCoreMask(int num_cores) {
    if (num_cores <= 0) num_cores = 1;
    if (num_cores > 64) num_cores = 64;
    uint64_t mask = (num_cores >= 64)
                        ? ~uint64_t(0)
                        : ((uint64_t(1) << num_cores) - 1);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%" PRIx64, mask);
    return buf;
}

static std::string FormatSize(uint64_t bytes) {
    const char *units[] = {"B", "KB", "MB", "GB"};
    double sz = static_cast<double>(bytes);
    int idx = 0;
    while (sz >= 1024.0 && idx < 3) { sz /= 1024.0; ++idx; }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f %s", sz, units[idx]);
    return buf;
}

static std::string FormatBW(double bytes, double secs) {
    double mbps = (bytes / (1024.0 * 1024.0)) / secs;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MB/s", mbps);
    return buf;
}

// 4096-aligned buffer for O_DIRECT compatibility.
struct AlignedBuf {
    char *ptr = nullptr;
    size_t len = 0;

    explicit AlignedBuf(size_t n) : len(n) {
        size_t aligned_n = (n + 4095) & ~size_t(4095);
        ptr = static_cast<char *>(aligned_alloc(4096, aligned_n));
    }
    ~AlignedBuf() { free(ptr); }
    AlignedBuf(const AlignedBuf &) = delete;
    AlignedBuf &operator=(const AlignedBuf &) = delete;
    char *data() { return ptr; }
    size_t size() const { return len; }
};

static int OpenPosixFlags(bool create) {
    int flags = O_RDWR | O_CLOEXEC;
    if (create) flags |= O_CREAT | O_TRUNC;
    if (FLAGS_posix_direct) flags |= O_DIRECT;
    return flags;
}

static bool PreallocateFile(int fd, size_t size) {
    if (fallocate(fd, 0, 0, static_cast<off_t>(size)) == 0) return true;
    return ftruncate(fd, static_cast<off_t>(size)) == 0;
}

struct LatencyStats {
    std::vector<double> samples_us;

    void Add(double us) { samples_us.push_back(us); }

    void Sort() { std::sort(samples_us.begin(), samples_us.end()); }

    double Percentile(double p) const {
        if (samples_us.empty()) return 0;
        size_t idx = static_cast<size_t>(p / 100.0 * (samples_us.size() - 1));
        return samples_us[idx];
    }

    double Mean() const {
        if (samples_us.empty()) return 0;
        return std::accumulate(samples_us.begin(), samples_us.end(), 0.0) /
               samples_us.size();
    }

    size_t Count() const { return samples_us.size(); }
};

struct BandwidthResult {
    double total_bytes = 0;
    double total_secs = 0;
    size_t total_ops = 0;
    LatencyStats latency;

    double BW_MBps() const {
        return total_secs > 0 ? (total_bytes / (1024.0 * 1024.0)) / total_secs
                              : 0;
    }

    double IOPS() const {
        if (total_secs <= 0) return 0;
        size_t ops = total_ops > 0 ? total_ops : latency.Count();
        return static_cast<double>(ops) / total_secs;
    }
};

static std::string FmtWR(double w, double r) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%9.1f / %-9.1f", w, r);
    return buf;
}

static std::string FmtSpeedup(double sw, double sr) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%6.2fx / %6.2fx", sw, sr);
    return buf;
}

static void FillPattern(char *buf, size_t len, uint32_t seed) {
    std::mt19937 gen(seed);
    auto *p = reinterpret_cast<uint32_t *>(buf);
    size_t n = len / sizeof(uint32_t);
    for (size_t i = 0; i < n; ++i) p[i] = gen();
    size_t tail = len % sizeof(uint32_t);
    if (tail) {
        uint32_t v = gen();
        std::memcpy(buf + n * sizeof(uint32_t), &v, tail);
    }
}

// ============================================================================
// Part 1: StorageFile-level sequential bandwidth + latency
// ============================================================================

// Single-thread worker operating on [range_start, range_start + range_bytes).
// I/O buffer is capped at kMaxIoPiece (4 MB) regardless of chunk_size.
// For large chunks the logical chunk is issued as multiple kMaxIoPiece
// pwritev/preadv calls, avoiding both dirty-page throttling and the
// massive page-fault overhead of allocating chunk_size user buffers
// (128 MB chunk × 4 threads = 512 MB → ~130 K page faults ≈ 150 ms).
static BandwidthResult BenchFileSeqWorker(int fd, size_t chunk_size,
                                          off_t range_start,
                                          size_t range_bytes, bool is_write) {
    static constexpr size_t kMaxIoPiece = 4ULL * 1024 * 1024;

    BandwidthResult result;
    size_t remaining = range_bytes;
    off_t offset = range_start;

    const size_t buf_size = std::min(chunk_size, kMaxIoPiece);
    AlignedBuf buf(buf_size);

    if (is_write)
        FillPattern(buf.data(), buf_size, static_cast<uint32_t>(range_start));

    while (remaining > 0) {
        size_t this_chunk = std::min(chunk_size, remaining);

        size_t done = 0;
        while (done < this_chunk) {
            size_t piece = std::min(buf_size, this_chunk - done);
            iovec iov{buf.data(), piece};
            ssize_t ret;
            if (is_write)
                ret = ::pwritev(fd, &iov, 1, offset + done);
            else
                ret = ::preadv(fd, &iov, 1, offset + done);
            if (ret < 0 || static_cast<size_t>(ret) != piece) {
                LOG(ERROR) << (is_write ? "Write" : "Read")
                           << " failed at offset " << (offset + done);
                return result;
            }
            done += piece;
        }
        result.total_bytes += this_chunk;
        offset += this_chunk;
        remaining -= this_chunk;
    }
    return result;
}

// Multi-threaded Posix sequential benchmark.  N threads share the same fd,
// each operating on a disjoint range.
static BandwidthResult BenchFileSeqMT(int fd, size_t chunk_size,
                                      size_t total_bytes, bool is_write,
                                      bool /* do_verify */, int nthreads) {
    if (nthreads <= 1) {
        auto wall_t0 = Clock::now();
        auto r = BenchFileSeqWorker(fd, chunk_size, 0, total_bytes, is_write);
        auto wall_t1 = Clock::now();
        r.total_secs = std::chrono::duration<double>(wall_t1 - wall_t0).count();
        return r;
    }

    size_t per_thread = (total_bytes / nthreads / chunk_size) * chunk_size;
    if (per_thread < chunk_size) {
        nthreads = std::max(1, static_cast<int>(total_bytes / chunk_size));
        per_thread = chunk_size;
    }
    std::vector<BandwidthResult> results(nthreads);

    auto wall_t0 = Clock::now();
    {
        std::vector<std::thread> pool;
        for (int t = 0; t < nthreads; ++t) {
            off_t start = static_cast<off_t>(t) * static_cast<off_t>(per_thread);
            size_t bytes = (t == nthreads - 1)
                ? (total_bytes - start) : per_thread;
            pool.emplace_back([&results, fd, chunk_size, start, bytes,
                               is_write, t]() {
                results[t] = BenchFileSeqWorker(fd, chunk_size, start, bytes,
                                                is_write);
            });
        }
        for (auto &th : pool) th.join();
    }
    auto wall_t1 = Clock::now();

    BandwidthResult merged;
    merged.total_secs =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();
    for (auto &r : results) merged.total_bytes += r.total_bytes;
    return merged;
}

// ============================================================================
// Part 2: StorageFile-level random I/O (IOPS + latency)
// ============================================================================

static BandwidthResult BenchFileRandWorker(int fd, size_t io_size,
                                           size_t file_size, int num_ops,
                                           bool is_write, uint32_t seed,
                                           bool collect_latency = false) {
    BandwidthResult result;
    size_t block_align = 4096;
    size_t max_offset = (file_size - io_size) / block_align * block_align;

    std::mt19937 gen(seed);
    std::uniform_int_distribution<size_t> dist(0, max_offset / block_align);

    AlignedBuf buf(io_size);

    if (is_write)
        FillPattern(buf.data(), io_size, seed);

    for (int i = 0; i < num_ops; ++i) {
        off_t offset = static_cast<off_t>(dist(gen) * block_align);

        if (is_write) {
            iovec iov{buf.data(), io_size};
            auto t0 = Clock::now();
            ssize_t ret = ::pwritev(fd, &iov, 1, offset);
            auto t1 = Clock::now();
            if (ret < 0) break;
            result.total_bytes += io_size;
            if (collect_latency)
                result.latency.Add(
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
        } else {
            iovec iov{buf.data(), io_size};
            auto t0 = Clock::now();
            ssize_t ret = ::preadv(fd, &iov, 1, offset);
            auto t1 = Clock::now();
            if (ret < 0) break;
            result.total_bytes += io_size;
            if (collect_latency)
                result.latency.Add(
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
    }
    result.total_ops = num_ops;
    return result;
}

static BandwidthResult BenchFileRandMT(int fd, size_t io_size,
                                       size_t file_size, int num_ops,
                                       bool is_write, int nthreads) {
    if (nthreads <= 1) {
        auto wall_t0 = Clock::now();
        auto r = BenchFileRandWorker(fd, io_size, file_size, num_ops,
                                     is_write, 12345, true);
        auto wall_t1 = Clock::now();
        r.total_secs = std::chrono::duration<double>(wall_t1 - wall_t0).count();
        r.latency.Sort();
        return r;
    }

    int ops_per_thread = num_ops / nthreads;
    std::vector<BandwidthResult> results(nthreads);

    auto wall_t0 = Clock::now();
    {
        std::vector<std::thread> pool;
        for (int t = 0; t < nthreads; ++t) {
            int ops = (t == nthreads - 1)
                ? (num_ops - ops_per_thread * t) : ops_per_thread;
            uint32_t seed = 12345 + t;
            pool.emplace_back([&results, fd, io_size, file_size, ops,
                               is_write, seed, t]() {
                results[t] = BenchFileRandWorker(fd, io_size, file_size, ops,
                                                 is_write, seed);
            });
        }
        for (auto &th : pool) th.join();
    }
    auto wall_t1 = Clock::now();

    BandwidthResult merged;
    merged.total_secs =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();
    for (auto &r : results) {
        merged.total_bytes += r.total_bytes;
        merged.total_ops += r.total_ops;
    }
    return merged;
}

// ============================================================================
// Part 2b: SPDK async sequential bandwidth (iodepth > 1)
//          Bypasses SpdkFile — uses SpdkEnv async API + DMA buffers directly.
// ============================================================================

static size_t spdk_align_up(size_t v) {
    size_t a = std::max<size_t>(SpdkEnv::Instance().GetBlockSize(), 4096);
    return (v + a - 1) & ~(a - 1);
}

static BandwidthResult BenchSpdkSeqAsync(size_t chunk_size,
                                         size_t total_bytes, bool is_write,
                                         bool do_verify, int iodepth,
                                         off_t base_offset = 0) {
    auto &env = SpdkEnv::Instance();
    size_t aligned_chunk = spdk_align_up(chunk_size);

    constexpr size_t kMaxDmaTotal = 256ULL * 1024 * 1024;
    int max_qd = static_cast<int>(kMaxDmaTotal / aligned_chunk);
    if (max_qd < 2) max_qd = 2;
    if (iodepth > max_qd) iodepth = max_qd;

    auto dma_bufs = std::make_unique<void *[]>(iodepth);
    auto reqs = std::make_unique<SpdkIoRequest[]>(iodepth);
    int actual_qd = 0;
    for (int i = 0; i < iodepth; ++i) {
        dma_bufs[i] = env.DmaMalloc(aligned_chunk, env.GetBlockSize());
        if (!dma_bufs[i]) break;
        ++actual_qd;
    }
    if (actual_qd == 0) {
        LOG(ERROR) << "DmaMalloc failed for chunk=" << FormatSize(chunk_size)
                   << ", cannot run SPDK test";
        return {};
    }
    if (actual_qd < iodepth) {
        LOG(WARNING) << "DmaMalloc partial " << actual_qd << "/" << iodepth
                     << " for chunk=" << FormatSize(chunk_size);
    }
    iodepth = actual_qd;

    auto src_bufs = std::make_unique<std::vector<char>[]>(iodepth);
    if (is_write) {
        for (int i = 0; i < iodepth; ++i) {
            src_bufs[i].resize(chunk_size);
            FillPattern(src_bufs[i].data(), chunk_size,
                        static_cast<uint32_t>(i));
        }
    }

    BandwidthResult result;
    size_t total_ops = total_bytes / chunk_size;
    size_t submitted = 0, completed = 0;
    int head = 0, tail = 0;
    off_t submit_offset = base_offset;

    std::vector<char> verify_buf;
    if (do_verify) verify_buf.resize(chunk_size);

    auto batch_ptrs = std::make_unique<SpdkIoRequest *[]>(iodepth);

    auto wall_t0 = Clock::now();

    while (completed < total_ops) {
        // --- fill pipeline (batch) ---
        int batch_count = 0;
        while (submitted - completed < static_cast<size_t>(iodepth) &&
               submitted < total_ops) {
            int slot = head;

            reqs[slot].completed.store(false, std::memory_order_relaxed);
            reqs[slot].success = false;
            reqs[slot].op =
                is_write ? SpdkIoRequest::WRITE : SpdkIoRequest::READ;
            reqs[slot].buf = dma_bufs[slot];
            reqs[slot].offset = submit_offset;
            reqs[slot].nbytes = aligned_chunk;

            if (is_write) {
                reqs[slot].src_data = src_bufs[slot].data();
                reqs[slot].src_len = chunk_size;
            } else {
                reqs[slot].src_data = nullptr;
                reqs[slot].src_len = 0;
            }
            reqs[slot].src_iov = nullptr;
            reqs[slot].src_iovcnt = 0;

            batch_ptrs[batch_count++] = &reqs[slot];

            submit_offset += chunk_size;
            submitted++;
            head = (head + 1) % iodepth;
        }

        if (batch_count > 0)
            env.SubmitIoBatchAsync(batch_ptrs.get(), batch_count);

        // --- drain completed (FIFO) ---
        while (completed < submitted) {
            if (!reqs[tail].completed.load(std::memory_order_acquire)) break;

            if (!is_write && do_verify && reqs[tail].success) {
                int expected_slot =
                    static_cast<int>(completed % static_cast<size_t>(iodepth));
                FillPattern(verify_buf.data(), chunk_size,
                            static_cast<uint32_t>(expected_slot));
                if (std::memcmp(dma_bufs[tail], verify_buf.data(),
                                chunk_size) != 0) {
                    LOG(ERROR) << "Async verify FAILED at offset "
                               << completed * chunk_size;
                }
            }

            result.total_bytes += chunk_size;
            result.total_ops++;
            completed++;
            tail = (tail + 1) % iodepth;
        }
    }

    auto wall_t1 = Clock::now();
    result.total_secs =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();
    result.latency.Sort();

    for (int i = 0; i < iodepth; ++i)
        if (dma_bufs[i]) env.DmaFree(dma_bufs[i]);
    return result;
}

static BandwidthResult BenchSpdkSeqAsyncMT(size_t chunk_size,
                                            size_t total_bytes, bool is_write,
                                            bool do_verify, int iodepth,
                                            int nthreads) {
    if (nthreads <= 1)
        return BenchSpdkSeqAsync(chunk_size, total_bytes, is_write,
                                 do_verify, iodepth);

    size_t per_thread = (total_bytes / nthreads / chunk_size) * chunk_size;
    if (per_thread < chunk_size) {
        nthreads = std::max(1, static_cast<int>(total_bytes / chunk_size));
        per_thread = chunk_size;
    }

    size_t aligned_chunk = spdk_align_up(chunk_size);
    constexpr size_t kTotalDmaBudget = 256ULL * 1024 * 1024;
    size_t dma_per_thread = kTotalDmaBudget / static_cast<size_t>(nthreads);
    int max_qd_dma = std::max(2,
        static_cast<int>(dma_per_thread / aligned_chunk));
    int per_thread_qd = std::min(std::max(2, iodepth / nthreads), max_qd_dma);

    std::vector<BandwidthResult> results(nthreads);
    auto wall_t0 = Clock::now();
    {
        std::vector<std::thread> pool;
        for (int t = 0; t < nthreads; ++t) {
            off_t start =
                static_cast<off_t>(t) * static_cast<off_t>(per_thread);
            size_t bytes = (t == nthreads - 1)
                ? (total_bytes - static_cast<size_t>(start)) : per_thread;
            pool.emplace_back(
                [&results, chunk_size, start, bytes, is_write,
                 do_verify, per_thread_qd, t]() {
                    results[t] = BenchSpdkSeqAsync(chunk_size, bytes, is_write,
                                                   do_verify, per_thread_qd,
                                                   start);
                });
        }
        for (auto &th : pool) th.join();
    }
    auto wall_t1 = Clock::now();

    BandwidthResult merged;
    merged.total_secs =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();
    for (auto &r : results) merged.total_bytes += r.total_bytes;
    return merged;
}

// ============================================================================
// Part 2c: SPDK async random I/O (iodepth > 1)
// ============================================================================

static BandwidthResult BenchSpdkRandAsync(size_t io_size, size_t file_size,
                                          int num_ops, bool is_write,
                                          int iodepth,
                                          uint32_t seed = 12345) {
    auto &env = SpdkEnv::Instance();
    size_t aligned_io = spdk_align_up(io_size);

    constexpr size_t kMaxDmaTotal = 256ULL * 1024 * 1024;
    int max_qd = static_cast<int>(kMaxDmaTotal / aligned_io);
    if (max_qd < 2) max_qd = 2;
    if (iodepth > max_qd) iodepth = max_qd;

    auto dma_bufs = std::make_unique<void *[]>(iodepth);
    auto reqs = std::make_unique<SpdkIoRequest[]>(iodepth);
    int actual_qd = 0;
    for (int i = 0; i < iodepth; ++i) {
        dma_bufs[i] = env.DmaMalloc(aligned_io, env.GetBlockSize());
        if (!dma_bufs[i]) break;
        ++actual_qd;
    }
    if (actual_qd == 0) {
        LOG(ERROR) << "DmaMalloc failed for io=" << FormatSize(io_size)
                   << ", cannot run SPDK random test";
        return {};
    }
    if (actual_qd < iodepth) {
        LOG(WARNING) << "DmaMalloc partial " << actual_qd << "/" << iodepth
                     << " for io=" << FormatSize(io_size);
    }
    iodepth = actual_qd;

    auto src_bufs = std::make_unique<std::vector<char>[]>(iodepth);
    if (is_write) {
        for (int i = 0; i < iodepth; ++i) {
            src_bufs[i].resize(io_size);
            FillPattern(src_bufs[i].data(), io_size,
                        static_cast<uint32_t>(i));
        }
    }

    size_t block_align = 4096;
    size_t max_off = (file_size - io_size) / block_align * block_align;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<size_t> dist(0, max_off / block_align);

    std::vector<off_t> offsets(num_ops);
    for (int i = 0; i < num_ops; ++i)
        offsets[i] = static_cast<off_t>(dist(gen) * block_align);

    auto batch_ptrs = std::make_unique<SpdkIoRequest *[]>(iodepth);
    auto submit_ts = std::make_unique<Clock::time_point[]>(iodepth);

    BandwidthResult result;
    int submitted = 0, completed_count = 0;
    int head = 0, tail = 0;

    auto wall_t0 = Clock::now();

    while (completed_count < num_ops) {
        int batch_count = 0;
        while (submitted - completed_count < iodepth && submitted < num_ops) {
            int slot = head;
            off_t off = offsets[submitted];

            reqs[slot].completed.store(false, std::memory_order_relaxed);
            reqs[slot].success = false;
            reqs[slot].op =
                is_write ? SpdkIoRequest::WRITE : SpdkIoRequest::READ;
            reqs[slot].buf = dma_bufs[slot];
            reqs[slot].offset = off;
            reqs[slot].nbytes = aligned_io;

            if (is_write) {
                reqs[slot].src_data = src_bufs[slot].data();
                reqs[slot].src_len = io_size;
            } else {
                reqs[slot].src_data = nullptr;
                reqs[slot].src_len = 0;
            }
            reqs[slot].src_iov = nullptr;
            reqs[slot].src_iovcnt = 0;

            submit_ts[slot] = Clock::now();
            batch_ptrs[batch_count++] = &reqs[slot];

            submitted++;
            head = (head + 1) % iodepth;
        }

        if (batch_count > 0)
            env.SubmitIoBatchAsync(batch_ptrs.get(), batch_count);

        while (completed_count < submitted) {
            if (!reqs[tail].completed.load(std::memory_order_acquire)) break;
            auto t_done = Clock::now();
            double us = std::chrono::duration<double, std::micro>(
                            t_done - submit_ts[tail]).count();
            result.latency.Add(us);
            result.total_bytes += io_size;
            result.total_ops++;
            completed_count++;
            tail = (tail + 1) % iodepth;
        }
    }

    auto wall_t1 = Clock::now();
    result.total_secs =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();
    result.latency.Sort();

    for (int i = 0; i < iodepth; ++i)
        if (dma_bufs[i]) env.DmaFree(dma_bufs[i]);
    return result;
}

static BandwidthResult BenchSpdkRandAsyncMT(size_t io_size, size_t file_size,
                                            int num_ops, bool is_write,
                                            int iodepth, int nthreads) {
    if (nthreads <= 1)
        return BenchSpdkRandAsync(io_size, file_size, num_ops, is_write,
                                  iodepth);

    size_t aligned_io = spdk_align_up(io_size);
    constexpr size_t kTotalDmaBudget = 256ULL * 1024 * 1024;
    size_t dma_per_thread = kTotalDmaBudget / static_cast<size_t>(nthreads);
    int max_qd_dma = std::max(2,
        static_cast<int>(dma_per_thread / aligned_io));
    int per_thread_qd = std::min(std::max(2, iodepth / nthreads), max_qd_dma);

    int ops_per_thread = num_ops / nthreads;
    std::vector<BandwidthResult> results(nthreads);

    auto wall_t0 = Clock::now();
    {
        std::vector<std::thread> pool;
        for (int t = 0; t < nthreads; ++t) {
            int ops = (t == nthreads - 1)
                ? (num_ops - ops_per_thread * t) : ops_per_thread;
            uint32_t seed = 12345 + static_cast<uint32_t>(t);
            pool.emplace_back(
                [&results, io_size, file_size, ops, is_write,
                 per_thread_qd, seed, t]() {
                    results[t] = BenchSpdkRandAsync(io_size, file_size, ops,
                                                    is_write, per_thread_qd,
                                                    seed);
                });
        }
        for (auto &th : pool) th.join();
    }
    auto wall_t1 = Clock::now();

    BandwidthResult merged;
    merged.total_secs =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();
    for (auto &r : results) {
        merged.total_bytes += r.total_bytes;
        merged.total_ops += r.total_ops;
    }
    return merged;
}

// ============================================================================
// Part 2e: FILE-level SPDK sequential / random benchmarks
//
// Architecture: identical to the VALUE-level path (SpdkFile::vector_read_batch):
//   - SubmitIoBatchAsync sends requests to reactors via spdk_thread_send_msg
//   - Reactor threads handle *only* I/O submission + NVMe completion callbacks
//   - Calling thread (this thread) handles drain / management / next-batch prep
// This two-thread split keeps reactors 100% dedicated to I/O processing.
// ============================================================================

// Worker for one thread's share of sequential I/O via SubmitIoBatchAsync.
// For writes, data is copied to the DMA buffer on the CALLING thread,
// keeping the reactor entirely free for NVMe command submission/completion.
// The per-slot memcpy provides natural submission pacing.
static void SeqDirectWorker(SpdkEnv &env, size_t io_size,
                            size_t aligned_io, bool is_write,
                            off_t start_offset, size_t my_io_ops, int qd,
                            BandwidthResult *out) {
    if (my_io_ops == 0) return;

    auto dma_bufs = std::make_unique<void *[]>(qd);
    int got = env.DmaPoolAllocBatch(dma_bufs.get(), aligned_io, qd,
                                    env.GetBlockSize());
    if (got == 0) {
        for (int t = std::max(1, qd / 2); t >= 1 && got == 0;
             t = t > 1 ? t / 2 : 0)
            got = env.DmaPoolAllocBatch(dma_bufs.get(), aligned_io, t,
                                        env.GetBlockSize());
        if (got == 0) { LOG(ERROR) << "DMA alloc failed"; return; }
    }
    qd = got;

    std::unique_ptr<std::vector<char>[]> src_bufs;
    if (is_write) {
        src_bufs = std::make_unique<std::vector<char>[]>(qd);
        for (int i = 0; i < qd; i++) {
            src_bufs[i].resize(io_size);
            FillPattern(src_bufs[i].data(), io_size,
                        static_cast<uint32_t>(i));
        }
    }

    auto reqs = std::make_unique<SpdkIoRequest[]>(qd);
    auto batch_ptrs = std::make_unique<SpdkIoRequest *[]>(qd);

    int submitted = 0, completed = 0, head = 0, tail = 0;
    off_t next_offset = start_offset;

    auto t0 = Clock::now();

    while (completed < static_cast<int>(my_io_ops)) {
        int batch_count = 0;
        while (submitted - completed < qd &&
               submitted < static_cast<int>(my_io_ops)) {
            int slot = head;
            auto &req = reqs[slot];
            req.op = is_write ? SpdkIoRequest::WRITE : SpdkIoRequest::READ;
            req.buf = dma_bufs[slot];
            req.offset = static_cast<uint64_t>(next_offset);
            req.nbytes = aligned_io;
            req.completed.store(false, std::memory_order_release);
            req.success = false;
            req._next_batch = nullptr;
            req.dst_iov = nullptr;
            req.dst_iovcnt = 0;
            req.dst_skip = 0;
            if (is_write) {
                std::memcpy(dma_bufs[slot], src_bufs[slot].data(), io_size);
                if (aligned_io > io_size)
                    std::memset(static_cast<char *>(dma_bufs[slot]) + io_size,
                                0, aligned_io - io_size);
            }
            req.src_data = nullptr;
            req.src_len = 0;
            req.src_iov = nullptr;
            req.src_iovcnt = 0;
            batch_ptrs[batch_count++] = &req;
            submitted++;
            next_offset += static_cast<off_t>(io_size);
            head = (head + 1) % qd;

            if (is_write && batch_count >= 8) {
                env.SubmitIoBatchAsync(batch_ptrs.get(), batch_count);
                batch_count = 0;
            }
        }

        if (batch_count > 0)
            env.SubmitIoBatchAsync(batch_ptrs.get(), batch_count);

        while (completed < submitted) {
            if (!reqs[tail].completed.load(std::memory_order_acquire))
                break;
            out->total_bytes += io_size;
            out->total_ops++;
            completed++;
            tail = (tail + 1) % qd;
        }
    }

    out->total_secs =
        std::chrono::duration<double>(Clock::now() - t0).count();
    env.DmaPoolFreeBatch(dma_bufs.get(), aligned_io, qd);
}

static BandwidthResult BenchSpdkSeqDirect(size_t chunk_size,
                                           size_t total_bytes, bool is_write,
                                           int iodepth) {
    auto &env = SpdkEnv::Instance();

    // Cap per-NVMe-request size: 2 MB keeps reactor memcpy at ~100 µs,
    // proven to sustain line-rate writes.  Larger per-request sizes cause
    // the reactor memcpy to starve NVMe completion processing.
    static constexpr size_t kMaxIoSize = 2ULL * 1024 * 1024;
    size_t io_size = std::min(chunk_size, kMaxIoSize);
    size_t aligned_io = spdk_align_up(io_size);

    size_t logical_chunks = total_bytes / chunk_size;
    size_t sub_per_chunk = (chunk_size + io_size - 1) / io_size;
    size_t total_io_ops = logical_chunks * sub_per_chunk;
    if (total_io_ops == 0) return {};

    int nthreads = std::max(1, env.GetNumReactors() * 2);
    constexpr size_t kMaxDmaTotal = 256ULL * 1024 * 1024;
    int per_qd = std::max(2, static_cast<int>(
        kMaxDmaTotal / nthreads / aligned_io));
    per_qd = std::min(iodepth, per_qd);

    size_t ops_per_thread = total_io_ops / nthreads;
    if (ops_per_thread == 0) { nthreads = 1; ops_per_thread = total_io_ops; }

    std::vector<BandwidthResult> results(nthreads);
    std::vector<std::thread> threads;

    off_t offset = 0;
    for (int t = 0; t < nthreads; t++) {
        size_t my_ops = (t < nthreads - 1) ? ops_per_thread
                                            : (total_io_ops - ops_per_thread * t);
        threads.emplace_back(SeqDirectWorker, std::ref(env), io_size,
                             aligned_io, is_write, offset, my_ops, per_qd,
                             &results[t]);
        offset += static_cast<off_t>(my_ops * io_size);
    }

    for (auto &th : threads) th.join();

    BandwidthResult merged;
    double max_secs = 0;
    for (auto &r : results) {
        merged.total_bytes += r.total_bytes;
        merged.total_ops += r.total_ops;
        max_secs = std::max(max_secs, r.total_secs);
    }
    merged.total_secs = max_secs;
    return merged;
}

// Worker for one thread's share of random I/O via SubmitIoBatchAsync.
static void RandDirectWorker(SpdkEnv &env, size_t io_size, size_t aligned_io,
                             bool is_write, const off_t *offsets, int my_ops,
                             int qd, bool collect_latency,
                             BandwidthResult *out) {
    if (my_ops == 0) return;

    auto dma_bufs = std::make_unique<void *[]>(qd);
    int got = env.DmaPoolAllocBatch(dma_bufs.get(), aligned_io, qd,
                                    env.GetBlockSize());
    if (got == 0) {
        for (int t = std::max(1, qd / 2); t >= 1 && got == 0;
             t = t > 1 ? t / 2 : 0)
            got = env.DmaPoolAllocBatch(dma_bufs.get(), aligned_io, t,
                                        env.GetBlockSize());
        if (got == 0) { LOG(ERROR) << "DMA alloc failed"; return; }
    }
    qd = got;

    // Fill DMA buffers directly for writes (same approach as SeqDirectWorker).
    if (is_write) {
        for (int i = 0; i < qd; i++) {
            FillPattern(static_cast<char *>(dma_bufs[i]), io_size,
                        static_cast<uint32_t>(i));
            if (aligned_io > io_size)
                std::memset(static_cast<char *>(dma_bufs[i]) + io_size,
                            0, aligned_io - io_size);
        }
    }

    auto reqs = std::make_unique<SpdkIoRequest[]>(qd);
    auto batch_ptrs = std::make_unique<SpdkIoRequest *[]>(qd);

    std::unique_ptr<Clock::time_point[]> submit_ts;
    if (collect_latency)
        submit_ts = std::make_unique<Clock::time_point[]>(qd);

    int submitted = 0, completed = 0, head = 0, tail = 0;

    auto t0 = Clock::now();

    while (completed < my_ops) {
        int batch_count = 0;
        while (submitted - completed < qd && submitted < my_ops) {
            int slot = head;
            auto &req = reqs[slot];
            req.op = is_write ? SpdkIoRequest::WRITE : SpdkIoRequest::READ;
            req.buf = dma_bufs[slot];
            req.offset = static_cast<uint64_t>(offsets[submitted]);
            req.nbytes = aligned_io;
            req.completed.store(false, std::memory_order_release);
            req.success = false;
            req._next_batch = nullptr;
            req.src_data = nullptr;
            req.src_len = 0;
            req.src_iov = nullptr;
            req.src_iovcnt = 0;
            req.dst_iov = nullptr;
            req.dst_iovcnt = 0;
            req.dst_skip = 0;
            if (submit_ts) submit_ts[slot] = Clock::now();
            batch_ptrs[batch_count++] = &req;
            submitted++;
            head = (head + 1) % qd;
        }

        if (batch_count > 0)
            env.SubmitIoBatchAsync(batch_ptrs.get(), batch_count);

        while (completed < submitted) {
            if (!reqs[tail].completed.load(std::memory_order_acquire))
                break;
            if (submit_ts) {
                double us = std::chrono::duration<double, std::micro>(
                                Clock::now() - submit_ts[tail]).count();
                out->latency.Add(us);
            }
            out->total_bytes += io_size;
            out->total_ops++;
            completed++;
            tail = (tail + 1) % qd;
        }
    }

    out->total_secs =
        std::chrono::duration<double>(Clock::now() - t0).count();
    env.DmaPoolFreeBatch(dma_bufs.get(), aligned_io, qd);
}

static BandwidthResult BenchSpdkRandDirect(size_t io_size, size_t file_size,
                                            int num_ops, bool is_write,
                                            int iodepth,
                                            bool collect_latency = false) {
    auto &env = SpdkEnv::Instance();
    size_t aligned_io = spdk_align_up(io_size);

    size_t block_align = 4096;
    size_t max_off = (file_size - io_size) / block_align * block_align;
    std::mt19937 gen(12345);
    std::uniform_int_distribution<size_t> dist(0, max_off / block_align);

    std::vector<off_t> all_offsets(num_ops);
    for (int i = 0; i < num_ops; ++i)
        all_offsets[i] = static_cast<off_t>(dist(gen) * block_align);

    // Latency test: single thread, QD=1
    if (collect_latency) {
        BandwidthResult result;
        RandDirectWorker(env, io_size, aligned_io, is_write,
                         all_offsets.data(), num_ops, 1, true, &result);
        result.latency.Sort();
        return result;
    }

    int nthreads = std::max(1, env.GetNumReactors() * 2);
    constexpr size_t kMaxDmaTotal = 256ULL * 1024 * 1024;
    int per_qd = std::max(2, static_cast<int>(
        kMaxDmaTotal / nthreads / aligned_io));
    per_qd = std::min(iodepth, per_qd);

    int ops_per_thread = num_ops / nthreads;
    if (ops_per_thread == 0) { nthreads = 1; ops_per_thread = num_ops; }

    std::vector<BandwidthResult> results(nthreads);
    std::vector<std::thread> threads;

    int ops_assigned = 0;
    for (int t = 0; t < nthreads; t++) {
        int my_ops = (t < nthreads - 1) ? ops_per_thread
                                         : (num_ops - ops_assigned);
        threads.emplace_back(RandDirectWorker, std::ref(env), io_size,
                             aligned_io, is_write,
                             all_offsets.data() + ops_assigned,
                             my_ops, per_qd, false, &results[t]);
        ops_assigned += my_ops;
    }

    for (auto &th : threads) th.join();

    BandwidthResult merged;
    double max_secs = 0;
    for (auto &r : results) {
        merged.total_bytes += r.total_bytes;
        merged.total_ops += r.total_ops;
        max_secs = std::max(max_secs, r.total_secs);
    }
    merged.total_secs = max_secs;
    return merged;
}

// ============================================================================
// Part 3: Backend-level (OffsetAllocatorStorageBackend) throughput
// ============================================================================

struct BackendBenchResult {
    double offload_bw_mbps = 0;
    double load_bw_mbps = 0;
    LatencyStats offload_lat;
    LatencyStats load_lat;
    int64_t keys_written = 0;
};

// Drop kernel page cache (sync first to flush dirty pages, then drop).
// Only meaningful for Posix paths; SPDK bypasses the kernel entirely.
static void DropPageCache() {
    ::sync();
    int fd = ::open("/proc/sys/vm/drop_caches", O_WRONLY);
    if (fd >= 0) {
        [[maybe_unused]] auto _ = ::write(fd, "3", 1);
        ::close(fd);
    }
}

static BackendBenchResult BenchBackend(OffsetAllocatorStorageBackend &be,
                                       size_t num_keys, size_t value_size,
                                       int threads, bool do_verify,
                                       bool cold_read = false) {
    BackendBenchResult result;

    auto no_op_handler = [](const std::vector<std::string> &,
                            std::vector<StorageObjectMetadata> &) {
        return ErrorCode::OK;
    };

    // Prepare data
    std::vector<std::string> keys(num_keys);
    std::vector<std::string> values(num_keys);
    for (size_t i = 0; i < num_keys; ++i) {
        keys[i] = "bench_key_" + std::to_string(i);
        values[i].resize(value_size);
        FillPattern(values[i].data(), value_size, static_cast<uint32_t>(i));
    }

    // --- Offload (write) — batch ALL keys per thread ---
    // Reduce threads if fewer keys than threads to avoid empty batches
    if (static_cast<size_t>(threads) > num_keys)
        threads = std::max(1, static_cast<int>(num_keys));
    size_t keys_per_thread = num_keys / threads;
    std::vector<LatencyStats> w_latencies(threads);
    std::vector<int64_t> w_counts(threads, 0);

    // Pre-build per-thread batch maps OUTSIDE the timer
    struct ThreadWriteCtx {
        std::unordered_map<std::string, std::vector<Slice>> batch;
        size_t start, count;
    };
    std::vector<ThreadWriteCtx> w_ctxs(threads);
    for (int t = 0; t < threads; ++t) {
        size_t start = t * keys_per_thread;
        size_t end =
            (t == threads - 1) ? num_keys : start + keys_per_thread;
        size_t count = end - start;
        w_ctxs[t].start = start;
        w_ctxs[t].count = count;
        w_ctxs[t].batch.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            w_ctxs[t].batch.emplace(
                keys[start + i],
                std::vector<Slice>{
                    Slice{const_cast<char *>(values[start + i].data()),
                          values[start + i].size()}});
        }
    }

    auto offload_total_t0 = Clock::now();
    {
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t]() {
                auto &ctx = w_ctxs[t];
                auto t0 = Clock::now();
                auto res = be.BatchOffload(ctx.batch, no_op_handler);
                auto t1 = Clock::now();

                if (res.has_value()) {
                    w_counts[t] = res.value();
                    double us = std::chrono::duration<double, std::micro>(
                                    t1 - t0)
                                    .count();
                    w_latencies[t].Add(us);
                }
            });
        }
        for (auto &th : pool) th.join();
    }
    auto offload_total_t1 = Clock::now();

    int64_t total_written = 0;
    for (int t = 0; t < threads; ++t) {
        total_written += w_counts[t];
        for (auto s : w_latencies[t].samples_us)
            result.offload_lat.Add(s);
    }
    result.offload_lat.Sort();
    result.keys_written = total_written;

    double offload_secs =
        std::chrono::duration<double>(offload_total_t1 - offload_total_t0)
            .count();
    double offload_bytes = static_cast<double>(total_written) * value_size;
    result.offload_bw_mbps = (offload_bytes / (1024.0 * 1024.0)) / offload_secs;

    if (FLAGS_profile) {
        std::cout << "  [profile] Offload: " << num_keys << " keys × "
                  << FormatSize(value_size)
                  << "  wall=" << std::fixed << std::setprecision(2)
                  << offload_secs * 1e6 << "us  BW="
                  << std::setprecision(1) << result.offload_bw_mbps
                  << " MB/s\n";
    }

    // --- Cold-read barrier: flush and drop page cache ---
    if (cold_read) {
        DropPageCache();
    }

    // --- Load (read) — batch ALL keys per thread to exploit SPDK multi-core ---
    std::vector<LatencyStats> r_latencies(threads);

    // Pre-allocate read destination buffers OUTSIDE the timer
    struct ThreadReadCtx {
        std::unique_ptr<std::unique_ptr<char[]>[]> rbufs;
        std::unordered_map<std::string, Slice> slices;
        size_t start, count;
    };
    std::vector<ThreadReadCtx> thread_ctxs(threads);
    for (int t = 0; t < threads; ++t) {
        size_t start = t * keys_per_thread;
        size_t end =
            (t == threads - 1) ? num_keys : start + keys_per_thread;
        size_t count = end - start;

        thread_ctxs[t].start = start;
        thread_ctxs[t].count = count;
        thread_ctxs[t].rbufs =
            std::make_unique<std::unique_ptr<char[]>[]>(count);
        thread_ctxs[t].slices.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            thread_ctxs[t].rbufs[i] =
                std::make_unique<char[]>(value_size);
            thread_ctxs[t].slices.emplace(
                keys[start + i],
                Slice{thread_ctxs[t].rbufs[i].get(), value_size});
        }
    }

    auto load_total_t0 = Clock::now();
    {
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t]() {
                auto &ctx = thread_ctxs[t];

                auto t0 = Clock::now();
                auto res = be.BatchLoad(ctx.slices);
                auto t1 = Clock::now();

                double us =
                    std::chrono::duration<double, std::micro>(t1 - t0).count();
                r_latencies[t].Add(us);

                if (do_verify && res.has_value()) {
                    for (size_t i = 0; i < ctx.count; ++i) {
                        if (std::memcmp(ctx.rbufs[i].get(),
                                        values[ctx.start + i].data(),
                                        value_size) != 0) {
                            LOG(ERROR)
                                << "Backend verify FAILED for "
                                << keys[ctx.start + i];
                        }
                    }
                }
            });
        }
        for (auto &th : pool) th.join();
    }
    auto load_total_t1 = Clock::now();

    for (int t = 0; t < threads; ++t) {
        for (auto s : r_latencies[t].samples_us)
            result.load_lat.Add(s);
    }
    result.load_lat.Sort();

    double load_secs =
        std::chrono::duration<double>(load_total_t1 - load_total_t0).count();
    double load_bytes = static_cast<double>(total_written) * value_size;
    result.load_bw_mbps = (load_bytes / (1024.0 * 1024.0)) / load_secs;

    if (FLAGS_profile) {
        std::cout << "  [profile] Load:    " << num_keys << " keys × "
                  << FormatSize(value_size)
                  << "  wall=" << std::fixed << std::setprecision(2)
                  << load_secs * 1e6 << "us  BW="
                  << std::setprecision(1) << result.load_bw_mbps
                  << " MB/s\n";
    }

    return result;
}

// ============================================================================
// Table formatting
// ============================================================================

static void PrintSeparator(int width = 110) {
    std::cout << std::string(width, '-') << "\n";
}

static void PrintLatencyRow(const std::string &label,
                            const LatencyStats &posix,
                            const LatencyStats &spdk) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "  %-10s  %8.1f / %8.1f / %8.1f      %8.1f / %8.1f / %8.1f",
                  label.c_str(), posix.Percentile(50), posix.Percentile(99),
                  posix.Percentile(99.9), spdk.Percentile(50),
                  spdk.Percentile(99), spdk.Percentile(99.9));
    std::cout << buf << "\n";
}

// ============================================================================
// Test: File-level sequential bandwidth
// ============================================================================

static void RunFileSeqBench() {
    auto &env = SpdkEnv::Instance();
    const char *spdk_mode = FLAGS_nvme_pci_addr.empty() ? "malloc" : "NVMe";
    const char *posix_mode = FLAGS_posix_direct ? "O_DIRECT" : "cached";
    bool on_real_disk = !FLAGS_nvme_pci_addr.empty();

    const int TW = on_real_disk ? 128 : 95;

    std::cout << "\n";
    PrintSeparator(TW);
    std::cout << "  FILE-LEVEL SEQUENTIAL BANDWIDTH   (SPDK=" << spdk_mode
              << ", Posix=" << posix_mode << " T=" << FLAGS_threads
              << ", cores=" << FLAGS_cores
              << ", iodepth=" << FLAGS_iodepth
              << ", iterations=" << FLAGS_iterations
              << (on_real_disk ? ", cold=drop_caches" : "") << ")\n";

    std::vector<size_t> chunk_sizes = {
        4096,               32 * 1024,           128 * 1024,
        512 * 1024,         1024 * 1024,         2ULL * 1024 * 1024,
        8ULL * 1024 * 1024, 16ULL * 1024 * 1024, 32ULL * 1024 * 1024,
        64ULL * 1024 * 1024, 128ULL * 1024 * 1024, 256ULL * 1024 * 1024,
        512ULL * 1024 * 1024};

    uint64_t bdev_size = env.GetBdevSize();
    size_t default_total = 512ULL * 1024 * 1024;
    size_t total_data = std::min(static_cast<size_t>(bdev_size / 2),
                                 default_total);

    int nc = env.GetNumReactors();

    auto trimmed_mean = [](std::vector<double> &v) -> double {
        std::sort(v.begin(), v.end());
        if (v.size() <= 2)
            return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double sum = 0;
        for (size_t i = 1; i + 1 < v.size(); ++i) sum += v[i];
        return sum / (v.size() - 2);
    };

    int spdk_nthreads = std::max(1, nc * 2);
    char spdk_col[64];
    std::snprintf(spdk_col, sizeof(spdk_col),
                  "SPDK %dC T=%d QD=%d W/R MB/s", nc, spdk_nthreads,
                  FLAGS_iodepth);

    if (on_real_disk) {
        PrintSeparator(TW);
        char hdr[512];
        std::snprintf(hdr, sizeof(hdr),
            "%12s  │  %-25s  │  %-25s  │  %-25s  │  %-21s",
            "ChunkSize", "Posix(warm) W / R MB/s",
            "Posix(cold) W / R MB/s", spdk_col,
            "Speedup(cold) W / R");
        std::cout << hdr << "\n";
        PrintSeparator(TW);
    } else {
        PrintSeparator(TW);
        char hdr[256];
        std::snprintf(hdr, sizeof(hdr),
            "%12s  │  %-25s  │  %-25s  │  %-18s",
            "ChunkSize", "Posix Write / Read MB/s",
            spdk_col, "Speedup W / R");
        std::cout << hdr << "\n";
        PrintSeparator(TW);
    }

    for (size_t chunk : chunk_sizes) {
        size_t half_bdev = static_cast<size_t>(bdev_size / 2);
        size_t min_for_threads =
            std::min(chunk * static_cast<size_t>(FLAGS_threads), half_bdev);
        size_t effective_total =
            std::max({total_data, chunk * 2, min_for_threads});
        if (effective_total > half_bdev) {
            std::cout << std::setw(12) << FormatSize(chunk).c_str()
                      << "  │  (skipped — exceeds bdev capacity)\n";
            continue;
        }

        std::vector<double> posix_w, posix_r;
        std::vector<double> posix_cold_w, posix_cold_r;
        std::vector<double> spdk_w, spdk_r;

        // Create + preallocate file ONCE per chunk size (like VALUE bench).
        // Avoids per-iteration ext4 extent allocation + journal overhead.
        int warm_fd = open(FLAGS_posix_path.c_str(),
                           OpenPosixFlags(true), 0644);
        if (warm_fd < 0) {
            LOG(ERROR) << "Cannot open " << FLAGS_posix_path
                       << " errno=" << errno;
            return;
        }
        if (!PreallocateFile(warm_fd, effective_total)) {
            LOG(ERROR) << "PreallocateFile failed";
            close(warm_fd);
            return;
        }

        for (int iter = 0; iter < FLAGS_iterations; ++iter) {
            bool verify_this = FLAGS_verify && (iter == 0);

            // --- Posix warm (write then read — page cache hot) ---
            {
                auto wr = BenchFileSeqMT(warm_fd, chunk, effective_total, true,
                                         false, FLAGS_threads);
                auto rd = BenchFileSeqMT(warm_fd, chunk, effective_total, false,
                                         verify_this, FLAGS_threads);
                posix_w.push_back(wr.BW_MBps());
                posix_r.push_back(rd.BW_MBps());
            }

            // --- Posix cold (drop cache, then write + read from disk) ---
            if (on_real_disk) {
                DropPageCache();
                {
                    auto wr = BenchFileSeqMT(warm_fd, chunk, effective_total,
                                             true, false, FLAGS_threads);
                    posix_cold_w.push_back(wr.BW_MBps());
                }
                DropPageCache();
                {
                    auto rd = BenchFileSeqMT(warm_fd, chunk, effective_total,
                                             false, false, FLAGS_threads);
                    posix_cold_r.push_back(rd.BW_MBps());
                }
            }

            // --- SPDK (direct-on-reactor poller, no msg passing) ---
            {
                auto wr = BenchSpdkSeqDirect(chunk, effective_total, true,
                                             FLAGS_iodepth);
                auto rd = BenchSpdkSeqDirect(chunk, effective_total, false,
                                             FLAGS_iodepth);
                spdk_w.push_back(wr.BW_MBps());
                spdk_r.push_back(rd.BW_MBps());
            }
        }
        close(warm_fd);
        unlink(FLAGS_posix_path.c_str());

        double pw = trimmed_mean(posix_w), pr = trimmed_mean(posix_r);
        double pcw = posix_cold_w.empty() ? 0 : trimmed_mean(posix_cold_w);
        double pcr = posix_cold_r.empty() ? 0 : trimmed_mean(posix_cold_r);
        double sw = trimmed_mean(spdk_w), sr = trimmed_mean(spdk_r);

        if (on_real_disk) {
            std::string c1 = FmtWR(pw, pr), c2 = FmtWR(pcw, pcr);
            std::string c3 = FmtWR(sw, sr);
            std::string c4 = FmtSpeedup(pcw > 0 ? sw / pcw : 0,
                                         pcr > 0 ? sr / pcr : 0);
            char row[512];
            std::snprintf(row, sizeof(row),
                "%12s  │  %-25s  │  %-25s  │  %-25s  │  %-21s",
                FormatSize(chunk).c_str(),
                c1.c_str(), c2.c_str(), c3.c_str(), c4.c_str());
            std::cout << row << "\n";
        } else {
            std::string c1 = FmtWR(pw, pr), c2 = FmtWR(sw, sr);
            std::string c3 = FmtSpeedup(pw > 0 ? sw / pw : 0,
                                         pr > 0 ? sr / pr : 0);
            char row[256];
            std::snprintf(row, sizeof(row),
                "%12s  │  %-25s  │  %-25s  │  %-18s",
                FormatSize(chunk).c_str(),
                c1.c_str(), c2.c_str(), c3.c_str());
            std::cout << row << "\n";
        }
    }
    PrintSeparator(TW);
}

// ============================================================================
// Test: File-level random I/O (IOPS + latency)
// ============================================================================

static void RunFileRandBench() {
    auto &env = SpdkEnv::Instance();
    const char *spdk_mode = FLAGS_nvme_pci_addr.empty() ? "malloc" : "NVMe";
    const char *posix_mode = FLAGS_posix_direct ? "O_DIRECT" : "cached";

    bool on_real_disk = !FLAGS_nvme_pci_addr.empty();
    const int TW = on_real_disk ? 128 : 95;

    std::cout << "\n";
    PrintSeparator(TW);
    std::cout << "  FILE-LEVEL RANDOM I/O  (4KB blocks, SPDK=" << spdk_mode
              << ", Posix=" << posix_mode << " T=" << FLAGS_threads
              << ", cores=" << FLAGS_cores
              << ", iodepth=" << FLAGS_iodepth
              << ", iterations=" << FLAGS_iterations
              << (on_real_disk ? ", cold=drop_caches" : "") << ")\n";

    uint64_t bdev_size = env.GetBdevSize();
    size_t file_size = bdev_size / 2;
    if (file_size > 512ULL * 1024 * 1024)
        file_size = 512ULL * 1024 * 1024;
    int nc = env.GetNumReactors();

    std::vector<int> ops_list = {10000, 50000, 200000};

    int spdk_nthreads_r = std::max(1, nc * 2);
    char spdk_col[64];
    std::snprintf(spdk_col, sizeof(spdk_col),
                  "SPDK %dC T=%d QD=%d W/R MB/s", nc, spdk_nthreads_r,
                  FLAGS_iodepth);

    if (on_real_disk) {
        PrintSeparator(TW);
        char rhdr[512];
        std::snprintf(rhdr, sizeof(rhdr),
            "%12s  │  %-25s  │  %-25s  │  %-25s  │  %-21s",
            "NumOps", "Posix(warm) W / R MB/s",
            "Posix(cold) W / R MB/s", spdk_col,
            "Speedup(cold) W / R");
        std::cout << rhdr << "\n";
        PrintSeparator(TW);
    } else {
        PrintSeparator(TW);
        char rhdr[512];
        std::snprintf(rhdr, sizeof(rhdr),
            "%12s  │  %-25s  │  %-25s  │  %-18s",
            "NumOps", "Posix W / R MB/s",
            spdk_col, "Speedup W / R");
        std::cout << rhdr << "\n";
        PrintSeparator(TW);
    }

    auto trimmed_mean_rand = [](std::vector<double> &v) -> double {
        std::sort(v.begin(), v.end());
        if (v.size() <= 2)
            return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double sum = 0;
        for (size_t i = 1; i + 1 < v.size(); ++i) sum += v[i];
        return sum / (v.size() - 2);
    };

    for (int num_ops : ops_list) {
        std::vector<double> pw_v, pr_v, pcw_v, pcr_v, sw_v, sr_v;

        for (int iter = 0; iter < FLAGS_iterations; ++iter) {
            // Pre-fill Posix + SPDK
            {
                int fd = open(FLAGS_posix_path.c_str(),
                              OpenPosixFlags(true), 0644);
                if (fd >= 0) {
                    PreallocateFile(fd, file_size);
                    BenchFileSeqMT(fd, 1024 * 1024, file_size, true,
                                   false, 1);
                    close(fd);
                }
            }
            BenchSpdkSeqDirect(1024 * 1024, file_size, true,
                               FLAGS_iodepth);

            // Posix warm random
            {
                int fd = open(FLAGS_posix_path.c_str(),
                              OpenPosixFlags(false), 0644);
                if (fd >= 0) {
                    auto wr = BenchFileRandMT(fd, 4096, file_size, num_ops,
                                              true, FLAGS_threads);
                    auto rd = BenchFileRandMT(fd, 4096, file_size, num_ops,
                                              false, FLAGS_threads);
                    pw_v.push_back(wr.BW_MBps());
                    pr_v.push_back(rd.BW_MBps());
                    close(fd);
                }
            }

            // Posix cold random
            if (on_real_disk) {
                DropPageCache();
                int fd = open(FLAGS_posix_path.c_str(),
                              OpenPosixFlags(false), 0644);
                if (fd >= 0) {
                    auto wr = BenchFileRandMT(fd, 4096, file_size, num_ops,
                                              true, FLAGS_threads);
                    auto rd = BenchFileRandMT(fd, 4096, file_size, num_ops,
                                              false, FLAGS_threads);
                    pcw_v.push_back(wr.BW_MBps());
                    pcr_v.push_back(rd.BW_MBps());
                    close(fd);
                }
            }
            unlink(FLAGS_posix_path.c_str());

            // SPDK random (direct-on-reactor poller)
            {
                auto wr = BenchSpdkRandDirect(4096, file_size, num_ops, true,
                                              FLAGS_iodepth);
                auto rd = BenchSpdkRandDirect(4096, file_size, num_ops, false,
                                              FLAGS_iodepth);
                sw_v.push_back(wr.BW_MBps());
                sr_v.push_back(rd.BW_MBps());
            }
        }

        double pw = trimmed_mean_rand(pw_v), pr = trimmed_mean_rand(pr_v);
        double pcw = pcw_v.empty() ? 0 : trimmed_mean_rand(pcw_v);
        double pcr = pcr_v.empty() ? 0 : trimmed_mean_rand(pcr_v);
        double sw = trimmed_mean_rand(sw_v), sr = trimmed_mean_rand(sr_v);

        if (on_real_disk) {
            std::string c1 = FmtWR(pw, pr), c2 = FmtWR(pcw, pcr);
            std::string c3 = FmtWR(sw, sr);
            std::string c4 = FmtSpeedup(pcw > 0 ? sw / pcw : 0,
                                         pcr > 0 ? sr / pcr : 0);
            char row[512];
            std::snprintf(row, sizeof(row),
                "%12d  │  %-25s  │  %-25s  │  %-25s  │  %-21s",
                num_ops, c1.c_str(), c2.c_str(), c3.c_str(), c4.c_str());
            std::cout << row << "\n";
        } else {
            std::string c1 = FmtWR(pw, pr), c2 = FmtWR(sw, sr);
            std::string c3 = FmtSpeedup(pw > 0 ? sw / pw : 0,
                                         pr > 0 ? sr / pr : 0);
            char row[512];
            std::snprintf(row, sizeof(row),
                "%12d  │  %-25s  │  %-25s  │  %-18s",
                num_ops, c1.c_str(), c2.c_str(), c3.c_str());
            std::cout << row << "\n";
        }
    }
    PrintSeparator(TW);

    // Latency comparison: 4KB random I/O at QD=1 (industry-standard latency test)
    constexpr int kLatencyOps = 50000;
    std::cout << "\n";
    int lat_width = on_real_disk ? 145 : 100;
    PrintSeparator(lat_width);
    std::cout << "  4KB RANDOM I/O LATENCY (us)  —  QD=1, " << kLatencyOps
              << " ops"
              << (on_real_disk ? ", cold=drop_caches" : "") << "\n";
    PrintSeparator(lat_width);

    if (on_real_disk) {
        char lat_hdr[512];
        std::snprintf(lat_hdr, sizeof(lat_hdr),
            "%14s  │  %10s  %10s  %10s  │  %10s  %10s  %10s  │  %10s  %10s  %10s",
            "", "Warm p50", "p99", "p99.9",
            "Cold p50", "p99", "p99.9",
            "SPDK p50", "p99", "p99.9");
        std::cout << lat_hdr << "\n";
    } else {
        char lat_hdr[256];
        std::snprintf(lat_hdr, sizeof(lat_hdr),
            "%14s  │  %10s  %10s  %10s  │  %10s  %10s  %10s",
            "", "Posix p50", "p99", "p99.9",
            "SPDK p50", "p99", "p99.9");
        std::cout << lat_hdr << "\n";
    }
    PrintSeparator(lat_width);

    // Pre-fill both
    {
        int fd = open(FLAGS_posix_path.c_str(), OpenPosixFlags(true), 0644);
        if (fd >= 0) {
            PreallocateFile(fd, file_size);
            BenchFileSeqMT(fd, 1024 * 1024, file_size, true, false, 1);
            close(fd);
        }
    }
    BenchSpdkSeqDirect(1024 * 1024, file_size, true, FLAGS_iodepth);

    // Posix warm latency — single-thread QD=1 for per-op measurement
    BandwidthResult posix_wlat, posix_rlat;
    {
        int fd = open(FLAGS_posix_path.c_str(), OpenPosixFlags(false), 0644);
        if (fd >= 0) {
            posix_wlat = BenchFileRandMT(fd, 4096, file_size, kLatencyOps,
                                         true, 1);
            posix_rlat = BenchFileRandMT(fd, 4096, file_size, kLatencyOps,
                                         false, 1);
            close(fd);
        }
    }

    // Posix cold latency — single-thread QD=1
    BandwidthResult posix_cold_wlat, posix_cold_rlat;
    if (on_real_disk) {
        DropPageCache();
        int fd = open(FLAGS_posix_path.c_str(), OpenPosixFlags(false), 0644);
        if (fd >= 0) {
            posix_cold_wlat = BenchFileRandMT(fd, 4096, file_size,
                                              kLatencyOps, true, 1);
            DropPageCache();
            posix_cold_rlat = BenchFileRandMT(fd, 4096, file_size,
                                              kLatencyOps, false, 1);
            close(fd);
        }
    }
    unlink(FLAGS_posix_path.c_str());

    // SPDK latency at QD=1 (direct poller — true single-op latency)
    auto spdk_wlat = BenchSpdkRandDirect(4096, file_size, kLatencyOps, true,
                                          1, true);
    auto spdk_rlat = BenchSpdkRandDirect(4096, file_size, kLatencyOps, false,
                                          1, true);

    if (on_real_disk) {
        auto print_lat_row_3 = [](const char *label, const LatencyStats &warm,
                                  const LatencyStats &cold,
                                  const LatencyStats &spdk) {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "%14s  │  %10.2f  %10.2f  %10.2f  │  %10.2f  %10.2f  %10.2f  │  %10.2f  %10.2f  %10.2f",
                label,
                warm.Percentile(50), warm.Percentile(99),
                warm.Percentile(99.9),
                cold.Percentile(50), cold.Percentile(99),
                cold.Percentile(99.9),
                spdk.Percentile(50), spdk.Percentile(99),
                spdk.Percentile(99.9));
            std::cout << buf << "\n";
        };
        print_lat_row_3("Write", posix_wlat.latency, posix_cold_wlat.latency,
                        spdk_wlat.latency);
        print_lat_row_3("Read", posix_rlat.latency, posix_cold_rlat.latency,
                        spdk_rlat.latency);
    } else {
        auto print_lat_row = [](const char *label, const LatencyStats &posix,
                                const LatencyStats &spdk) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "%14s  │  %10.2f  %10.2f  %10.2f  │  %10.2f  %10.2f  %10.2f",
                label,
                posix.Percentile(50), posix.Percentile(99),
                posix.Percentile(99.9),
                spdk.Percentile(50), spdk.Percentile(99),
                spdk.Percentile(99.9));
            std::cout << buf << "\n";
        };
        print_lat_row("Write", posix_wlat.latency, spdk_wlat.latency);
        print_lat_row("Read", posix_rlat.latency, spdk_rlat.latency);
    }
    PrintSeparator(lat_width);
}

// ============================================================================
// Test: Backend-level throughput (OffsetAllocatorStorageBackend)
// ============================================================================

static void RunBackendBench() {
    std::cout << "\n";
    PrintSeparator();
    const char *spdk_mode = FLAGS_nvme_pci_addr.empty() ? "malloc" : "NVMe";
    bool on_real_disk = !FLAGS_nvme_pci_addr.empty();
    std::cout << "  BACKEND THROUGHPUT: OffsetAllocatorStorageBackend"
              << "  (SPDK=" << spdk_mode
              << ", keys=" << FLAGS_backend_keys
              << ", value=" << FLAGS_backend_value_kb << "KB"
              << ", threads=" << FLAGS_threads
              << ", iterations=" << FLAGS_iterations
              << (on_real_disk ? ", cold=drop_caches" : "") << ")\n";
    PrintSeparator();

    namespace fs = std::filesystem;
    auto &env = SpdkEnv::Instance();

    fs::path posix_base = FLAGS_posix_backend_dir.empty()
                              ? fs::temp_directory_path()
                              : fs::path(FLAGS_posix_backend_dir);
    std::string posix_dir = (posix_base / "spdk_bench_posix").string();
    std::string spdk_dir =
        (fs::temp_directory_path() / "spdk_bench_spdk").string();

    size_t value_size = FLAGS_backend_value_kb * 1024;
    size_t num_keys = FLAGS_backend_keys;
    int threads = FLAGS_threads;
    int iters = FLAGS_iterations;

    // Pre-warm DMA pool: allocate enough buffers for ALL concurrent threads
    {
        size_t warm_size = spdk_align_up(value_size + 4096 + 64);
        constexpr size_t kDmaBudget = 256ULL * 1024 * 1024;
        int max_qd_total = warm_size > 0
            ? std::max(4, static_cast<int>(kDmaBudget / warm_size))
            : 128;
        int per_thread_qd = std::max(4, max_qd_total / std::max(1, threads));
        per_thread_qd = std::min(per_thread_qd, 128);
        int total_warm = per_thread_qd * threads;
        env.DmaPoolPrewarm(warm_size, total_warm, env.GetBlockSize());
    }

    // Header
    char thdr[512];
    std::snprintf(thdr, sizeof(thdr),
                  "%14s  │  %14s  │  %14s  │  %14s  │  %14s",
                  "Backend", "Offload MB/s", "Load MB/s",
                  "Offload p99us", "Load p99us");
    std::cout << thdr << "\n";
    PrintSeparator();

    auto run_one = [&](const std::string &label, bool use_spdk,
                       const std::string &dir, bool cold_read) {
        std::vector<double> offload_bws, load_bws;
        LatencyStats offload_lat_agg, load_lat_agg;

        for (int iter = 0; iter < iters; ++iter) {
            fs::create_directories(dir);

            FileStorageConfig config;
            config.storage_filepath = dir;
            config.storage_backend_type = StorageBackendType::kOffsetAllocator;
            config.total_size_limit =
                static_cast<int64_t>(num_keys * (value_size + 4096) * 2);
            config.total_keys_limit = static_cast<int64_t>(num_keys * 2);
            config.use_spdk = use_spdk;
            if (use_spdk) config.spdk_bdev_name = FLAGS_spdk_bdev_name;

            OffsetAllocatorStorageBackend be(config);
            auto init_res = be.Init();
            if (!init_res.has_value()) {
                LOG(ERROR) << label << " Init failed";
                return;
            }

            bool verify_this = FLAGS_verify && (iter == 0);
            auto res = BenchBackend(be, num_keys, value_size, threads,
                                    verify_this, cold_read);
            offload_bws.push_back(res.offload_bw_mbps);
            load_bws.push_back(res.load_bw_mbps);

            if (iter == iters - 1) {
                offload_lat_agg = std::move(res.offload_lat);
                load_lat_agg = std::move(res.load_lat);
            }

            fs::remove_all(dir);
        }

        auto trimmed_mean = [](std::vector<double> &v) -> double {
            std::sort(v.begin(), v.end());
            if (v.size() <= 2)
                return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            double sum = 0;
            for (size_t i = 1; i + 1 < v.size(); ++i) sum += v[i];
            return sum / (v.size() - 2);
        };

        double ob = trimmed_mean(offload_bws);
        double lb = trimmed_mean(load_bws);
        char row[256];
        std::snprintf(
            row, sizeof(row),
            "%14s  │  %14.1f  │  %14.1f  │  %14.1f  │  %14.1f",
            label.c_str(), ob, lb, offload_lat_agg.Percentile(99),
            load_lat_agg.Percentile(99));
        std::cout << row << "\n";
    };

    run_one(on_real_disk ? "Posix(warm)" : "Posix", false, posix_dir, false);
    if (on_real_disk) {
        run_one("Posix(cold)", false, posix_dir, true);
    }
    run_one("SPDK", true, spdk_dir, false);

    PrintSeparator();

    // Multi-value-size sweep
    const int SW = on_real_disk ? 134 : 102;
    std::cout << "\n";
    PrintSeparator(SW);
    std::cout << "  BACKEND THROUGHPUT BY VALUE SIZE  (SPDK=" << spdk_mode
              << ", max_keys=" << num_keys
              << ", threads=" << threads
              << ", bdev=" << FormatSize(env.GetBdevSize())
              << (on_real_disk ? ", cold=drop_caches" : "") << ")\n";

    if (on_real_disk) {
        PrintSeparator(SW);
        char shdr[512];
        std::snprintf(shdr, sizeof(shdr),
            "%12s %5s  │  %-25s  │  %-25s  │  %-25s  │  %-21s",
            "ValueSize", "Keys",
            "Posix(warm) O / L MB/s",
            "Posix(cold) O / L MB/s",
            "SPDK O / L MB/s",
            "Speedup O / L(cold)");
        std::cout << shdr << "\n";
        PrintSeparator(SW);
    } else {
        PrintSeparator(SW);
        char shdr[512];
        std::snprintf(shdr, sizeof(shdr),
            "%12s %6s  │  %-25s  │  %-25s  │  %-18s",
            "ValueSize", "Keys",
            "Posix Offload / Load MB/s",
            "SPDK Offload / Load MB/s",
            "Speedup O / L");
        std::cout << shdr << "\n";
        PrintSeparator(SW);
    }

    std::vector<size_t> value_sizes = {
        4096,              32 * 1024,           128 * 1024,
        512 * 1024,        1024 * 1024,         2ULL * 1024 * 1024,
        8ULL * 1024 * 1024, 16ULL * 1024 * 1024, 32ULL * 1024 * 1024,
        64ULL * 1024 * 1024, 128ULL * 1024 * 1024, 256ULL * 1024 * 1024,
        512ULL * 1024 * 1024};

    for (size_t vsz : value_sizes) {
        uint64_t bdev_cap = env.GetBdevSize();
        size_t per_key_overhead = vsz + 4096 + 64;
        size_t effective_keys = num_keys;

        // Cap by bdev capacity
        if (per_key_overhead > 0) {
            size_t max_keys = bdev_cap / per_key_overhead / 2;
            if (max_keys < 4) max_keys = 4;
            effective_keys = std::min(effective_keys, max_keys);
        }

        // Cap total data per test to ~512MB for fast sweep
        constexpr size_t kMaxTotalData = 512ULL * 1024 * 1024;
        if (vsz > 0) {
            size_t max_by_data = std::max<size_t>(2, kMaxTotalData / vsz);
            effective_keys = std::min(effective_keys, max_by_data);
        }

        double posix_ob = 0, posix_lb = 0;
        double posix_cold_ob = 0, posix_cold_lb = 0;
        double spdk_ob = 0, spdk_lb = 0;

        auto make_posix_be = [&](const std::string &dir)
            -> std::unique_ptr<OffsetAllocatorStorageBackend> {
            fs::create_directories(dir);
            FileStorageConfig config;
            config.storage_filepath = dir;
            config.storage_backend_type = StorageBackendType::kOffsetAllocator;
            config.total_size_limit =
                static_cast<int64_t>(effective_keys * (vsz + 4096) * 2);
            config.total_keys_limit =
                static_cast<int64_t>(effective_keys * 2);
            config.use_spdk = false;
            auto be = std::make_unique<OffsetAllocatorStorageBackend>(config);
            if (!be->Init().has_value()) return nullptr;
            return be;
        };

        // Posix warm read
        {
            auto be = make_posix_be(posix_dir);
            if (be) {
                auto res = BenchBackend(*be, effective_keys, vsz, threads,
                                        false, false);
                posix_ob = res.offload_bw_mbps;
                posix_lb = res.load_bw_mbps;
            }
            fs::remove_all(posix_dir);
        }

        // Posix cold read (only on real disk)
        if (on_real_disk) {
            auto be = make_posix_be(posix_dir);
            if (be) {
                auto res = BenchBackend(*be, effective_keys, vsz, threads,
                                        false, true);
                posix_cold_ob = res.offload_bw_mbps;
                posix_cold_lb = res.load_bw_mbps;
            }
            fs::remove_all(posix_dir);
        }

        // Pre-warm DMA pool for ALL concurrent threads at this value size.
        size_t warm_size = spdk_align_up(vsz + 4096 + 64);
        constexpr size_t kDmaBudget = 256ULL * 1024 * 1024;
        int max_qd_total = warm_size > 0
            ? std::max(4, static_cast<int>(kDmaBudget / warm_size))
            : 128;
        int per_thread_qd = std::max(4,
            std::min(128, max_qd_total / std::max(1, threads)));
        int sweep_qd = std::min(per_thread_qd,
                                static_cast<int>(effective_keys));
        int total_warm = sweep_qd * threads;
        env.DmaPoolDrain();
        env.DmaPoolPrewarm(warm_size, total_warm, env.GetBlockSize());

        // SPDK
        {
            fs::create_directories(spdk_dir);
            FileStorageConfig config;
            config.storage_filepath = spdk_dir;
            config.storage_backend_type = StorageBackendType::kOffsetAllocator;
            config.total_size_limit =
                static_cast<int64_t>(effective_keys * (vsz + 4096) * 2);
            config.total_keys_limit =
                static_cast<int64_t>(effective_keys * 2);
            config.use_spdk = true;
            config.spdk_bdev_name = FLAGS_spdk_bdev_name;

            OffsetAllocatorStorageBackend be(config);
            if (be.Init().has_value()) {
                auto res = BenchBackend(be, effective_keys, vsz, threads,
                                        false, false);
                spdk_ob = res.offload_bw_mbps;
                spdk_lb = res.load_bw_mbps;
            }
            fs::remove_all(spdk_dir);
        }
        env.DmaPoolDrain();

        if (on_real_disk) {
            double cold_lb = posix_cold_lb > 0 ? posix_cold_lb : posix_lb;
            std::string c1 = FmtWR(posix_ob, posix_lb);
            std::string c2 = FmtWR(posix_cold_ob, posix_cold_lb);
            std::string c3 = FmtWR(spdk_ob, spdk_lb);
            std::string c4 = FmtSpeedup(
                posix_ob > 0 ? spdk_ob / posix_ob : 0,
                cold_lb > 0 ? spdk_lb / cold_lb : 0);
            char row[512];
            std::snprintf(row, sizeof(row),
                "%12s %5zu  │  %-25s  │  %-25s  │  %-25s  │  %-21s",
                FormatSize(vsz).c_str(), effective_keys,
                c1.c_str(), c2.c_str(), c3.c_str(), c4.c_str());
            std::cout << row << "\n";
        } else {
            std::string c1 = FmtWR(posix_ob, posix_lb);
            std::string c2 = FmtWR(spdk_ob, spdk_lb);
            std::string c3 = FmtSpeedup(
                posix_ob > 0 ? spdk_ob / posix_ob : 0,
                posix_lb > 0 ? spdk_lb / posix_lb : 0);
            char row[300];
            std::snprintf(row, sizeof(row),
                "%12s %6zu  │  %-25s  │  %-25s  │  %-18s",
                FormatSize(vsz).c_str(), effective_keys,
                c1.c_str(), c2.c_str(), c3.c_str());
            std::cout << row << "\n";
        }
    }
    PrintSeparator(SW);
}

// ============================================================================
// Pipeline chunk size tuning benchmark
// ============================================================================

static void RunPipelineTuneBench() {
    auto &env = SpdkEnv::Instance();
    namespace fs = std::filesystem;
    std::string spdk_dir =
        (fs::temp_directory_path() / "spdk_bench_pipe").string();

    std::vector<size_t> value_sizes = {
        8ULL * 1024 * 1024, 16ULL * 1024 * 1024,
        32ULL * 1024 * 1024, 64ULL * 1024 * 1024,
        128ULL * 1024 * 1024};
    std::vector<int> chunk_kbs = {512, 1024, 2048, 4096};

    int threads = FLAGS_threads;
    size_t num_keys = FLAGS_backend_keys;
    constexpr int kIters = 2;

    std::cout << "\n";
    PrintSeparator(120);
    std::cout << "  PIPELINE CHUNK SIZE TUNING  (threads=" << threads
              << ", keys=" << num_keys << ", iterations=" << kIters << ")\n";
    PrintSeparator(120);

    // Header
    std::cout << std::setw(12) << "ValueSize";
    for (int ck : chunk_kbs) {
        char label[32];
        std::snprintf(label, sizeof(label), " │ chunk=%dKB O/L", ck);
        std::cout << std::setw(28) << label;
    }
    std::cout << "\n";
    PrintSeparator(120);

    for (size_t vsz : value_sizes) {
        uint64_t bdev_cap = env.GetBdevSize();
        size_t per_key_overhead = vsz + 4096 + 64;
        size_t effective_keys = num_keys;
        if (per_key_overhead > 0) {
            size_t max_keys = bdev_cap / per_key_overhead / 2;
            if (max_keys < 4) max_keys = 4;
            effective_keys = std::min(effective_keys, max_keys);
        }
        constexpr size_t kMaxTotalData = 512ULL * 1024 * 1024;
        if (vsz > 0) {
            size_t max_by_data = std::max<size_t>(2, kMaxTotalData / vsz);
            effective_keys = std::min(effective_keys, max_by_data);
        }

        std::cout << std::setw(12) << FormatSize(vsz);

        for (int ck : chunk_kbs) {
            // Set pipeline params for this iteration
            size_t thresh = std::min<size_t>(vsz, 4ULL * 1024 * 1024);
            SpdkEnv::SetPipelineParams(thresh, static_cast<size_t>(ck) * 1024);

            // Pre-warm DMA
            size_t warm_size = spdk_align_up(vsz + 4096 + 64);
            constexpr size_t kDmaBudget = 256ULL * 1024 * 1024;
            int max_qd_total = warm_size > 0
                ? std::max(4, static_cast<int>(kDmaBudget / warm_size))
                : 128;
            int per_thread_qd = std::max(4,
                std::min(128, max_qd_total / std::max(1, threads)));
            int sweep_qd = std::min(per_thread_qd,
                                    static_cast<int>(effective_keys));
            int total_warm = sweep_qd * threads;
            env.DmaPoolDrain();
            env.DmaPoolPrewarm(warm_size, total_warm, env.GetBlockSize());

            std::vector<double> obs, lbs;
            for (int iter = 0; iter < kIters; ++iter) {
                fs::create_directories(spdk_dir);
                FileStorageConfig config;
                config.storage_filepath = spdk_dir;
                config.storage_backend_type =
                    StorageBackendType::kOffsetAllocator;
                config.total_size_limit =
                    static_cast<int64_t>(effective_keys * (vsz + 4096) * 2);
                config.total_keys_limit =
                    static_cast<int64_t>(effective_keys * 2);
                config.use_spdk = true;
                config.spdk_bdev_name = FLAGS_spdk_bdev_name;

                OffsetAllocatorStorageBackend be(config);
                if (be.Init().has_value()) {
                    auto res = BenchBackend(be, effective_keys, vsz, threads,
                                            false, false);
                    obs.push_back(res.offload_bw_mbps);
                    lbs.push_back(res.load_bw_mbps);
                }
                fs::remove_all(spdk_dir);
            }
            env.DmaPoolDrain();

            double ob_avg = obs.empty() ? 0 :
                std::accumulate(obs.begin(), obs.end(), 0.0) / obs.size();
            double lb_avg = lbs.empty() ? 0 :
                std::accumulate(lbs.begin(), lbs.end(), 0.0) / lbs.size();

            char cell[64];
            std::snprintf(cell, sizeof(cell), " │ %8.0f / %-8.0f",
                          ob_avg, lb_avg);
            std::cout << cell;
        }
        std::cout << "\n";
    }
    PrintSeparator(120);

    // Restore original settings
    size_t orig_thresh = FLAGS_pipeline_threshold_kb > 0
        ? static_cast<size_t>(FLAGS_pipeline_threshold_kb) * 1024
        : 4ULL * 1024 * 1024;
    size_t orig_chunk = FLAGS_pipeline_chunk_kb > 0
        ? static_cast<size_t>(FLAGS_pipeline_chunk_kb) * 1024
        : 2ULL * 1024 * 1024;
    SpdkEnv::SetPipelineParams(orig_thresh, orig_chunk);
    std::cout << "\nRecommendation: pick the chunk size with highest Offload + Load sum.\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    gflags::SetUsageMessage(
        "SPDK vs PosixFile performance benchmark.\n"
        "  sudo ./spdk_bench [--test=all|file_seq|file_rand|backend|pipeline_tune] "
        "[--iterations=5] ...");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // Init SPDK
    SpdkEnvConfig cfg;
    cfg.reactor_mask = BuildCoreMask(FLAGS_cores);
    cfg.mem_size_mb = FLAGS_mem_size_mb;
    cfg.pipeline_chunk_kb = FLAGS_pipeline_chunk_kb;
    cfg.pipeline_threshold_kb = FLAGS_pipeline_threshold_kb;

    if (!FLAGS_nvme_pci_addr.empty()) {
        cfg.use_malloc_bdev = false;
        cfg.nvme_pci_addr = FLAGS_nvme_pci_addr;
        cfg.bdev_name = cfg.nvme_ctrl_name + "n1";
        FLAGS_spdk_bdev_name = cfg.bdev_name;
        std::cout << "Initializing SPDK (NVMe PCIe traddr="
                  << FLAGS_nvme_pci_addr << ", bdev=" << cfg.bdev_name
                  << ", cores=" << FLAGS_cores
                  << ", mask=" << cfg.reactor_mask << ")...\n";
    } else {
        cfg.bdev_name = FLAGS_spdk_bdev_name;
        cfg.use_malloc_bdev = true;
        cfg.malloc_num_blocks =
            (FLAGS_spdk_malloc_mb * 1024ULL * 1024) / 4096;
        cfg.malloc_block_size = 4096;
        std::cout << "Initializing SPDK (malloc bdev "
                  << FLAGS_spdk_malloc_mb << " MB, cores="
                  << FLAGS_cores << ", mask=" << cfg.reactor_mask << ")...\n";
    }

    int rc = SpdkEnv::Instance().Init(cfg);
    if (rc != 0) {
        LOG(FATAL) << "SpdkEnv::Init failed rc=" << rc;
        return 1;
    }
    auto &env = SpdkEnv::Instance();
    std::cout << "SPDK ready: block_size=" << env.GetBlockSize()
              << " bdev_size=" << FormatSize(env.GetBdevSize())
              << " reactors=" << env.GetNumReactors() << "\n";

    std::string test = FLAGS_test;

    if (test == "all" || test == "file_seq") RunFileSeqBench();
    if (test == "all" || test == "file_rand") RunFileRandBench();
    if (test == "all" || test == "backend") RunBackendBench();
    if (test == "pipeline_tune") RunPipelineTuneBench();

    SpdkEnv::Instance().Shutdown();
    std::cout << "\nDone.\n";
    return 0;
}

#else  // !USE_SPDK

#include <iostream>
int main() {
    std::cerr << "USE_SPDK not enabled. Rebuild with -DUSE_SPDK=ON.\n";
    return 1;
}

#endif

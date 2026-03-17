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
DEFINE_int32(iterations, 5,
             "Iterations per data point (min/max trimmed, report mean of rest)");
DEFINE_string(test, "all",
              "Which test to run: file_seq, file_rand, backend, all");
DEFINE_bool(verify, true, "Verify data correctness on first iteration");
DEFINE_uint64(backend_keys, 2000,
              "Number of keys for backend throughput test");
DEFINE_uint64(backend_value_kb, 128,
              "Value size in KB for backend test (default: 128)");
DEFINE_int32(threads, 1,
             "Thread count for concurrent backend test (default: 1)");
DEFINE_int32(iodepth, 128,
             "I/O queue depth for SPDK async benchmarks (default: 128)");
DEFINE_int32(cores, 8,
             "Number of SPDK reactor cores (default: 8)");
DEFINE_bool(profile, false,
            "Enable per-phase timing breakdown (pass --v=1 for backend detail)");

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

static BandwidthResult BenchFileSeq(StorageFile &file, size_t chunk_size,
                                    size_t total_bytes, bool is_write,
                                    bool do_verify) {
    BandwidthResult result;
    size_t remaining = total_bytes;
    off_t offset = 0;

    // Pre-allocate buffers once outside the hot loop
    std::vector<char> buf(chunk_size);
    std::vector<char> verify_buf;
    if (do_verify) verify_buf.resize(chunk_size);

    auto wall_t0 = Clock::now();

    while (remaining > 0) {
        size_t this_chunk = std::min(chunk_size, remaining);

        if (is_write) {
            FillPattern(buf.data(), this_chunk, static_cast<uint32_t>(offset));

            auto t0 = Clock::now();
            iovec iov{buf.data(), this_chunk};
            auto res = file.vector_write(&iov, 1, offset);
            auto t1 = Clock::now();

            if (!res.has_value() || res.value() != this_chunk) {
                LOG(ERROR) << "Write failed at offset " << offset;
                break;
            }
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            result.latency.Add(us);
            result.total_bytes += this_chunk;
        } else {
            iovec iov{buf.data(), this_chunk};

            auto t0 = Clock::now();
            auto res = file.vector_read(&iov, 1, offset);
            auto t1 = Clock::now();

            if (!res.has_value() || res.value() != this_chunk) {
                LOG(ERROR) << "Read failed at offset " << offset;
                break;
            }
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            result.latency.Add(us);
            result.total_bytes += this_chunk;

            if (do_verify) {
                FillPattern(verify_buf.data(), this_chunk,
                            static_cast<uint32_t>(offset));
                if (std::memcmp(buf.data(), verify_buf.data(), this_chunk) != 0) {
                    LOG(ERROR) << "Verify FAILED at offset " << offset;
                }
            }
        }
        offset += this_chunk;
        remaining -= this_chunk;
    }

    auto wall_t1 = Clock::now();
    result.total_secs =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();
    result.latency.Sort();
    return result;
}

// ============================================================================
// Part 2: StorageFile-level random I/O (IOPS + latency)
// ============================================================================

static BandwidthResult BenchFileRand(StorageFile &file, size_t io_size,
                                     size_t file_size, int num_ops,
                                     bool is_write) {
    BandwidthResult result;
    size_t block_align = 4096;
    size_t max_offset = (file_size - io_size) / block_align * block_align;

    std::mt19937 gen(12345);
    std::uniform_int_distribution<size_t> dist(0, max_offset / block_align);

    std::vector<char> buf(io_size);

    auto wall_t0 = Clock::now();

    for (int i = 0; i < num_ops; ++i) {
        off_t offset = static_cast<off_t>(dist(gen) * block_align);

        if (is_write) {
            FillPattern(buf.data(), io_size, static_cast<uint32_t>(offset ^ i));
            iovec iov{buf.data(), io_size};

            auto t0 = Clock::now();
            auto res = file.vector_write(&iov, 1, offset);
            auto t1 = Clock::now();

            if (!res.has_value()) break;
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            result.latency.Add(us);
            result.total_bytes += io_size;
        } else {
            iovec iov{buf.data(), io_size};

            auto t0 = Clock::now();
            auto res = file.vector_read(&iov, 1, offset);
            auto t1 = Clock::now();

            if (!res.has_value()) break;
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            result.latency.Add(us);
            result.total_bytes += io_size;
        }
    }

    auto wall_t1 = Clock::now();
    result.total_secs =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();
    result.latency.Sort();
    return result;
}

// ============================================================================
// Part 2b: SPDK async sequential bandwidth (iodepth > 1)
//          Bypasses SpdkFile — uses SpdkEnv async API + DMA buffers directly.
// ============================================================================

static size_t spdk_align_up(size_t v) {
    return (v + 4095) & ~4095UL;
}

static BandwidthResult BenchSpdkSeqAsync(size_t chunk_size,
                                         size_t total_bytes, bool is_write,
                                         bool do_verify, int iodepth) {
    auto &env = SpdkEnv::Instance();
    size_t aligned_chunk = spdk_align_up(chunk_size);

    constexpr size_t kMaxDmaTotal = 256ULL * 1024 * 1024;
    int max_qd = static_cast<int>(kMaxDmaTotal / aligned_chunk);
    if (max_qd < 2) max_qd = 2;
    if (iodepth > max_qd) iodepth = max_qd;

    auto dma_bufs = std::make_unique<void *[]>(iodepth);
    auto reqs = std::make_unique<SpdkIoRequest[]>(iodepth);
    for (int i = 0; i < iodepth; ++i) {
        dma_bufs[i] = env.DmaMalloc(aligned_chunk, env.GetBlockSize());
        if (!dma_bufs[i]) {
            LOG(ERROR) << "DmaMalloc failed for slot " << i;
            for (int j = 0; j < i; ++j) env.DmaFree(dma_bufs[j]);
            return {};
        }
    }

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
    off_t submit_offset = 0;

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

    for (int i = 0; i < iodepth; ++i)
        if (dma_bufs[i]) env.DmaFree(dma_bufs[i]);
    return result;
}

// ============================================================================
// Part 2c: SPDK async random I/O (iodepth > 1)
// ============================================================================

static BandwidthResult BenchSpdkRandAsync(size_t io_size, size_t file_size,
                                          int num_ops, bool is_write,
                                          int iodepth) {
    auto &env = SpdkEnv::Instance();
    size_t aligned_io = spdk_align_up(io_size);

    constexpr size_t kMaxDmaTotal = 256ULL * 1024 * 1024;
    int max_qd = static_cast<int>(kMaxDmaTotal / aligned_io);
    if (max_qd < 2) max_qd = 2;
    if (iodepth > max_qd) iodepth = max_qd;

    auto dma_bufs = std::make_unique<void *[]>(iodepth);
    auto reqs = std::make_unique<SpdkIoRequest[]>(iodepth);
    for (int i = 0; i < iodepth; ++i) {
        dma_bufs[i] = env.DmaMalloc(aligned_io, env.GetBlockSize());
        if (!dma_bufs[i]) {
            for (int j = 0; j < i; ++j) env.DmaFree(dma_bufs[j]);
            return {};
        }
    }

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
    std::mt19937 gen(12345);
    std::uniform_int_distribution<size_t> dist(0, max_off / block_align);

    std::vector<off_t> offsets(num_ops);
    for (int i = 0; i < num_ops; ++i)
        offsets[i] = static_cast<off_t>(dist(gen) * block_align);

    auto batch_ptrs = std::make_unique<SpdkIoRequest *[]>(iodepth);

    BandwidthResult result;
    int submitted = 0, completed_count = 0;
    int head = 0, tail = 0;

    auto wall_t0 = Clock::now();

    while (completed_count < num_ops) {
        int batch_count = 0;
        while (submitted - completed_count < iodepth && submitted < num_ops) {
            int slot = head;
            off_t off = offsets[submitted];

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

            batch_ptrs[batch_count++] = &reqs[slot];

            submitted++;
            head = (head + 1) % iodepth;
        }

        if (batch_count > 0)
            env.SubmitIoBatchAsync(batch_ptrs.get(), batch_count);

        while (completed_count < submitted) {
            if (!reqs[tail].completed.load(std::memory_order_acquire)) break;
            result.total_bytes += io_size;
            result.total_ops++;
            completed_count++;
            tail = (tail + 1) % iodepth;
        }
    }

    auto wall_t1 = Clock::now();
    result.total_secs =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();

    for (int i = 0; i < iodepth; ++i)
        if (dma_bufs[i]) env.DmaFree(dma_bufs[i]);
    return result;
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

static BackendBenchResult BenchBackend(OffsetAllocatorStorageBackend &be,
                                       size_t num_keys, size_t value_size,
                                       int threads, bool do_verify) {
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
    std::cout << "\n";
    PrintSeparator();
    std::cout << "  FILE-LEVEL SEQUENTIAL BANDWIDTH   (cores="
              << FLAGS_cores << ", iodepth=" << FLAGS_iodepth
              << ", iterations=" << FLAGS_iterations << ")\n";
    PrintSeparator();

    std::vector<size_t> chunk_sizes = {
        4096,          64 * 1024,  256 * 1024,  1024 * 1024,
        4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024,
        32 * 1024 * 1024, 64 * 1024 * 1024};

    size_t total_data = FLAGS_spdk_malloc_mb * 1024ULL * 1024 / 2;

    auto &env = SpdkEnv::Instance();
    uint64_t bdev_size = env.GetBdevSize();
    int nc = env.GetNumReactors();

    char hdr[256];
    std::snprintf(hdr, sizeof(hdr),
                  "%12s  │ %26s  │  SPDK %dC QD=%-3d W/R (MB/s) │ %20s",
                  "ChunkSize", "Posix Write / Read (MB/s)", nc, FLAGS_iodepth,
                  "Speedup W / R");
    std::cout << hdr << "\n";
    PrintSeparator();

    for (size_t chunk : chunk_sizes) {
        if (chunk > total_data) {
            std::cout << std::setw(12) << FormatSize(chunk).c_str()
                      << "  │  (skipped — chunk > total_data "
                      << FormatSize(total_data) << ")\n";
            continue;
        }
        size_t effective_total = std::max(total_data, chunk * 4);
        if (effective_total > bdev_size) effective_total = total_data;

        std::vector<double> posix_w, posix_r, spdk_w, spdk_r;

        for (int iter = 0; iter < FLAGS_iterations; ++iter) {
            bool verify_this = FLAGS_verify && (iter == 0);

            // --- Posix (sync preadv/pwritev) ---
            {
                int flags = O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC;
                int fd = open(FLAGS_posix_path.c_str(), flags, 0644);
                if (fd < 0) {
                    LOG(ERROR) << "Cannot open " << FLAGS_posix_path;
                    return;
                }
                if (ftruncate(fd, effective_total) != 0) {
                    LOG(ERROR) << "ftruncate failed";
                    close(fd);
                    return;
                }
                PosixFile pf(FLAGS_posix_path, fd);

                auto wr = BenchFileSeq(pf, chunk, effective_total, true, false);
                auto rd = BenchFileSeq(pf, chunk, effective_total, false, verify_this);
                posix_w.push_back(wr.BW_MBps());
                posix_r.push_back(rd.BW_MBps());
            }
            unlink(FLAGS_posix_path.c_str());

            // --- SPDK (async pipeline, QD = iodepth) ---
            {
                auto wr = BenchSpdkSeqAsync(chunk, effective_total, true,
                                            false, FLAGS_iodepth);
                auto rd = BenchSpdkSeqAsync(chunk, effective_total, false,
                                            verify_this, FLAGS_iodepth);
                spdk_w.push_back(wr.BW_MBps());
                spdk_r.push_back(rd.BW_MBps());
            }
        }

        auto trimmed_mean = [](std::vector<double> &v) -> double {
            std::sort(v.begin(), v.end());
            if (v.size() <= 2) {
                return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            }
            double sum = 0;
            for (size_t i = 1; i + 1 < v.size(); ++i) sum += v[i];
            return sum / (v.size() - 2);
        };

        double pw = trimmed_mean(posix_w), pr = trimmed_mean(posix_r);
        double sw = trimmed_mean(spdk_w), sr = trimmed_mean(spdk_r);

        char row[256];
        std::snprintf(row, sizeof(row),
                      "%12s  │  %10.1f / %-10.1f   │  %10.1f / %-10.1f   │  "
                      "%6.2fx / %.2fx",
                      FormatSize(chunk).c_str(), pw, pr, sw, sr,
                      pw > 0 ? sw / pw : 0, pr > 0 ? sr / pr : 0);
        std::cout << row << "\n";
    }
    PrintSeparator();
}

// ============================================================================
// Test: File-level random I/O (IOPS + latency)
// ============================================================================

static void RunFileRandBench() {
    std::cout << "\n";
    PrintSeparator();
    std::cout << "  FILE-LEVEL RANDOM I/O  (4KB blocks, cores="
              << FLAGS_cores << ", iodepth=" << FLAGS_iodepth
              << ", iterations=" << FLAGS_iterations << ")\n";
    PrintSeparator();

    auto &env = SpdkEnv::Instance();
    uint64_t bdev_size = env.GetBdevSize();
    size_t file_size = FLAGS_spdk_malloc_mb * 1024ULL * 1024 / 2;
    int nc = env.GetNumReactors();

    std::vector<int> ops_list = {10000, 50000, 200000};

    char rhdr[512];
    std::snprintf(rhdr, sizeof(rhdr),
                  "%10s  │ %24s  │ %24s  │  SPDK %dC QD=%-3d W/R (MB/s) │ %20s",
                  "NumOps", "Posix IOPS (W / R)",
                  "Posix W / R (MB/s)", nc, FLAGS_iodepth,
                  "Speedup W / R");
    std::cout << rhdr << "\n";
    PrintSeparator();

    for (int num_ops : ops_list) {
        std::vector<double> pw_iops, pr_iops, sw_iops, sr_iops;

        for (int iter = 0; iter < FLAGS_iterations; ++iter) {
            // Pre-fill Posix
            {
                int fd = open(FLAGS_posix_path.c_str(),
                              O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
                if (fd >= 0) {
                    ftruncate(fd, file_size);
                    PosixFile pf(FLAGS_posix_path, fd);
                    BenchFileSeq(pf, 1024 * 1024, file_size, true, false);
                }
            }

            // Pre-fill SPDK via async sequential write
            BenchSpdkSeqAsync(1024 * 1024, file_size, true, false,
                              FLAGS_iodepth);

            // Posix random (sync)
            {
                int fd = open(FLAGS_posix_path.c_str(),
                              O_RDWR | O_CLOEXEC, 0644);
                PosixFile pf(FLAGS_posix_path, fd);
                auto wr = BenchFileRand(pf, 4096, file_size, num_ops, true);
                auto rd = BenchFileRand(pf, 4096, file_size, num_ops, false);
                pw_iops.push_back(wr.IOPS());
                pr_iops.push_back(rd.IOPS());
            }
            unlink(FLAGS_posix_path.c_str());

            // SPDK random (async pipeline)
            {
                auto wr = BenchSpdkRandAsync(4096, file_size, num_ops, true,
                                             FLAGS_iodepth);
                auto rd = BenchSpdkRandAsync(4096, file_size, num_ops, false,
                                             FLAGS_iodepth);
                sw_iops.push_back(wr.IOPS());
                sr_iops.push_back(rd.IOPS());
            }
        }

        auto trimmed_mean = [](std::vector<double> &v) -> double {
            std::sort(v.begin(), v.end());
            if (v.size() <= 2)
                return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            double sum = 0;
            for (size_t i = 1; i + 1 < v.size(); ++i) sum += v[i];
            return sum / (v.size() - 2);
        };

        double pw = trimmed_mean(pw_iops), pr = trimmed_mean(pr_iops);
        double sw = trimmed_mean(sw_iops), sr = trimmed_mean(sr_iops);

        constexpr double kIoSize = 4096.0;
        constexpr double kMB = 1024.0 * 1024.0;
        double pw_mb = pw * kIoSize / kMB, pr_mb = pr * kIoSize / kMB;
        double sw_mb = sw * kIoSize / kMB, sr_mb = sr * kIoSize / kMB;

        char row[512];
        std::snprintf(row, sizeof(row),
                      "%10d  │  %10.0f / %-10.0f  │  %10.1f / %-10.1f  │  "
                      "%10.1f / %-10.1f  │  %6.2fx / %.2fx",
                      num_ops, pw, pr, pw_mb, pr_mb, sw_mb, sr_mb,
                      pw_mb > 0 ? sw_mb / pw_mb : 0,
                      pr_mb > 0 ? sr_mb / pr_mb : 0);
        std::cout << row << "\n";
    }

    // Latency detail: re-run Posix sync for per-op latency.
    // (SPDK async doesn't track per-op latency in this pipeline mode.)
    std::cout << "\n  Posix per-op latency (us) at 4KB random I/O, "
              << ops_list.back() << " ops:\n";
    std::cout << "              "
              << "       Posix  p50 /    p99 /  p99.9\n";

    {
        int fd = open(FLAGS_posix_path.c_str(),
                      O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        ftruncate(fd, file_size);
        PosixFile pf(FLAGS_posix_path, fd);
        BenchFileSeq(pf, 1024 * 1024, file_size, true, false);

        auto pw = BenchFileRand(pf, 4096, file_size, ops_list.back(), true);
        auto pr = BenchFileRand(pf, 4096, file_size, ops_list.back(), false);
        close(fd);
        unlink(FLAGS_posix_path.c_str());

        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "  %-10s  %8.1f / %8.1f / %8.1f", "Write",
                      pw.latency.Percentile(50), pw.latency.Percentile(99),
                      pw.latency.Percentile(99.9));
        std::cout << buf << "\n";
        std::snprintf(buf, sizeof(buf),
                      "  %-10s  %8.1f / %8.1f / %8.1f", "Read",
                      pr.latency.Percentile(50), pr.latency.Percentile(99),
                      pr.latency.Percentile(99.9));
        std::cout << buf << "\n";
    }
    PrintSeparator();
}

// ============================================================================
// Test: Backend-level throughput (OffsetAllocatorStorageBackend)
// ============================================================================

static void RunBackendBench() {
    std::cout << "\n";
    PrintSeparator();
    std::cout << "  BACKEND THROUGHPUT: OffsetAllocatorStorageBackend"
              << "  (keys=" << FLAGS_backend_keys
              << ", value=" << FLAGS_backend_value_kb << "KB"
              << ", threads=" << FLAGS_threads
              << ", iterations=" << FLAGS_iterations << ")\n";
    PrintSeparator();

    namespace fs = std::filesystem;
    auto &env = SpdkEnv::Instance();

    std::string posix_dir =
        (fs::temp_directory_path() / "spdk_bench_posix").string();
    std::string spdk_dir =
        (fs::temp_directory_path() / "spdk_bench_spdk").string();

    size_t value_size = FLAGS_backend_value_kb * 1024;
    size_t num_keys = FLAGS_backend_keys;
    int threads = FLAGS_threads;
    int iters = FLAGS_iterations;

    // Pre-warm DMA pool with pipeline-depth buffers
    {
        constexpr int kPipelineQD = 128;
        size_t warm_size = spdk_align_up(value_size + 4096 + 64);
        env.DmaPoolPrewarm(warm_size, kPipelineQD, env.GetBlockSize());
    }

    // Header
    std::cout << std::setw(8) << "Backend"
              << "  │ " << std::setw(14) << "Offload MB/s"
              << "  │ " << std::setw(14) << "Load MB/s"
              << "  │ " << std::setw(14) << "Offload p99us"
              << "  │ " << std::setw(14) << "Load p99us"
              << "\n";
    PrintSeparator();

    auto run_one = [&](const std::string &label, bool use_spdk,
                       const std::string &dir) {
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
            auto res =
                BenchBackend(be, num_keys, value_size, threads, verify_this);
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
            "%8s  │  %10.1f     │  %10.1f     │  %10.1f     │  %10.1f",
            label.c_str(), ob, lb, offload_lat_agg.Percentile(99),
            load_lat_agg.Percentile(99));
        std::cout << row << "\n";
    };

    run_one("Posix", false, posix_dir);
    run_one("SPDK", true, spdk_dir);

    PrintSeparator();

    // Multi-value-size sweep
    std::cout << "\n";
    PrintSeparator();
    std::cout << "  VALUE SIZE SWEEP  (max_keys=" << num_keys
              << ", threads=" << threads
              << ", bdev=" << FormatSize(env.GetBdevSize()) << ")\n";
    PrintSeparator(120);

    std::cout << std::setw(12) << "ValueSize"
              << std::setw(8) << "  Keys"
              << "  │ " << std::setw(26) << "Posix Offload / Load MB/s"
              << "  │ " << std::setw(26) << "SPDK  Offload / Load MB/s"
              << "  │ " << std::setw(20) << "Speedup O / L"
              << "\n";
    PrintSeparator(120);

    std::vector<size_t> value_sizes = {
        64 * 1024 * 1024, 32 * 1024 * 1024, 16 * 1024 * 1024,
        8 * 1024 * 1024, 2 * 1024 * 1024, 512 * 1024,
        128 * 1024,      32 * 1024,      4096};

    // No pre-warm for the sweep: each value size naturally fills the pool via
    // the write phase, and the read phase reuses those buffers. Large sizes
    // have few keys (small pipeline QD), so DMA memory stays bounded.
    // Pre-warming with large buffers would exhaust hugepages.

    for (size_t vsz : value_sizes) {
        uint64_t bdev_cap = env.GetBdevSize();
        size_t per_key_overhead = vsz + 4096 + 64;
        size_t effective_keys = num_keys;
        if (per_key_overhead > 0) {
            size_t max_keys = bdev_cap / per_key_overhead / 2;
            if (max_keys < 4) max_keys = 4;
            effective_keys = std::min(num_keys, max_keys);
        }

        double posix_ob = 0, posix_lb = 0, spdk_ob = 0, spdk_lb = 0;

        // Posix
        {
            fs::create_directories(posix_dir);
            FileStorageConfig config;
            config.storage_filepath = posix_dir;
            config.storage_backend_type = StorageBackendType::kOffsetAllocator;
            config.total_size_limit =
                static_cast<int64_t>(effective_keys * (vsz + 4096) * 2);
            config.total_keys_limit =
                static_cast<int64_t>(effective_keys * 2);
            config.use_spdk = false;

            OffsetAllocatorStorageBackend be(config);
            if (be.Init().has_value()) {
                auto res = BenchBackend(be, effective_keys, vsz, threads, false);
                posix_ob = res.offload_bw_mbps;
                posix_lb = res.load_bw_mbps;
            }
            fs::remove_all(posix_dir);
        }

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
                auto res = BenchBackend(be, effective_keys, vsz, threads, false);
                spdk_ob = res.offload_bw_mbps;
                spdk_lb = res.load_bw_mbps;
            }
            fs::remove_all(spdk_dir);
        }
        // Release DMA buffers accumulated during this value-size test
        // to avoid exhausting hugepages across the sweep.
        env.DmaPoolDrain();

        char row[300];
        std::snprintf(
            row, sizeof(row),
            "%12s  %5zu  │  %10.1f / %-10.1f   │  %10.1f / %-10.1f   │  "
            "%6.2fx / %.2fx",
            FormatSize(vsz).c_str(), effective_keys, posix_ob, posix_lb,
            spdk_ob, spdk_lb,
            posix_ob > 0 ? spdk_ob / posix_ob : 0,
            posix_lb > 0 ? spdk_lb / posix_lb : 0);
        std::cout << row << "\n";
    }
    PrintSeparator();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    gflags::SetUsageMessage(
        "SPDK vs PosixFile performance benchmark.\n"
        "  sudo ./spdk_bench [--test=all|file_seq|file_rand|backend] "
        "[--iterations=5] ...");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // Init SPDK
    SpdkEnvConfig cfg;
    cfg.bdev_name = FLAGS_spdk_bdev_name;
    cfg.use_malloc_bdev = true;
    cfg.malloc_num_blocks =
        (FLAGS_spdk_malloc_mb * 1024ULL * 1024) / 4096;
    cfg.malloc_block_size = 4096;
    cfg.reactor_mask = BuildCoreMask(FLAGS_cores);

    std::cout << "Initializing SPDK (malloc bdev "
              << FLAGS_spdk_malloc_mb << " MB, cores="
              << FLAGS_cores << ", mask=" << cfg.reactor_mask << ")...\n";
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

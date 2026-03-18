#include <algorithm>
#include <cstring>
#include <glog/logging.h>

#include "file_interface.h"
#include "spdk/spdk_env.h"

extern "C" {
#include "spdk/bdev.h"
}

namespace mooncake {

// Defaults; overridable via SpdkEnvConfig / --pipeline_chunk_kb at runtime.
// 4MB chunk empirically optimal on PCIe Gen4 NVMe (Micron 7450).
// Threshold = 4×chunk ensures at least 4 chunks so overlap amortizes startup cost.
// Below this, the existing batch pipeline is more efficient.
static size_t g_pipeline_chunk = 4ULL * 1024 * 1024;
static size_t g_pipeline_threshold = 4 * g_pipeline_chunk;

void SpdkEnv::SetPipelineParams(size_t threshold, size_t chunk) {
    g_pipeline_threshold = threshold;
    g_pipeline_chunk = chunk;
}

size_t SpdkEnv::GetPipelineChunk() const { return g_pipeline_chunk; }
size_t SpdkEnv::GetPipelineThreshold() const { return g_pipeline_threshold; }

static ErrorCode ChunkedReadOne(SpdkEnv &env, uint64_t disk_offset,
                                size_t aligned_len, size_t skip,
                                const iovec *iov, int iovcnt,
                                size_t total_user, uint32_t block_size);
static ErrorCode ChunkedWriteOne(SpdkEnv &env, uint64_t disk_offset,
                                 size_t aligned_len,
                                 const iovec *iov, int iovcnt,
                                 size_t total_user, uint32_t block_size);

SpdkFile::SpdkFile(const std::string &filename, uint64_t base_offset,
                   uint64_t max_size)
    : StorageFile(filename, -1),
      base_offset_(base_offset),
      current_offset_(0),
      max_size_(max_size),
      block_size_(SpdkEnv::Instance().GetBlockSize()) {
    if (!SpdkEnv::Instance().IsInitialized()) {
        error_code_ = ErrorCode::INTERNAL_ERROR;
    }
}

SpdkFile::~SpdkFile() {
    if (cached_dma_buf_) {
        SpdkEnv::Instance().DmaFree(cached_dma_buf_);
        cached_dma_buf_ = nullptr;
        cached_dma_size_ = 0;
    }
    fd_ = -1;
}

// ---------------------------------------------------------------------------
// DMA buffer cache — avoids spdk_dma_malloc/free per I/O
// ---------------------------------------------------------------------------
void *SpdkFile::AcquireDmaBuf(size_t needed) {
    if (cached_dma_buf_ && cached_dma_size_ >= needed) {
        return cached_dma_buf_;
    }
    if (cached_dma_buf_) {
        SpdkEnv::Instance().DmaFree(cached_dma_buf_);
    }
    size_t alloc_size = std::max(needed, static_cast<size_t>(64 * 1024));
    cached_dma_buf_ = SpdkEnv::Instance().DmaMalloc(alloc_size, block_size_);
    cached_dma_size_ = cached_dma_buf_ ? alloc_size : 0;
    return cached_dma_buf_;
}

void ReleaseDmaBuf(void *, size_t) {
    // No-op: buffer stays cached for reuse, freed in destructor
}

// ---------------------------------------------------------------------------
// Sequential write
// ---------------------------------------------------------------------------
tl::expected<size_t, ErrorCode> SpdkFile::write(const std::string &buffer,
                                                size_t length) {
    return write(std::span<const char>(buffer.data(), length), length);
}

tl::expected<size_t, ErrorCode> SpdkFile::write(std::span<const char> data,
                                                size_t length) {
    if (error_code_ != ErrorCode::OK) {
        return make_error<size_t>(error_code_);
    }
    if (length == 0) {
        return make_error<size_t>(ErrorCode::FILE_INVALID_BUFFER);
    }
    if (max_size_ > 0 && current_offset_ + length > max_size_) {
        return make_error<size_t>(ErrorCode::FILE_WRITE_FAIL);
    }

    size_t aligned_len = align_up(length);
    void *dma_buf = AcquireDmaBuf(aligned_len);
    if (!dma_buf) {
        return make_error<size_t>(ErrorCode::FILE_WRITE_FAIL);
    }

    SpdkIoRequest req;
    req.op = SpdkIoRequest::WRITE;
    req.buf = dma_buf;
    req.offset = base_offset_ + current_offset_;
    req.nbytes = aligned_len;
    req.src_data = data.data();
    req.src_len = length;
    SpdkEnv::Instance().SubmitIo(&req);

    if (!req.success) {
        return make_error<size_t>(ErrorCode::FILE_WRITE_FAIL);
    }

    current_offset_ += length;
    return length;
}

// ---------------------------------------------------------------------------
// Sequential read
// ---------------------------------------------------------------------------
tl::expected<size_t, ErrorCode> SpdkFile::read(std::string &buffer,
                                               size_t length) {
    if (error_code_ != ErrorCode::OK) {
        return make_error<size_t>(error_code_);
    }
    if (length == 0) {
        return make_error<size_t>(ErrorCode::FILE_INVALID_BUFFER);
    }

    size_t aligned_len = align_up(length);
    void *dma_buf = AcquireDmaBuf(aligned_len);
    if (!dma_buf) {
        return make_error<size_t>(ErrorCode::FILE_READ_FAIL);
    }

    SpdkIoRequest req;
    req.op = SpdkIoRequest::READ;
    req.buf = dma_buf;
    req.offset = base_offset_ + current_offset_;
    req.nbytes = aligned_len;
    SpdkEnv::Instance().SubmitIo(&req);

    if (!req.success) {
        return make_error<size_t>(ErrorCode::FILE_READ_FAIL);
    }

    buffer.assign(static_cast<const char *>(dma_buf), length);

    current_offset_ += length;
    return length;
}

// ---------------------------------------------------------------------------
// Vectored write at explicit offset (relative to base_offset_)
// ---------------------------------------------------------------------------
tl::expected<size_t, ErrorCode> SpdkFile::vector_write(const iovec *iov,
                                                       int iovcnt,
                                                       off_t offset) {
    if (error_code_ != ErrorCode::OK) {
        return make_error<size_t>(error_code_);
    }

    size_t total = 0;
    for (int i = 0; i < iovcnt; ++i) total += iov[i].iov_len;
    if (total == 0) {
        return make_error<size_t>(ErrorCode::FILE_INVALID_BUFFER);
    }

    size_t aligned_len = align_up(total);
    void *dma_buf = AcquireDmaBuf(aligned_len);
    if (!dma_buf) {
        return make_error<size_t>(ErrorCode::FILE_WRITE_FAIL);
    }

    uint64_t abs_offset =
        base_offset_ + static_cast<uint64_t>(offset);
    size_t aligned_offset = abs_offset & ~(static_cast<size_t>(block_size_) - 1);
    if (abs_offset != aligned_offset) {
        LOG(WARNING) << "SpdkFile: vector_write offset not block-aligned, "
                        "rounding down "
                     << abs_offset << " → " << aligned_offset;
    }

    SpdkIoRequest req;
    req.op = SpdkIoRequest::WRITE;
    req.buf = dma_buf;
    req.offset = aligned_offset;
    req.nbytes = aligned_len;
    req.src_iov = iov;
    req.src_iovcnt = iovcnt;
    SpdkEnv::Instance().SubmitIo(&req);

    if (!req.success) {
        return make_error<size_t>(ErrorCode::FILE_WRITE_FAIL);
    }
    return total;
}

// ---------------------------------------------------------------------------
// Vectored read at explicit offset (relative to base_offset_)
// ---------------------------------------------------------------------------
tl::expected<size_t, ErrorCode> SpdkFile::vector_read(const iovec *iov,
                                                      int iovcnt,
                                                      off_t offset) {
    if (error_code_ != ErrorCode::OK) {
        return make_error<size_t>(error_code_);
    }

    size_t total = 0;
    for (int i = 0; i < iovcnt; ++i) total += iov[i].iov_len;
    if (total == 0) {
        return make_error<size_t>(ErrorCode::FILE_INVALID_BUFFER);
    }

    uint64_t abs_offset =
        base_offset_ + static_cast<uint64_t>(offset);
    size_t aligned_offset =
        abs_offset & ~(static_cast<size_t>(block_size_) - 1);
    size_t skip = abs_offset - aligned_offset;
    size_t aligned_len = align_up(total + skip);

    void *dma_buf = AcquireDmaBuf(aligned_len);
    if (!dma_buf) {
        return make_error<size_t>(ErrorCode::FILE_READ_FAIL);
    }

    SpdkIoRequest req;
    req.op = SpdkIoRequest::READ;
    req.buf = dma_buf;
    req.offset = aligned_offset;
    req.nbytes = aligned_len;
    SpdkEnv::Instance().SubmitIo(&req);

    if (!req.success) {
        return make_error<size_t>(ErrorCode::FILE_READ_FAIL);
    }

    const char *src = static_cast<const char *>(dma_buf) + skip;
    for (int i = 0; i < iovcnt; ++i) {
        std::memcpy(iov[i].iov_base, src, iov[i].iov_len);
        src += iov[i].iov_len;
    }

    return total;
}

// ---------------------------------------------------------------------------
// Batch pipelined write — submit all writes via SubmitIoBatchAsync, wait all.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Batch pipelined write — continuous pipeline with fixed DMA window.
// ---------------------------------------------------------------------------
tl::expected<void, ErrorCode> SpdkFile::vector_write_batch(
    const BatchWriteEntry *entries, int count) {
    if (error_code_ != ErrorCode::OK) {
        return tl::make_unexpected(error_code_);
    }
    if (count <= 0) return {};

    auto &env = SpdkEnv::Instance();

    // Partition entries: large ones use chunked pipeline, small ones use batch
    std::vector<int> small_idx, large_idx;
    small_idx.reserve(count);
    large_idx.reserve(count);

    for (int i = 0; i < count; ++i) {
        size_t total = 0;
        for (int j = 0; j < entries[i].iovcnt; ++j)
            total += entries[i].iov[j].iov_len;
        size_t al = align_up(total);
        if (al >= g_pipeline_threshold)
            large_idx.push_back(i);
        else
            small_idx.push_back(i);
    }

    ErrorCode first_err = ErrorCode::OK;

    // --- Process large entries via chunked double-buffer pipeline ---
    for (int i : large_idx) {
        size_t total = 0;
        for (int j = 0; j < entries[i].iovcnt; ++j)
            total += entries[i].iov[j].iov_len;
        uint64_t abs_off =
            base_offset_ + static_cast<uint64_t>(entries[i].offset);
        size_t aligned_off =
            abs_off & ~(static_cast<size_t>(block_size_) - 1);
        size_t al = align_up(total);

        auto ec = ChunkedWriteOne(env, aligned_off, al,
                                  entries[i].iov, entries[i].iovcnt,
                                  total, block_size_);
        if (ec != ErrorCode::OK && first_err == ErrorCode::OK)
            first_err = ec;
    }

    // --- Process small entries via existing batch pipeline ---
    int small_count = static_cast<int>(small_idx.size());
    if (small_count > 0 && first_err == ErrorCode::OK) {
        size_t max_aligned = 0;
        for (int i : small_idx) {
            size_t total = 0;
            for (int j = 0; j < entries[i].iovcnt; ++j)
                total += entries[i].iov[j].iov_len;
            size_t al = align_up(total);
            if (al > max_aligned) max_aligned = al;
        }

        constexpr size_t kDmaBudgetBytes = 256ULL * 1024 * 1024;
        int max_qd_by_mem = max_aligned > 0
            ? std::max(4, static_cast<int>(kDmaBudgetBytes / max_aligned))
            : 128;
        int qd = std::min({small_count, 128, max_qd_by_mem});

        auto dma_bufs = std::make_unique<void *[]>(qd);
        int got = env.DmaPoolAllocBatch(dma_bufs.get(), max_aligned, qd,
                                        block_size_);
        if (got == 0) {
            for (int try_qd = std::max(1, qd / 2); try_qd >= 1 && got == 0;
                 try_qd = try_qd > 1 ? try_qd / 2 : 0)
                got = env.DmaPoolAllocBatch(dma_bufs.get(), max_aligned,
                                            try_qd, block_size_);
            if (got == 0) {
                LOG(ERROR) << "SpdkFile: write_batch DMA alloc 0 buffers";
                return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
            }
        }
        qd = got;

        auto reqs = std::make_unique<SpdkIoRequest[]>(qd);
        auto batch_ptrs = std::make_unique<SpdkIoRequest *[]>(qd);

        int submitted = 0, completed = 0;
        int head = 0, tail = 0;

        while (completed < small_count) {
            int batch_count = 0;
            while (submitted - completed < qd && submitted < small_count) {
                int slot = head;
                int idx = small_idx[submitted];

                size_t total = 0;
                for (int j = 0; j < entries[idx].iovcnt; ++j)
                    total += entries[idx].iov[j].iov_len;
                uint64_t abs_off =
                    base_offset_ +
                    static_cast<uint64_t>(entries[idx].offset);
                size_t aligned_off =
                    abs_off & ~(static_cast<size_t>(block_size_) - 1);
                size_t al = align_up(total);

                // Copy user data → DMA buffer on the calling thread so the
                // reactor only issues the NVMe command without any memcpy.
                char *dst = static_cast<char *>(dma_bufs[slot]);
                size_t copied = 0;
                for (int j = 0; j < entries[idx].iovcnt; ++j) {
                    std::memcpy(dst, entries[idx].iov[j].iov_base,
                                entries[idx].iov[j].iov_len);
                    dst += entries[idx].iov[j].iov_len;
                    copied += entries[idx].iov[j].iov_len;
                }
                if (al > copied)
                    std::memset(dst, 0, al - copied);

                reqs[slot].op = SpdkIoRequest::WRITE;
                reqs[slot].buf = dma_bufs[slot];
                reqs[slot].offset = aligned_off;
                reqs[slot].nbytes = al;
                reqs[slot].src_iov = nullptr;
                reqs[slot].src_iovcnt = 0;
                reqs[slot].src_data = nullptr;
                reqs[slot].src_len = 0;
                reqs[slot].dst_iov = nullptr;
                reqs[slot].dst_iovcnt = 0;

                batch_ptrs[batch_count++] = &reqs[slot];
                submitted++;
                head = (head + 1) % qd;
            }

            if (batch_count > 0)
                env.SubmitIoBatchAsync(batch_ptrs.get(), batch_count);

            while (completed < submitted) {
                if (!reqs[tail].completed.load(std::memory_order_acquire))
                    break;
                if (!reqs[tail].success && first_err == ErrorCode::OK)
                    first_err = ErrorCode::FILE_WRITE_FAIL;
                completed++;
                tail = (tail + 1) % qd;
            }
        }

        env.DmaPoolFreeBatch(dma_bufs.get(), max_aligned, qd);
    }

    if (first_err != ErrorCode::OK)
        return tl::make_unexpected(first_err);
    return {};
}

// ---------------------------------------------------------------------------
// Chunked read pipeline for a single large entry.
// Uses double-buffering: overlap DMA read of chunk[N+1] with memcpy of chunk[N].
// ---------------------------------------------------------------------------
static ErrorCode ChunkedReadOne(SpdkEnv &env, uint64_t disk_offset,
                                size_t aligned_len, size_t skip,
                                const iovec *iov, int iovcnt,
                                size_t total_user, uint32_t block_size) {
    size_t chunk_aligned =
        (g_pipeline_chunk + block_size - 1) & ~(size_t(block_size) - 1);

    void *buf[2];
    buf[0] = env.DmaPoolAlloc(chunk_aligned, block_size);
    buf[1] = env.DmaPoolAlloc(chunk_aligned, block_size);
    if (!buf[0] || !buf[1]) {
        if (buf[0]) env.DmaPoolFree(buf[0], chunk_aligned);
        if (buf[1]) env.DmaPoolFree(buf[1], chunk_aligned);
        return ErrorCode::FILE_READ_FAIL;
    }

    size_t num_chunks = (aligned_len + chunk_aligned - 1) / chunk_aligned;
    size_t disk_remaining = aligned_len;

    // Flatten iov into a linear destination cursor
    int cur_iov = 0;
    size_t cur_iov_off = 0;
    size_t user_copied = 0;
    size_t skip_remaining = skip;

    auto copy_from_dma = [&](const char *src, size_t dma_bytes) {
        // Skip alignment prefix bytes
        if (skip_remaining > 0) {
            size_t s = std::min(skip_remaining, dma_bytes);
            src += s;
            dma_bytes -= s;
            skip_remaining -= s;
        }
        // Copy to iov entries
        while (dma_bytes > 0 && cur_iov < iovcnt &&
               user_copied < total_user) {
            size_t avail =
                iov[cur_iov].iov_len - cur_iov_off;
            size_t to_copy = std::min({dma_bytes, avail,
                                       total_user - user_copied});
            std::memcpy(static_cast<char *>(iov[cur_iov].iov_base) +
                            cur_iov_off,
                        src, to_copy);
            src += to_copy;
            dma_bytes -= to_copy;
            user_copied += to_copy;
            cur_iov_off += to_copy;
            if (cur_iov_off >= iov[cur_iov].iov_len) {
                cur_iov++;
                cur_iov_off = 0;
            }
        }
    };

    ErrorCode err = ErrorCode::OK;

    // Submit first chunk
    SpdkIoRequest req0;
    size_t c0_bytes = std::min(chunk_aligned, disk_remaining);
    req0.op = SpdkIoRequest::READ;
    req0.buf = buf[0];
    req0.offset = disk_offset;
    req0.nbytes = c0_bytes;
    req0.src_data = nullptr; req0.src_iov = nullptr; req0.src_iovcnt = 0;
    req0.dst_iov = nullptr; req0.dst_iovcnt = 0;
    req0.completed.store(false, std::memory_order_relaxed);
    req0.success = false;
    env.SubmitIoAsync(&req0);

    // Wait for first chunk
    while (!req0.completed.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#else
        std::this_thread::yield();
#endif
    }
    if (!req0.success) { err = ErrorCode::FILE_READ_FAIL; goto cleanup; }

    disk_remaining -= c0_bytes;

    // Pipeline loop: for chunk i (>= 1), submit DMA for chunk[i] then memcpy chunk[i-1]
    for (size_t ci = 1; ci < num_chunks; ++ci) {
        int cur_buf = ci & 1;
        int prev_buf = 1 - cur_buf;

        size_t ci_bytes = std::min(chunk_aligned, disk_remaining);
        SpdkIoRequest req_next;
        req_next.op = SpdkIoRequest::READ;
        req_next.buf = buf[cur_buf];
        req_next.offset = disk_offset + ci * chunk_aligned;
        req_next.nbytes = ci_bytes;
        req_next.src_data = nullptr; req_next.src_iov = nullptr;
        req_next.src_iovcnt = 0;
        req_next.dst_iov = nullptr; req_next.dst_iovcnt = 0;
        req_next.completed.store(false, std::memory_order_relaxed);
        req_next.success = false;
        env.SubmitIoAsync(&req_next);

        // Memcpy previous chunk while NVMe reads current chunk
        size_t prev_bytes = (ci == 1) ? c0_bytes : chunk_aligned;
        copy_from_dma(static_cast<const char *>(buf[prev_buf]), prev_bytes);

        // Wait for current chunk DMA
        while (!req_next.completed.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
#else
            std::this_thread::yield();
#endif
        }
        if (!req_next.success) { err = ErrorCode::FILE_READ_FAIL; goto cleanup; }
        disk_remaining -= ci_bytes;
    }

    // Memcpy the last chunk
    {
        int last_buf = (num_chunks > 1) ? ((num_chunks - 1) & 1) : 0;
        size_t last_bytes = (num_chunks == 1) ? c0_bytes
            : std::min(chunk_aligned, aligned_len - (num_chunks - 1) * chunk_aligned);
        copy_from_dma(static_cast<const char *>(buf[last_buf]), last_bytes);
    }

cleanup:
    env.DmaPoolFree(buf[0], chunk_aligned);
    env.DmaPoolFree(buf[1], chunk_aligned);
    return err;
}

// ---------------------------------------------------------------------------
// Chunked write pipeline for a single large entry.
// Uses double-buffering: overlap memcpy of chunk[N+1] with DMA write of chunk[N].
// ---------------------------------------------------------------------------
static ErrorCode ChunkedWriteOne(SpdkEnv &env, uint64_t disk_offset,
                                 size_t aligned_len,
                                 const iovec *iov, int iovcnt,
                                 size_t total_user, uint32_t block_size) {
    size_t chunk_aligned =
        (g_pipeline_chunk + block_size - 1) & ~(size_t(block_size) - 1);

    void *buf[2];
    buf[0] = env.DmaPoolAlloc(chunk_aligned, block_size);
    buf[1] = env.DmaPoolAlloc(chunk_aligned, block_size);
    if (!buf[0] || !buf[1]) {
        if (buf[0]) env.DmaPoolFree(buf[0], chunk_aligned);
        if (buf[1]) env.DmaPoolFree(buf[1], chunk_aligned);
        return ErrorCode::FILE_WRITE_FAIL;
    }

    size_t num_chunks = (aligned_len + chunk_aligned - 1) / chunk_aligned;
    size_t disk_remaining = aligned_len;

    int cur_iov = 0;
    size_t cur_iov_off = 0;
    size_t user_copied = 0;

    auto copy_to_dma = [&](char *dst, size_t dma_bytes) {
        size_t filled = 0;
        while (filled < dma_bytes && cur_iov < iovcnt &&
               user_copied < total_user) {
            size_t avail = iov[cur_iov].iov_len - cur_iov_off;
            size_t to_copy = std::min({dma_bytes - filled, avail,
                                       total_user - user_copied});
            std::memcpy(dst + filled,
                        static_cast<const char *>(iov[cur_iov].iov_base) +
                            cur_iov_off,
                        to_copy);
            filled += to_copy;
            user_copied += to_copy;
            cur_iov_off += to_copy;
            if (cur_iov_off >= iov[cur_iov].iov_len) {
                cur_iov++;
                cur_iov_off = 0;
            }
        }
        if (filled < dma_bytes)
            std::memset(dst + filled, 0, dma_bytes - filled);
    };

    ErrorCode err = ErrorCode::OK;

    // Memcpy first chunk to buf[0]
    size_t c0_bytes = std::min(chunk_aligned, disk_remaining);
    copy_to_dma(static_cast<char *>(buf[0]), c0_bytes);

    // Submit first chunk DMA write
    SpdkIoRequest req_prev;
    req_prev.op = SpdkIoRequest::WRITE;
    req_prev.buf = buf[0];
    req_prev.offset = disk_offset;
    req_prev.nbytes = c0_bytes;
    req_prev.src_data = nullptr; req_prev.src_iov = nullptr;
    req_prev.src_iovcnt = 0;
    req_prev.dst_iov = nullptr; req_prev.dst_iovcnt = 0;
    req_prev.completed.store(false, std::memory_order_relaxed);
    req_prev.success = false;
    env.SubmitIoAsync(&req_prev);
    disk_remaining -= c0_bytes;

    // Pipeline loop: memcpy chunk[i] while DMA writes chunk[i-1]
    for (size_t ci = 1; ci < num_chunks; ++ci) {
        int cur_buf = ci & 1;

        size_t ci_bytes = std::min(chunk_aligned, disk_remaining);
        copy_to_dma(static_cast<char *>(buf[cur_buf]), ci_bytes);

        // Wait for previous DMA write
        while (!req_prev.completed.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
#else
            std::this_thread::yield();
#endif
        }
        if (!req_prev.success) { err = ErrorCode::FILE_WRITE_FAIL; goto cleanup; }

        // Submit current chunk DMA write
        req_prev.op = SpdkIoRequest::WRITE;
        req_prev.buf = buf[cur_buf];
        req_prev.offset = disk_offset + ci * chunk_aligned;
        req_prev.nbytes = ci_bytes;
        req_prev.src_data = nullptr; req_prev.src_iov = nullptr;
        req_prev.src_iovcnt = 0;
        req_prev.dst_iov = nullptr; req_prev.dst_iovcnt = 0;
        req_prev.completed.store(false, std::memory_order_relaxed);
        req_prev.success = false;
        env.SubmitIoAsync(&req_prev);
        disk_remaining -= ci_bytes;
    }

    // Wait for last DMA write
    while (!req_prev.completed.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#else
        std::this_thread::yield();
#endif
    }
    if (!req_prev.success) err = ErrorCode::FILE_WRITE_FAIL;

cleanup:
    env.DmaPoolFree(buf[0], chunk_aligned);
    env.DmaPoolFree(buf[1], chunk_aligned);
    return err;
}

// ---------------------------------------------------------------------------
// Batch pipelined read — continuous pipeline with fixed DMA window.
// Large entries use chunked double-buffer pipeline for DMA+memcpy overlap.
// ---------------------------------------------------------------------------
tl::expected<void, ErrorCode> SpdkFile::vector_read_batch(
    const BatchReadEntry *entries, int count) {
    if (error_code_ != ErrorCode::OK) {
        return tl::make_unexpected(error_code_);
    }
    if (count <= 0) return {};

    auto &env = SpdkEnv::Instance();

    // Partition entries: large ones use chunked pipeline, small ones use batch
    std::vector<int> small_idx, large_idx;
    small_idx.reserve(count);
    large_idx.reserve(count);

    for (int i = 0; i < count; ++i) {
        size_t total = 0;
        for (int j = 0; j < entries[i].iovcnt; ++j)
            total += entries[i].iov[j].iov_len;
        uint64_t abs_off =
            base_offset_ + static_cast<uint64_t>(entries[i].offset);
        size_t aligned_off =
            abs_off & ~(static_cast<size_t>(block_size_) - 1);
        size_t skip = abs_off - aligned_off;
        size_t al = align_up(total + skip);
        if (al >= g_pipeline_threshold)
            large_idx.push_back(i);
        else
            small_idx.push_back(i);
    }

    ErrorCode first_err = ErrorCode::OK;

    // --- Process large entries via chunked double-buffer pipeline ---
    for (int i : large_idx) {
        size_t total = 0;
        for (int j = 0; j < entries[i].iovcnt; ++j)
            total += entries[i].iov[j].iov_len;
        uint64_t abs_off =
            base_offset_ + static_cast<uint64_t>(entries[i].offset);
        size_t aligned_off =
            abs_off & ~(static_cast<size_t>(block_size_) - 1);
        size_t skip = abs_off - aligned_off;
        size_t al = align_up(total + skip);

        auto ec = ChunkedReadOne(env, aligned_off, al, skip,
                                 entries[i].iov, entries[i].iovcnt,
                                 total, block_size_);
        if (ec != ErrorCode::OK && first_err == ErrorCode::OK)
            first_err = ec;
    }

    // --- Process small entries via existing batch pipeline ---
    int small_count = static_cast<int>(small_idx.size());
    if (small_count > 0 && first_err == ErrorCode::OK) {
        size_t max_aligned = 0;
        for (int i : small_idx) {
            size_t total = 0;
            for (int j = 0; j < entries[i].iovcnt; ++j)
                total += entries[i].iov[j].iov_len;
            uint64_t abs_off =
                base_offset_ + static_cast<uint64_t>(entries[i].offset);
            size_t aligned_off =
                abs_off & ~(static_cast<size_t>(block_size_) - 1);
            size_t skip = abs_off - aligned_off;
            size_t al = align_up(total + skip);
            if (al > max_aligned) max_aligned = al;
        }

        constexpr size_t kDmaBudgetBytes = 256ULL * 1024 * 1024;
        int max_qd_by_mem = max_aligned > 0
            ? std::max(4, static_cast<int>(kDmaBudgetBytes / max_aligned))
            : 128;
        int qd = std::min({small_count, 128, max_qd_by_mem});

        auto dma_bufs = std::make_unique<void *[]>(qd);
        int got = env.DmaPoolAllocBatch(dma_bufs.get(), max_aligned, qd,
                                        block_size_);
        if (got == 0) {
            for (int try_qd = std::max(1, qd / 2); try_qd >= 1 && got == 0;
                 try_qd = try_qd > 1 ? try_qd / 2 : 0)
                got = env.DmaPoolAllocBatch(dma_bufs.get(), max_aligned,
                                            try_qd, block_size_);
            if (got == 0) {
                LOG(ERROR) << "SpdkFile: read_batch DMA alloc 0 buffers";
                return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
            }
        }
        qd = got;

        auto reqs = std::make_unique<SpdkIoRequest[]>(qd);
        auto batch_ptrs = std::make_unique<SpdkIoRequest *[]>(qd);

        // Per-slot copy info so the calling thread (not reactor) does memcpy.
        struct ReadCopyInfo {
            const iovec *dst_iov;
            int dst_iovcnt;
            size_t dst_skip;
        };
        auto copy_info = std::make_unique<ReadCopyInfo[]>(qd);

        int submitted = 0, completed = 0;
        int head = 0, tail = 0;

        while (completed < small_count) {
            int batch_count = 0;
            while (submitted - completed < qd && submitted < small_count) {
                int slot = head;
                int idx = small_idx[submitted];

                size_t total = 0;
                for (int j = 0; j < entries[idx].iovcnt; ++j)
                    total += entries[idx].iov[j].iov_len;
                uint64_t abs_off =
                    base_offset_ +
                    static_cast<uint64_t>(entries[idx].offset);
                size_t aligned_off =
                    abs_off & ~(static_cast<size_t>(block_size_) - 1);
                size_t skip = abs_off - aligned_off;
                size_t al = align_up(total + skip);

                // Save copy info — calling thread will do DMA→user memcpy.
                copy_info[slot] = {entries[idx].iov, entries[idx].iovcnt, skip};

                reqs[slot].op = SpdkIoRequest::READ;
                reqs[slot].buf = dma_bufs[slot];
                reqs[slot].offset = aligned_off;
                reqs[slot].nbytes = al;
                reqs[slot].src_data = nullptr;
                reqs[slot].src_iov = nullptr;
                reqs[slot].src_iovcnt = 0;
                reqs[slot].dst_iov = nullptr;
                reqs[slot].dst_iovcnt = 0;
                reqs[slot].dst_skip = 0;

                batch_ptrs[batch_count++] = &reqs[slot];
                submitted++;
                head = (head + 1) % qd;
            }

            if (batch_count > 0)
                env.SubmitIoBatchAsync(batch_ptrs.get(), batch_count);

            while (completed < submitted) {
                if (!reqs[tail].completed.load(std::memory_order_acquire))
                    break;
                if (reqs[tail].success) {
                    // Copy DMA → user buffers on calling thread (not reactor).
                    auto &ci = copy_info[tail];
                    const char *src =
                        static_cast<const char *>(dma_bufs[tail]) + ci.dst_skip;
                    for (int j = 0; j < ci.dst_iovcnt; ++j) {
                        std::memcpy(ci.dst_iov[j].iov_base, src,
                                    ci.dst_iov[j].iov_len);
                        src += ci.dst_iov[j].iov_len;
                    }
                } else if (first_err == ErrorCode::OK) {
                    first_err = ErrorCode::FILE_READ_FAIL;
                }
                completed++;
                tail = (tail + 1) % qd;
            }
        }

        env.DmaPoolFreeBatch(dma_bufs.get(), max_aligned, qd);
    }

    if (first_err != ErrorCode::OK)
        return tl::make_unexpected(first_err);
    return {};
}

}  // namespace mooncake

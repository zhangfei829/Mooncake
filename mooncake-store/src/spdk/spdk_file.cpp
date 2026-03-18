#include <cstring>
#include <glog/logging.h>

#include "file_interface.h"
#include "spdk/spdk_env.h"

extern "C" {
#include "spdk/bdev.h"
}

namespace mooncake {

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

    size_t max_aligned = 0;
    for (int i = 0; i < count; ++i) {
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
    int qd = std::min({count, 128, max_qd_by_mem});

    auto dma_bufs = std::make_unique<void *[]>(qd);
    int got = env.DmaPoolAllocBatch(dma_bufs.get(), max_aligned, qd,
                                    block_size_);
    if (got == 0) {
        for (int try_qd = std::max(1, qd / 2); try_qd >= 1 && got == 0;
             try_qd = try_qd > 1 ? try_qd / 2 : 0) {
            got = env.DmaPoolAllocBatch(dma_bufs.get(), max_aligned,
                                        try_qd, block_size_);
        }
        if (got == 0) {
            LOG(ERROR) << "SpdkFile: write_batch DMA alloc got 0 buffers"
                       << " after retries (requested=" << qd
                       << " × " << max_aligned << ")";
            return tl::make_unexpected(ErrorCode::FILE_WRITE_FAIL);
        }
        LOG(WARNING) << "SpdkFile: write_batch DMA alloc retried, got "
                     << got << "/" << qd << " × " << max_aligned;
    } else if (got < qd) {
        LOG(WARNING) << "SpdkFile: write_batch DMA alloc partial "
                     << got << "/" << qd << " × " << max_aligned;
    }
    qd = got;

    auto reqs = std::make_unique<SpdkIoRequest[]>(qd);
    auto batch_ptrs = std::make_unique<SpdkIoRequest *[]>(qd);

    int submitted = 0, completed = 0;
    int head = 0, tail = 0;
    ErrorCode first_err = ErrorCode::OK;

    while (completed < count) {
        int batch_count = 0;
        while (submitted - completed < qd && submitted < count) {
            int slot = head;
            int idx = submitted;

            size_t total = 0;
            for (int j = 0; j < entries[idx].iovcnt; ++j)
                total += entries[idx].iov[j].iov_len;

            uint64_t abs_off =
                base_offset_ + static_cast<uint64_t>(entries[idx].offset);
            size_t aligned_off =
                abs_off & ~(static_cast<size_t>(block_size_) - 1);
            size_t al = align_up(total);

            reqs[slot].op = SpdkIoRequest::WRITE;
            reqs[slot].buf = dma_bufs[slot];
            reqs[slot].offset = aligned_off;
            reqs[slot].nbytes = al;
            reqs[slot].src_iov = entries[idx].iov;
            reqs[slot].src_iovcnt = entries[idx].iovcnt;
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

    if (first_err != ErrorCode::OK)
        return tl::make_unexpected(first_err);
    return {};
}

// ---------------------------------------------------------------------------
// Batch pipelined read — continuous pipeline with fixed DMA window.
// Mirrors the file-level benchmark's submit/drain loop so reactors never idle.
// ---------------------------------------------------------------------------
tl::expected<void, ErrorCode> SpdkFile::vector_read_batch(
    const BatchReadEntry *entries, int count) {
    if (error_code_ != ErrorCode::OK) {
        return tl::make_unexpected(error_code_);
    }
    if (count <= 0) return {};

    auto &env = SpdkEnv::Instance();

    size_t max_aligned = 0;
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
        if (al > max_aligned) max_aligned = al;
    }

    constexpr size_t kDmaBudgetBytes = 256ULL * 1024 * 1024;
    int max_qd_by_mem = max_aligned > 0
        ? std::max(4, static_cast<int>(kDmaBudgetBytes / max_aligned))
        : 128;
    int qd = std::min({count, 128, max_qd_by_mem});

    auto dma_bufs = std::make_unique<void *[]>(qd);
    int got = env.DmaPoolAllocBatch(dma_bufs.get(), max_aligned, qd,
                                    block_size_);
    if (got == 0) {
        for (int try_qd = std::max(1, qd / 2); try_qd >= 1 && got == 0;
             try_qd = try_qd > 1 ? try_qd / 2 : 0) {
            got = env.DmaPoolAllocBatch(dma_bufs.get(), max_aligned,
                                        try_qd, block_size_);
        }
        if (got == 0) {
            LOG(ERROR) << "SpdkFile: read_batch DMA alloc got 0 buffers"
                       << " after retries (requested=" << qd
                       << " × " << max_aligned << ")";
            return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
        }
        LOG(WARNING) << "SpdkFile: read_batch DMA alloc retried, got "
                     << got << "/" << qd << " × " << max_aligned;
    } else if (got < qd) {
        LOG(WARNING) << "SpdkFile: read_batch DMA alloc partial "
                     << got << "/" << qd << " × " << max_aligned;
    }
    qd = got;

    auto reqs = std::make_unique<SpdkIoRequest[]>(qd);
    auto batch_ptrs = std::make_unique<SpdkIoRequest *[]>(qd);

    int submitted = 0, completed = 0;
    int head = 0, tail = 0;
    ErrorCode first_err = ErrorCode::OK;
    int fail_idx = -1;

    while (completed < count) {
        int batch_count = 0;
        while (submitted - completed < qd && submitted < count) {
            int slot = head;
            int idx = submitted;

            size_t total = 0;
            for (int j = 0; j < entries[idx].iovcnt; ++j)
                total += entries[idx].iov[j].iov_len;
            uint64_t abs_off =
                base_offset_ + static_cast<uint64_t>(entries[idx].offset);
            size_t aligned_off =
                abs_off & ~(static_cast<size_t>(block_size_) - 1);
            size_t skip = abs_off - aligned_off;
            size_t al = align_up(total + skip);

            reqs[slot].op = SpdkIoRequest::READ;
            reqs[slot].buf = dma_bufs[slot];
            reqs[slot].offset = aligned_off;
            reqs[slot].nbytes = al;
            reqs[slot].src_data = nullptr;
            reqs[slot].src_iov = nullptr;
            reqs[slot].src_iovcnt = 0;
            reqs[slot].dst_iov = entries[idx].iov;
            reqs[slot].dst_iovcnt = entries[idx].iovcnt;
            reqs[slot].dst_skip = skip;

            batch_ptrs[batch_count++] = &reqs[slot];
            submitted++;
            head = (head + 1) % qd;
        }

        if (batch_count > 0)
            env.SubmitIoBatchAsync(batch_ptrs.get(), batch_count);

        while (completed < submitted) {
            if (!reqs[tail].completed.load(std::memory_order_acquire))
                break;
            if (!reqs[tail].success && first_err == ErrorCode::OK) {
                first_err = ErrorCode::FILE_READ_FAIL;
                fail_idx = completed;
            }
            completed++;
            tail = (tail + 1) % qd;
        }
    }

    env.DmaPoolFreeBatch(dma_bufs.get(), max_aligned, qd);

    if (first_err != ErrorCode::OK) {
        LOG(ERROR) << "SpdkFile: read_batch I/O failed at entry " << fail_idx
                   << "/" << count << " (qd=" << qd
                   << ", buf=" << max_aligned << ")";
        return tl::make_unexpected(first_err);
    }
    return {};
}

}  // namespace mooncake

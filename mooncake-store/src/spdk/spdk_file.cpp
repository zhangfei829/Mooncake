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
    auto &env = SpdkEnv::Instance();
    for (auto &e : dma_pool_) env.DmaFree(e.buf);
    dma_pool_.clear();
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

// Pool for batch reads — avoids spdk_dma_malloc/free in hot path.
// Returns the best-fit buffer from pool; falls back to DmaMalloc.
void *SpdkFile::DmaPoolAlloc(size_t needed) {
    int best = -1;
    size_t best_size = SIZE_MAX;
    for (int i = 0; i < static_cast<int>(dma_pool_.size()); ++i) {
        if (dma_pool_[i].size >= needed && dma_pool_[i].size < best_size) {
            best = i;
            best_size = dma_pool_[i].size;
        }
    }
    if (best >= 0) {
        void *buf = dma_pool_[best].buf;
        dma_pool_[best] = dma_pool_.back();
        dma_pool_.pop_back();
        return buf;
    }
    return SpdkEnv::Instance().DmaMalloc(needed, block_size_);
}

void SpdkFile::DmaPoolFree(void *buf, size_t size) {
    dma_pool_.push_back({buf, size});
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
// Batch pipelined read — submit all reads via SubmitIoBatchAsync, wait all.
// ---------------------------------------------------------------------------
tl::expected<void, ErrorCode> SpdkFile::vector_read_batch(
    const BatchReadEntry *entries, int count) {
    if (error_code_ != ErrorCode::OK) {
        return tl::make_unexpected(error_code_);
    }
    if (count <= 0) return {};

    auto &env = SpdkEnv::Instance();

    struct ReadCtx {
        void *dma_buf;
        size_t alloc_len;
        size_t skip;
    };

    auto ctxs = std::make_unique<ReadCtx[]>(count);
    auto reqs = std::make_unique<SpdkIoRequest[]>(count);
    auto ptrs = std::make_unique<SpdkIoRequest *[]>(count);

    for (int i = 0; i < count; ++i) {
        size_t total = 0;
        for (int j = 0; j < entries[i].iovcnt; ++j)
            total += entries[i].iov[j].iov_len;

        uint64_t abs_off =
            base_offset_ + static_cast<uint64_t>(entries[i].offset);
        size_t aligned_off =
            abs_off & ~(static_cast<size_t>(block_size_) - 1);
        size_t skip = abs_off - aligned_off;
        size_t aligned_len = align_up(total + skip);

        void *dma_buf = DmaPoolAlloc(aligned_len);
        if (!dma_buf) {
            for (int j = 0; j < i; ++j) DmaPoolFree(ctxs[j].dma_buf, ctxs[j].alloc_len);
            return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
        }

        ctxs[i] = {dma_buf, aligned_len, skip};
        reqs[i].op = SpdkIoRequest::READ;
        reqs[i].buf = dma_buf;
        reqs[i].offset = aligned_off;
        reqs[i].nbytes = aligned_len;
        reqs[i].src_data = nullptr;
        reqs[i].src_iov = nullptr;
        reqs[i].src_iovcnt = 0;
        reqs[i].dst_iov = entries[i].iov;
        reqs[i].dst_iovcnt = entries[i].iovcnt;
        reqs[i].dst_skip = skip;
        ptrs[i] = &reqs[i];
    }

    env.SubmitIoBatchAsync(ptrs.get(), count);

    for (int i = 0; i < count; ++i) {
        while (!reqs[i].completed.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
#else
            std::this_thread::yield();
#endif
        }
    }

    for (int i = 0; i < count; ++i) DmaPoolFree(ctxs[i].dma_buf, ctxs[i].alloc_len);

    for (int i = 0; i < count; ++i) {
        if (!reqs[i].success)
            return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
    }
    return {};
}

}  // namespace mooncake

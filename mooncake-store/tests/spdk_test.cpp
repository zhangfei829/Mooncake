// SPDK integration tests — requires USE_SPDK build, hugepages, and root.
// Run with: sudo ./spdk_test
//
// Step 1: SpdkEnv init/shutdown
// Step 2: SpdkFile read/write/vector_write/vector_read
// Step 3: OffsetAllocatorStorageBackend via SPDK

#ifdef USE_SPDK

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>

#include "file_interface.h"
#include "spdk/spdk_env.h"
#include "storage_backend.h"

namespace mooncake::test {

static mooncake::SpdkEnvConfig MakeTestConfig() {
    mooncake::SpdkEnvConfig cfg;
    cfg.bdev_name = "TestMalloc0";
    cfg.use_malloc_bdev = true;
    cfg.malloc_num_blocks = 32768;  // 128 MB with 4K blocks
    cfg.malloc_block_size = 4096;
    return cfg;
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 1: SpdkEnv init / shutdown
// ═══════════════════════════════════════════════════════════════════════════

class SpdkEnvTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto cfg = MakeTestConfig();
        int rc = SpdkEnv::Instance().Init(cfg);
        ASSERT_EQ(rc, 0) << "SpdkEnv::Init failed";
    }
    void TearDown() override { SpdkEnv::Instance().Shutdown(); }
};

TEST_F(SpdkEnvTest, InitAndShutdown) {
    EXPECT_TRUE(SpdkEnv::Instance().IsInitialized());
    EXPECT_GT(SpdkEnv::Instance().GetBlockSize(), 0u);
    EXPECT_GT(SpdkEnv::Instance().GetBdevSize(), 0u);
    LOG(INFO) << "block_size=" << SpdkEnv::Instance().GetBlockSize()
              << " bdev_size=" << SpdkEnv::Instance().GetBdevSize();
}

TEST_F(SpdkEnvTest, DmaMallocFree) {
    void *buf = SpdkEnv::Instance().DmaMalloc(4096);
    ASSERT_NE(buf, nullptr);
    std::memset(buf, 0xAB, 4096);
    SpdkEnv::Instance().DmaFree(buf);
}

TEST_F(SpdkEnvTest, RawReadWrite) {
    auto &env = SpdkEnv::Instance();
    const size_t len = 4096;
    void *wbuf = env.DmaMalloc(len);
    void *rbuf = env.DmaMalloc(len);
    ASSERT_NE(wbuf, nullptr);
    ASSERT_NE(rbuf, nullptr);

    std::memset(wbuf, 0x42, len);
    std::memset(rbuf, 0, len);

    SpdkIoRequest wreq;
    wreq.op = SpdkIoRequest::WRITE;
    wreq.buf = wbuf;
    wreq.offset = 0;
    wreq.nbytes = len;
    env.SubmitIo(&wreq);
    EXPECT_TRUE(wreq.success);

    SpdkIoRequest rreq;
    rreq.op = SpdkIoRequest::READ;
    rreq.buf = rbuf;
    rreq.offset = 0;
    rreq.nbytes = len;
    env.SubmitIo(&rreq);
    EXPECT_TRUE(rreq.success);

    EXPECT_EQ(std::memcmp(wbuf, rbuf, len), 0);

    env.DmaFree(wbuf);
    env.DmaFree(rbuf);
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 2: SpdkFile read / write / vector_write / vector_read
// ═══════════════════════════════════════════════════════════════════════════

class SpdkFileTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto cfg = MakeTestConfig();
        int rc = SpdkEnv::Instance().Init(cfg);
        ASSERT_EQ(rc, 0);
    }
    void TearDown() override { SpdkEnv::Instance().Shutdown(); }
};

TEST_F(SpdkFileTest, SequentialWriteRead) {
    SpdkFile file("test_seq", 0, 1024 * 1024);
    ASSERT_EQ(file.get_error_code(), ErrorCode::OK);

    std::string data(8000, 'X');
    auto wres = file.write(data, data.size());
    ASSERT_TRUE(wres.has_value());
    EXPECT_EQ(wres.value(), data.size());

    SpdkFile file_r("test_seq_r", 0, 1024 * 1024);
    std::string out;
    auto rres = file_r.read(out, data.size());
    ASSERT_TRUE(rres.has_value());
    EXPECT_EQ(rres.value(), data.size());
    EXPECT_EQ(out, data);
}

TEST_F(SpdkFileTest, VectorWriteRead) {
    const uint64_t base = 4096 * 100;
    SpdkFile file_w("test_vec", base, 1024 * 1024);

    std::string part1 = "Hello, ";
    std::string part2 = "SPDK!";
    std::string expected = part1 + part2;

    iovec wiovs[2] = {
        {const_cast<char *>(part1.data()), part1.size()},
        {const_cast<char *>(part2.data()), part2.size()},
    };
    auto wres = file_w.vector_write(wiovs, 2, 0);
    ASSERT_TRUE(wres.has_value());
    EXPECT_EQ(wres.value(), expected.size());

    SpdkFile file_r("test_vec_r", base, 1024 * 1024);
    std::string rbuf(expected.size(), '\0');
    iovec riovs[1] = {{rbuf.data(), rbuf.size()}};
    auto rres = file_r.vector_read(riovs, 1, 0);
    ASSERT_TRUE(rres.has_value());
    EXPECT_EQ(rbuf, expected);
}

TEST_F(SpdkFileTest, LargeWriteRead) {
    const size_t sz = 1024 * 1024;  // 1 MB
    SpdkFile file_w("test_large", 0, sz * 2);

    std::string data(sz, '\0');
    for (size_t i = 0; i < sz; ++i) data[i] = static_cast<char>(i & 0xFF);

    auto wres = file_w.write(data, data.size());
    ASSERT_TRUE(wres.has_value());

    SpdkFile file_r("test_large_r", 0, sz * 2);
    std::string out;
    auto rres = file_r.read(out, sz);
    ASSERT_TRUE(rres.has_value());
    EXPECT_EQ(out, data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 3: OffsetAllocatorStorageBackend with SPDK
// ═══════════════════════════════════════════════════════════════════════════

class SpdkStorageBackendTest : public ::testing::Test {
   protected:
    std::string data_path;

    void SetUp() override {
        data_path = (std::filesystem::temp_directory_path() / "spdk_test_data")
                        .string();
        std::filesystem::create_directories(data_path);
    }

    void TearDown() override {
        SpdkEnv::Instance().Shutdown();
        std::filesystem::remove_all(data_path);
    }
};

TEST_F(SpdkStorageBackendTest, BasicPutGet) {
    FileStorageConfig config;
    config.storage_filepath = data_path;
    config.storage_backend_type = StorageBackendType::kOffsetAllocator;
    config.total_size_limit = 50 * 1024 * 1024;
    config.total_keys_limit = 10000;
    config.use_spdk = true;
    config.spdk_bdev_name = "TestMalloc0";

    OffsetAllocatorStorageBackend backend(config);
    auto init_res = backend.Init();
    ASSERT_TRUE(init_res.has_value()) << "Init failed";

    std::unordered_map<std::string, std::string> test_data = {
        {"key1", std::string(1024, 'A')},
        {"key2", std::string(2048, 'B')},
        {"key3", std::string(512, 'C')},
    };

    // BatchOffload
    std::unordered_map<std::string, std::vector<Slice>> batch;
    std::vector<std::unique_ptr<char[]>> bufs;
    for (auto &[k, v] : test_data) {
        auto buf = std::make_unique<char[]>(v.size());
        std::memcpy(buf.get(), v.data(), v.size());
        batch.emplace(k, std::vector<Slice>{Slice{buf.get(), v.size()}});
        bufs.push_back(std::move(buf));
    }

    auto offload_res = backend.BatchOffload(
        batch,
        [](const std::vector<std::string> &,
           std::vector<StorageObjectMetadata> &) { return ErrorCode::OK; });
    ASSERT_TRUE(offload_res.has_value());
    EXPECT_EQ(offload_res.value(), 3);

    // BatchLoad
    for (auto &[k, v] : test_data) {
        auto rbuf = std::make_unique<char[]>(v.size());
        std::unordered_map<std::string, Slice> load_slices;
        load_slices.emplace(k, Slice{rbuf.get(), v.size()});

        auto load_res = backend.BatchLoad(load_slices);
        ASSERT_TRUE(load_res.has_value()) << "Load failed for " << k;

        std::string loaded(rbuf.get(), v.size());
        EXPECT_EQ(loaded, v) << "Data mismatch for " << k;
    }
}

}  // namespace mooncake::test

#else  // !USE_SPDK

#include <gtest/gtest.h>
TEST(SpdkSkipped, NotBuiltWithSPDK) {
    GTEST_SKIP() << "USE_SPDK not enabled, skipping SPDK tests";
}

#endif  // USE_SPDK

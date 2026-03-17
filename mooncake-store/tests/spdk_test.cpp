// SPDK integration tests — requires USE_SPDK build, hugepages, and root.
// Run with: sudo ./spdk_test
//
// Step 1: SpdkEnv init/shutdown
// Step 2: SpdkFile read/write/vector_write/vector_read
// Step 3: OffsetAllocatorStorageBackend via SPDK (basic)
// Step 4: OffsetAllocatorStorageBackend via SPDK (comprehensive)

#ifdef USE_SPDK

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>
#include <thread>

#include "file_interface.h"
#include "spdk/spdk_env.h"
#include "storage_backend.h"

namespace mooncake::test {

// Global SPDK environment — initialized once, shared by all tests.
// DPDK cannot be re-initialized within the same process.
class SpdkTestEnv : public ::testing::Environment {
   public:
    void SetUp() override {
        SpdkEnvConfig cfg;
        cfg.bdev_name = "TestMalloc0";
        cfg.use_malloc_bdev = true;
        cfg.malloc_num_blocks = 32768;  // 128 MB
        cfg.malloc_block_size = 4096;
        int rc = SpdkEnv::Instance().Init(cfg);
        ASSERT_EQ(rc, 0) << "SpdkEnv global init failed";
    }
    void TearDown() override { SpdkEnv::Instance().Shutdown(); }
};

static auto *g_spdk_env [[maybe_unused]] =
    ::testing::AddGlobalTestEnvironment(new SpdkTestEnv);

// ═══════════════════════════════════════════════════════════════════════════
// Step 1: SpdkEnv init / shutdown
// ═══════════════════════════════════════════════════════════════════════════

class SpdkEnvTest : public ::testing::Test {};

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

class SpdkFileTest : public ::testing::Test {};

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

// ═══════════════════════════════════════════════════════════════════════════
// Step 4: Comprehensive OffsetAllocatorStorageBackend + SPDK
// ═══════════════════════════════════════════════════════════════════════════

class SpdkBackendFullTest : public ::testing::Test {
   protected:
    std::string data_path;
    static constexpr int64_t kCapacity = 64 * 1024 * 1024;

    void SetUp() override {
        data_path = (std::filesystem::temp_directory_path() /
                     "spdk_full_test_data")
                        .string();
        std::filesystem::create_directories(data_path);
    }

    void TearDown() override { std::filesystem::remove_all(data_path); }

    std::unique_ptr<OffsetAllocatorStorageBackend> MakeBackend(
        int64_t capacity = kCapacity, int64_t keys_limit = 100000) {
        FileStorageConfig config;
        config.storage_filepath = data_path;
        config.storage_backend_type = StorageBackendType::kOffsetAllocator;
        config.total_size_limit = capacity;
        config.total_keys_limit = keys_limit;
        config.use_spdk = true;
        config.spdk_bdev_name = "TestMalloc0";
        return std::make_unique<OffsetAllocatorStorageBackend>(config);
    }

    static auto NoOpHandler() {
        return [](const std::vector<std::string> &,
                  std::vector<StorageObjectMetadata> &) {
            return ErrorCode::OK;
        };
    }

    static int64_t Offload(
        OffsetAllocatorStorageBackend &be,
        const std::unordered_map<std::string, std::string> &kv) {
        std::unordered_map<std::string, std::vector<Slice>> batch;
        std::vector<std::unique_ptr<char[]>> bufs;
        for (auto &[k, v] : kv) {
            auto buf = std::make_unique<char[]>(v.size());
            std::memcpy(buf.get(), v.data(), v.size());
            batch.emplace(k,
                          std::vector<Slice>{Slice{buf.get(), v.size()}});
            bufs.push_back(std::move(buf));
        }
        auto res = be.BatchOffload(batch, NoOpHandler());
        return res.has_value() ? res.value() : -1;
    }

    static bool Load(OffsetAllocatorStorageBackend &be,
                     const std::string &key,
                     const std::string &expected) {
        auto rbuf = std::make_unique<char[]>(expected.size());
        std::unordered_map<std::string, Slice> slices;
        slices.emplace(key, Slice{rbuf.get(), expected.size()});
        auto res = be.BatchLoad(slices);
        if (!res.has_value()) return false;
        return std::string(rbuf.get(), expected.size()) == expected;
    }
};

// --- 4-1: Overwrite an existing key and verify new data ---
TEST_F(SpdkBackendFullTest, OverwriteKey) {
    auto be = MakeBackend();
    ASSERT_TRUE(be->Init().has_value());

    std::string v1(2048, 'X');
    ASSERT_EQ(Offload(*be, {{"mykey", v1}}), 1);
    ASSERT_TRUE(Load(*be, "mykey", v1));

    std::string v2(2048, 'Y');
    ASSERT_EQ(Offload(*be, {{"mykey", v2}}), 1);
    ASSERT_TRUE(Load(*be, "mykey", v2)) << "Should read updated value";
}

// --- 4-2: Many keys (100+), verifying allocator + sharded metadata ---
TEST_F(SpdkBackendFullTest, ManyKeys) {
    auto be = MakeBackend();
    ASSERT_TRUE(be->Init().has_value());

    constexpr int N = 200;
    std::unordered_map<std::string, std::string> all;
    for (int i = 0; i < N; ++i) {
        std::string key = "batch_key_" + std::to_string(i);
        std::string val(256 + (i % 512), static_cast<char>('A' + (i % 26)));
        all[key] = val;
    }

    ASSERT_EQ(Offload(*be, all), N);

    for (auto &[k, v] : all) {
        ASSERT_TRUE(Load(*be, k, v)) << "Mismatch for " << k;
    }
}

// --- 4-3: Various value sizes (1B, 4095, 4096, 4097, 64KB, 1MB) ---
TEST_F(SpdkBackendFullTest, VariedValueSizes) {
    auto be = MakeBackend();
    ASSERT_TRUE(be->Init().has_value());

    auto make_data = [](size_t sz) {
        std::string s(sz, '\0');
        for (size_t i = 0; i < sz; ++i) s[i] = static_cast<char>(i & 0xFF);
        return s;
    };

    struct Case {
        std::string tag;
        size_t size;
    };
    std::vector<Case> cases = {
        {"1B", 1},
        {"sub_block", 4095},
        {"exact_block", 4096},
        {"cross_block", 4097},
        {"64KB", 65536},
        {"1MB", 1024 * 1024},
    };

    for (auto &c : cases) {
        std::string key = "size_" + c.tag;
        std::string val = make_data(c.size);
        ASSERT_EQ(Offload(*be, {{key, val}}), 1)
            << "Offload failed: " << c.tag;
        ASSERT_TRUE(Load(*be, key, val))
            << "Load mismatch: " << c.tag << " (" << c.size << " bytes)";
    }
}

// --- 4-4: IsExist before and after offload ---
TEST_F(SpdkBackendFullTest, IsExist) {
    auto be = MakeBackend();
    ASSERT_TRUE(be->Init().has_value());

    auto exists = be->IsExist("phantom");
    ASSERT_TRUE(exists.has_value());
    EXPECT_FALSE(exists.value());

    ASSERT_EQ(Offload(*be, {{"phantom", std::string(128, 'Z')}}), 1);

    exists = be->IsExist("phantom");
    ASSERT_TRUE(exists.has_value());
    EXPECT_TRUE(exists.value());

    exists = be->IsExist("still_missing");
    ASSERT_TRUE(exists.has_value());
    EXPECT_FALSE(exists.value());
}

// --- 4-5: ScanMeta reports all offloaded keys ---
TEST_F(SpdkBackendFullTest, ScanMeta) {
    auto be = MakeBackend();
    ASSERT_TRUE(be->Init().has_value());

    const int N = 20;
    std::unordered_map<std::string, std::string> kv;
    for (int i = 0; i < N; ++i) {
        kv["scan_" + std::to_string(i)] = std::string(100 + i, 'M');
    }
    ASSERT_EQ(Offload(*be, kv), N);

    std::unordered_map<std::string, int64_t> scanned;
    auto scan_res = be->ScanMeta(
        [&](const std::vector<std::string> &keys,
            std::vector<StorageObjectMetadata> &metas) {
            for (size_t i = 0; i < keys.size(); ++i) {
                scanned[keys[i]] = metas[i].data_size;
            }
            return ErrorCode::OK;
        });
    ASSERT_TRUE(scan_res.has_value());
    EXPECT_EQ(static_cast<int>(scanned.size()), N);

    for (auto &[k, v] : kv) {
        auto it = scanned.find(k);
        ASSERT_NE(it, scanned.end()) << "Key missing from ScanMeta: " << k;
        EXPECT_EQ(it->second, static_cast<int64_t>(v.size()))
            << "Size mismatch in ScanMeta for " << k;
    }
}

// --- 4-6: Multi-slice offload (gather write) ---
TEST_F(SpdkBackendFullTest, MultiSliceOffload) {
    auto be = MakeBackend();
    ASSERT_TRUE(be->Init().has_value());

    std::string p1(1000, 'A');
    std::string p2(2000, 'B');
    std::string p3(500, 'C');
    std::string expected = p1 + p2 + p3;

    std::unordered_map<std::string, std::vector<Slice>> batch;
    batch["multi"] = {
        Slice{const_cast<char *>(p1.data()), p1.size()},
        Slice{const_cast<char *>(p2.data()), p2.size()},
        Slice{const_cast<char *>(p3.data()), p3.size()},
    };

    auto res = be->BatchOffload(batch, NoOpHandler());
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 1);

    ASSERT_TRUE(Load(*be, "multi", expected));
}

// --- 4-7: IsEnableOffloading respects keys limit ---
TEST_F(SpdkBackendFullTest, IsEnableOffloading) {
    auto be = MakeBackend(kCapacity, /*keys_limit=*/5);
    ASSERT_TRUE(be->Init().has_value());

    auto check = be->IsEnableOffloading();
    ASSERT_TRUE(check.has_value());
    EXPECT_TRUE(check.value()) << "Should allow offloading initially";

    std::unordered_map<std::string, std::string> kv;
    for (int i = 0; i < 5; ++i)
        kv["limit_" + std::to_string(i)] = std::string(64, 'L');
    ASSERT_EQ(Offload(*be, kv), 5);

    check = be->IsEnableOffloading();
    ASSERT_TRUE(check.has_value());
    EXPECT_FALSE(check.value()) << "Should be disabled after hitting key limit";
}

// --- 4-8: Batch load of multiple keys in one call ---
TEST_F(SpdkBackendFullTest, BatchLoadMultipleKeys) {
    auto be = MakeBackend();
    ASSERT_TRUE(be->Init().has_value());

    constexpr int N = 10;
    std::unordered_map<std::string, std::string> kv;
    for (int i = 0; i < N; ++i) {
        kv["bm_" + std::to_string(i)] =
            std::string(512 * (i + 1), static_cast<char>('a' + i));
    }
    ASSERT_EQ(Offload(*be, kv), N);

    std::unordered_map<std::string, Slice> load_map;
    std::vector<std::unique_ptr<char[]>> rbufs;
    for (auto &[k, v] : kv) {
        auto buf = std::make_unique<char[]>(v.size());
        load_map.emplace(k, Slice{buf.get(), v.size()});
        rbufs.push_back(std::move(buf));
    }

    auto res = be->BatchLoad(load_map);
    ASSERT_TRUE(res.has_value());

    for (auto &[k, v] : kv) {
        auto &sl = load_map[k];
        std::string loaded(static_cast<char *>(sl.ptr), sl.size);
        EXPECT_EQ(loaded, v) << "Mismatch for " << k;
    }
}

// --- 4-9: Concurrent offload + load from multiple threads ---
TEST_F(SpdkBackendFullTest, ConcurrentPutGet) {
    auto be = MakeBackend();
    ASSERT_TRUE(be->Init().has_value());

    constexpr int kWriters = 4;
    constexpr int kKeysPerWriter = 25;

    std::vector<std::unordered_map<std::string, std::string>> per_thread(
        kWriters);
    for (int t = 0; t < kWriters; ++t) {
        for (int i = 0; i < kKeysPerWriter; ++i) {
            std::string key =
                "t" + std::to_string(t) + "_k" + std::to_string(i);
            per_thread[t][key] = std::string(256 + i * 10, static_cast<char>('0' + t));
        }
    }

    std::atomic<int> errors{0};

    // Phase 1: parallel writes
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < kWriters; ++t) {
            threads.emplace_back([&, t]() {
                int64_t n = Offload(*be, per_thread[t]);
                if (n != kKeysPerWriter)
                    errors.fetch_add(1, std::memory_order_relaxed);
            });
        }
        for (auto &th : threads) th.join();
    }
    EXPECT_EQ(errors.load(), 0) << "Some writer threads failed";

    // Phase 2: parallel reads (each thread reads another thread's keys)
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < kWriters; ++t) {
            int read_from = (t + 1) % kWriters;
            threads.emplace_back([&, read_from]() {
                for (auto &[k, v] : per_thread[read_from]) {
                    if (!Load(*be, k, v))
                        errors.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto &th : threads) th.join();
    }
    EXPECT_EQ(errors.load(), 0) << "Some reader threads got wrong data";
}

// --- 4-10: Deterministic data pattern (byte-level integrity) ---
TEST_F(SpdkBackendFullTest, ByteLevelIntegrity) {
    auto be = MakeBackend();
    ASSERT_TRUE(be->Init().has_value());

    constexpr size_t kSize = 16384;
    std::string data(kSize, '\0');
    std::mt19937 gen(42);
    for (size_t i = 0; i < kSize; ++i) data[i] = static_cast<char>(gen() & 0xFF);

    ASSERT_EQ(Offload(*be, {{"rng_key", data}}), 1);

    auto rbuf = std::make_unique<char[]>(kSize);
    std::unordered_map<std::string, Slice> slices;
    slices.emplace("rng_key", Slice{rbuf.get(), kSize});
    ASSERT_TRUE(be->BatchLoad(slices).has_value());

    for (size_t i = 0; i < kSize; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(rbuf[i]),
                  static_cast<uint8_t>(data[i]))
            << "Byte mismatch at offset " << i;
    }
}

}  // namespace mooncake::test

#else  // !USE_SPDK

#include <gtest/gtest.h>
TEST(SpdkSkipped, NotBuiltWithSPDK) {
    GTEST_SKIP() << "USE_SPDK not enabled, skipping SPDK tests";
}

#endif  // USE_SPDK

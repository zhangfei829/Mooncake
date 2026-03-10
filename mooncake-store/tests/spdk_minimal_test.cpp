// Minimal SPDK init test — links only against SPDK, no mooncake_store.
// Validates that spdk_app_start works in isolation on this machine.

#include <cstdio>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>

extern "C" {
#include "spdk/bdev.h"
#include "spdk/env.h"
#include "spdk/event.h"
#include "spdk/thread.h"
#include "bdev/malloc/bdev_malloc.h"
}

static std::mutex g_mtx;
static std::condition_variable g_cv;
static bool g_ready = false;
static int g_rc = -1;

static void bdev_event_cb(enum spdk_bdev_event_type, struct spdk_bdev *, void *) {}

static void start_cb(void *) {
    struct malloc_bdev_opts mopts = {};
    mopts.name = const_cast<char *>("TestMalloc0");
    mopts.num_blocks = 4096;
    mopts.block_size = 4096;
    struct spdk_bdev *bd = nullptr;
    int rc = create_malloc_disk(&bd, &mopts);
    if (rc != 0) {
        fprintf(stderr, "create_malloc_disk failed: %d\n", rc);
        g_rc = rc;
        { std::lock_guard<std::mutex> lk(g_mtx); g_ready = true; }
        g_cv.notify_one();
        spdk_app_stop(-1);
        return;
    }

    struct spdk_bdev_desc *desc = nullptr;
    rc = spdk_bdev_open_ext("TestMalloc0", true, bdev_event_cb, nullptr, &desc);
    if (rc != 0) {
        fprintf(stderr, "spdk_bdev_open_ext failed: %d\n", rc);
        g_rc = rc;
        { std::lock_guard<std::mutex> lk(g_mtx); g_ready = true; }
        g_cv.notify_one();
        spdk_app_stop(-1);
        return;
    }

    auto *b = spdk_bdev_desc_get_bdev(desc);
    printf("OK: bdev opened, block_size=%u num_blocks=%lu\n",
           spdk_bdev_get_block_size(b), spdk_bdev_get_num_blocks(b));

    spdk_bdev_close(desc);
    g_rc = 0;
    { std::lock_guard<std::mutex> lk(g_mtx); g_ready = true; }
    g_cv.notify_one();
    spdk_app_stop(0);
}

int main() {
    std::thread t([]() {
        struct spdk_app_opts opts = {};
        spdk_app_opts_init(&opts, sizeof(opts));
        opts.name = "spdk_minimal";
        opts.shutdown_cb = nullptr;
        int rc = spdk_app_start(&opts, start_cb, nullptr);
        spdk_app_fini();
        if (!g_ready) {
            g_rc = rc;
            { std::lock_guard<std::mutex> lk(g_mtx); g_ready = true; }
            g_cv.notify_one();
        }
    });

    { std::unique_lock<std::mutex> lk(g_mtx); g_cv.wait(lk, [] { return g_ready; }); }
    t.join();

    printf("Result: %s (rc=%d)\n", g_rc == 0 ? "PASS" : "FAIL", g_rc);
    return g_rc;
}

# Find Python package
add_subdirectory(cachelib_memory_allocator)

set(MOONCAKE_STORE_SOURCES
    allocator.cpp
    master_service.cpp
    client_service.cpp
    client_metric.cpp
    types.cpp
    master_client.cpp
    utils.cpp
    master_metric_manager.cpp
    storage_backend.cpp
    thread_pool.cpp
    etcd_helper.cpp
    ha_helper.cpp
    segment.cpp
    transfer_task.cpp
    etcd_helper.cpp
    ha_helper.cpp
    rpc_service.cpp
    offset_allocator.cpp
    posix_file.cpp
    client_buffer.cpp
    aligned_client_buffer.cpp
    real_client.cpp
    dummy_client.cpp
    shm_helper.cpp
    http_metadata_server.cpp
    file_storage.cpp
    serialize/serializer.cpp
    serialize/serializer_backend.cpp
    utils/type_util.cpp
    utils/file_util.cpp
    task_manager.cpp
    local_hot_cache.cpp
)

set(EXTRA_LIBS "")

# Find AWS SDK
find_package(AWSSDK QUIET COMPONENTS s3)
if(AWSSDK_FOUND)
    message(STATUS "AWS SDK found: ${AWSSDK_VERSION}")
    set(HAVE_AWS_SDK TRUE)
    # Add S3 related source files
    list(APPEND MOONCAKE_STORE_SOURCES utils/s3_helper.cpp)
else()
    message(STATUS "AWS SDK not found, S3 functionality will be disabled")
    set(HAVE_AWS_SDK FALSE)
endif()

# Find zstd library
find_library(ZSTD_LIBRARY NAMES zstd)
if(NOT ZSTD_LIBRARY)
    message(FATAL_ERROR "zstd library not found")
endif()

set(EXTRA_LIBS ${ZSTD_LIBRARY})

# If AWS SDK is found, add it to EXTRA_LIBS
if(HAVE_AWS_SDK)
    list(APPEND EXTRA_LIBS ${AWSSDK_LINK_LIBRARIES})
    add_definitions(-DHAVE_AWS_SDK)
    if(AWS_S3_LIB AND AWS_CORE_LIB)
        list(APPEND EXTRA_LIBS ${AWS_S3_LIB} ${AWS_CORE_LIB})
        add_definitions(-DHAVE_AWS_SDK)
    endif()
endif()

if(USE_3FS)
  add_subdirectory(hf3fs)
  list(APPEND MOONCAKE_STORE_SOURCES  ${HF3FS_SOURCES})
  find_library(HF3FS_API_LIB hf3fs_api_shared PATHS /usr/lib NO_DEFAULT_PATH)
  if(NOT HF3FS_API_LIB)
    message(FATAL_ERROR "hf3fs_api_shared library not found in /usr/lib")
  endif()
  set(EXTRA_LIBS ${HF3FS_API_LIB})
endif()

# io_uring support (auto-detected)
find_library(URING_LIB uring PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64)
find_path(URING_INCLUDE liburing.h PATHS /usr/include /usr/local/include)
if(URING_LIB AND URING_INCLUDE)
  message(STATUS "io_uring: Enabled for mooncake_store")
  list(APPEND MOONCAKE_STORE_SOURCES uring_file.cpp)
  list(APPEND EXTRA_LIBS ${URING_LIB})
else()
  message(STATUS "io_uring: Disabled (liburing not found)")
endif()

# SPDK support (opt-in via -DUSE_SPDK=ON -DSPDK_DIR=<path>)
if(USE_SPDK)
  if(NOT SPDK_DIR)
    set(SPDK_DIR "${CMAKE_SOURCE_DIR}/spdk" CACHE PATH "Path to SPDK source/build tree")
  endif()
  set(SPDK_INCLUDE_DIR "${SPDK_DIR}/build/include" "${SPDK_DIR}/include"
                       "${SPDK_DIR}/module" "${SPDK_DIR}/dpdk/build/include")
  set(SPDK_LIB_DIR     "${SPDK_DIR}/build/lib" "${SPDK_DIR}/dpdk/build/lib")

  find_path(SPDK_INCLUDE spdk/bdev.h PATHS ${SPDK_INCLUDE_DIR} NO_DEFAULT_PATH)
  if(NOT SPDK_INCLUDE)
    message(FATAL_ERROR "SPDK headers not found. Set SPDK_DIR to your SPDK build tree.")
  endif()

  # Use SPDK's own pkg-config for correct library ordering and deps.
  # This is the officially recommended way to link SPDK.
  find_package(PkgConfig REQUIRED)
  set(SPDK_PC_PATH "${SPDK_DIR}/build/lib/pkgconfig:${SPDK_DIR}/dpdk/build/lib/pkgconfig")
  set(ENV{PKG_CONFIG_PATH} "${SPDK_PC_PATH}:$ENV{PKG_CONFIG_PATH}")

  execute_process(
    COMMAND pkg-config --libs --static spdk_event spdk_event_bdev spdk_bdev spdk_bdev_malloc spdk_env_dpdk
    OUTPUT_VARIABLE SPDK_LDFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SPDK_PC_RC
  )

  # Include ALL SPDK + DPDK libs (same as SPDK's own build for examples).
  # Only exclude ut/ut_mock (test stubs requiring CUnit + --wrap).
  file(GLOB SPDK_SELECTED_LIBS "${SPDK_DIR}/build/lib/libspdk_*.a")
  file(GLOB DPDK_ALL_LIBS "${SPDK_DIR}/dpdk/build/lib/librte_*.a")
  list(FILTER SPDK_SELECTED_LIBS EXCLUDE REGEX
       "libspdk_ut\\.a$|libspdk_ut_mock|libspdk_iscsi|libspdk_scsi|libspdk_vhost|libspdk_nvmf|libspdk_nbd")

  if(TRUE)
    message(STATUS "SPDK: Using whole-archive DPDK + start-group SPDK linking")

    add_subdirectory(spdk)
    list(APPEND MOONCAKE_STORE_SOURCES ${SPDK_SOURCES})
    link_directories("${SPDK_DIR}/isa-l/.libs" "${SPDK_DIR}/isa-l-crypto/.libs" "${SPDK_DIR}/build/lib")
    list(APPEND EXTRA_LIBS
      -Wl,--whole-archive ${SPDK_SELECTED_LIBS} ${DPDK_ALL_LIBS} -Wl,--no-whole-archive
      isal isal_crypto ssl crypto numa dl uuid pthread aio keyutils fuse3 m)
  else()
    message(STATUS "SPDK: pkg-config unavailable, falling back to manual linking")
    # Minimal set — no NVMe (avoids fuse3 dep), no VMD
    set(SPDK_LINK_LIBS
      spdk_bdev spdk_bdev_malloc
      spdk_env_dpdk spdk_thread spdk_log spdk_util spdk_json spdk_jsonrpc
      spdk_rpc spdk_sock spdk_sock_posix spdk_notify spdk_trace
      rte_eal rte_mempool rte_ring rte_telemetry rte_kvargs
    )
    set(SPDK_FOUND_LIBS "")
    foreach(lib ${SPDK_LINK_LIBS})
      find_library(SPDK_LIB_${lib} ${lib} PATHS ${SPDK_LIB_DIR} NO_DEFAULT_PATH)
      if(SPDK_LIB_${lib})
        list(APPEND SPDK_FOUND_LIBS ${SPDK_LIB_${lib}})
      endif()
    endforeach()

    # ISA-L (CRC/XOR acceleration, required by spdk_util)
    find_library(ISAL_LIB isal PATHS
      "${SPDK_DIR}/isa-l/.libs" "${SPDK_DIR}/build/lib" /usr/lib /usr/lib64)

    add_subdirectory(spdk)
    list(APPEND MOONCAKE_STORE_SOURCES ${SPDK_SOURCES})
    list(APPEND EXTRA_LIBS
      -Wl,--start-group ${SPDK_FOUND_LIBS} -Wl,--end-group
      ${ISAL_LIB} numa dl uuid pthread ssl crypto aio)
  endif()
  message(STATUS "SPDK: Enabled for mooncake_store (SPDK_DIR=${SPDK_DIR})")
endif()

# The cache_allocator library
include_directories(${Python3_INCLUDE_DIRS})
add_library(mooncake_store ${MOONCAKE_STORE_SOURCES})
# Note: transfer_engine is PRIVATE to avoid propagating its dependencies (e.g., vendor-specific hardware)
# to targets that don't need it (e.g., mooncake_master).
# Targets that need transfer_engine should link it explicitly.
target_link_libraries(mooncake_store
    PUBLIC
        cachelib_memory_allocator
        ${ETCD_WRAPPER_LIB}
        glog::glog
        gflags::gflags
        ${EXTRA_LIBS}
        asio_shared
    PRIVATE
        transfer_engine
)
if (STORE_USE_ETCD)
    add_dependencies(mooncake_store build_etcd_wrapper)
endif()

if(URING_LIB AND URING_INCLUDE)
    target_compile_definitions(mooncake_store PUBLIC USE_URING)
    target_include_directories(mooncake_store PRIVATE ${URING_INCLUDE})
endif()

if(USE_SPDK)
    target_compile_definitions(mooncake_store PUBLIC USE_SPDK)
    target_include_directories(mooncake_store PUBLIC ${SPDK_INCLUDE_DIR})
endif()

if (USE_ASCEND_DIRECT OR USE_UBSHMEM)
    set(ACL_RUNTIME_HEADER_PATH "${ASCEND_INCLUDE_DIR}/acl/acl_rt.h")
    if(EXISTS "${ACL_RUNTIME_HEADER_PATH}")
        file(READ "${ACL_RUNTIME_HEADER_PATH}" ACL_RUNTIME_CONTENT)
        if("${ACL_RUNTIME_CONTENT}" MATCHES "ACL_MEM_P2P_HUGE1G")
            message(STATUS "ACL_MEM_P2P_HUGE1G exist.")
            add_compile_definitions(ASCEND_SUPPORT_FABRIC_MEM)
        else()
            message(STATUS "ACL_MEM_P2P_HUGE1G is not exist.")
        endif()
    else()
        message(WARNING "Acl runtime header file is not exist")
    endif()
endif()

if (BUILD_SHARED_LIBS)
  install(TARGETS mooncake_store DESTINATION lib)
endif()

# Master binary
add_executable(mooncake_master master.cpp)

set(MASTER_EXTRA_INCS)
set(MASTER_EXTRA_LIBS)

if (STORE_USE_JEMALLOC)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(JEMALLOC REQUIRED jemalloc)
    list(APPEND MASTER_EXTRA_INCS ${JEMALLOC_INCLUDE_DIRS})
    list(APPEND MASTER_EXTRA_LIBS ${JEMALLOC_STATIC_LIBRARIES})
endif()

target_include_directories(mooncake_master PRIVATE ${MASTER_EXTRA_INCS})
target_link_libraries(mooncake_master PRIVATE
    mooncake_store
    cachelib_memory_allocator
    pthread
    ibverbs
    mooncake_common
    ${ETCD_WRAPPER_LIB}
    ${MASTER_EXTRA_LIBS}
    asio_shared
)

if (STORE_USE_ETCD)
    add_dependencies(mooncake_master build_etcd_wrapper)
endif()

target_compile_options(mooncake_master PRIVATE -Os)
target_link_options(mooncake_master PRIVATE -Os -s)

# Client server binary
add_executable(mooncake_client real_client_main.cpp)
# Client needs transfer_engine for data transfer operations
target_link_libraries(mooncake_client PRIVATE mooncake_store transfer_engine asio_shared)
target_compile_options(mooncake_client PRIVATE -Os)
target_link_options(mooncake_client PRIVATE -Os -s)

install(TARGETS mooncake_master mooncake_client DESTINATION bin)

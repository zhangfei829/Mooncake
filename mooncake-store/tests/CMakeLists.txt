function(add_store_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PUBLIC
        mooncake_store
        transfer_engine
        cachelib_memory_allocator
        ${ETCD_WRAPPER_LIB}
        glog
        ibverbs
        gtest
        gtest_main
        pthread
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_store_test(buffer_allocator_test buffer_allocator_test.cpp)
add_store_test(allocation_strategy_test allocation_strategy_test.cpp)
add_store_test(eviction_strategy_test eviction_strategy_test.cpp)
add_store_test(master_service_test master_service_test.cpp)
add_store_test(master_service_ssd_test master_service_ssd_test.cpp)
add_store_test(master_service_ssd_test_for_snapshot master_service_ssd_test_for_snapshot.cpp)
add_store_test(client_integration_test client_integration_test.cpp)
add_store_test(cxl_client_integration_test cxl_client_integration_test.cpp)
add_store_test(master_metrics_test master_metrics_test.cpp)
add_store_test(posix_file_test posix_file_test.cpp)
add_store_test(thread_pool_test thread_pool_test.cpp)
add_store_test(transfer_task_test transfer_task_test.cpp)
add_store_test(segment_test segment_test.cpp)
add_store_test(offset_allocator_test offset_allocator_test.cpp)
add_store_test(utils_test utils_test.cpp)
add_store_test(client_buffer_test client_buffer_test.cpp)
add_store_test(client_local_hot_cache_test client_local_hot_cache_test.cpp)
add_store_test(pybind_client_test pybind_client_test.cpp)
add_store_test(ipv6_client_test ipv6_client_test.cpp)
add_store_test(client_metrics_test client_metrics_test.cpp)
add_store_test(serializer_test serializer_test.cpp)
add_store_test(zstd_util_test zstd_util_test.cpp)
add_store_test(local_file_backend_test local_file_backend_test.cpp)
add_store_test(file_util_test file_util_test.cpp)
add_store_test(snapshot_child_process_test snapshot_child_process_test.cpp)
add_store_test(master_service_test_for_snapshot master_service_test_for_snapshot.cpp)
add_store_test(non_ha_reconnect_test non_ha_reconnect_test.cpp)
add_store_test(storage_backend_test storage_backend_test.cpp)
add_store_test(mutex_test mutex_test.cpp)
add_store_test(file_storage_test file_storage_test.cpp)
add_store_test(task_manager_test task_manager_test.cpp)
add_store_test(task_executor_test task_executor_test.cpp)
add_store_test(task_integration_test task_integration_test.cpp)
add_store_test(spdk_test spdk_test.cpp)

if(USE_SPDK)
  if(NOT SPDK_DIR)
    set(SPDK_DIR "${CMAKE_SOURCE_DIR}/spdk")
  endif()
  set(_SPDK_INC "${SPDK_DIR}/build/include" "${SPDK_DIR}/include"
                "${SPDK_DIR}/module" "${SPDK_DIR}/dpdk/build/include")

  # All SPDK libs except test utilities / NVMe (avoids fuse3, CUnit deps).
  # DPDK libs need --whole-archive for constructor-based driver registration.
  file(GLOB _ALL_SPDK_LIBS "${SPDK_DIR}/build/lib/libspdk_*.a")
  file(GLOB _ALL_DPDK_LIBS "${SPDK_DIR}/dpdk/build/lib/librte_*.a")
  list(FILTER _ALL_SPDK_LIBS EXCLUDE REGEX
    "libspdk_ut\\.a|libspdk_ut_mock|libspdk_keyring|libspdk_nvme\\.a|libspdk_bdev_nvme|libspdk_vmd")

  add_executable(spdk_minimal_test spdk_minimal_test.cpp)
  target_include_directories(spdk_minimal_test PRIVATE ${_SPDK_INC})
  link_directories("${SPDK_DIR}/isa-l/.libs" "${SPDK_DIR}/isa-l-crypto/.libs" "${SPDK_DIR}/build/lib")
  target_link_libraries(spdk_minimal_test PRIVATE
    -Wl,--start-group ${_ALL_SPDK_LIBS} -Wl,--end-group
    -Wl,--whole-archive ${_ALL_DPDK_LIBS} -Wl,--no-whole-archive
    isal isal_crypto ssl crypto numa dl uuid pthread aio)
endif()
add_store_test(dummy_client_get_buffer_test dummy_client_get_buffer_test.cpp)
add_subdirectory(e2e)

add_executable(high_availability_test high_availability_test.cpp)
target_link_libraries(high_availability_test PUBLIC mooncake_store transfer_engine cachelib_memory_allocator ${ETCD_WRAPPER_LIB} glog gtest gtest_main pthread)
if (STORE_USE_ETCD)
    add_test(NAME high_availability_test COMMAND high_availability_test)
endif()

add_executable(stress_workload_test stress_workload_test.cpp)
target_link_libraries(stress_workload_test PUBLIC
    mooncake_store
    transfer_engine
    cachelib_memory_allocator
    ${ETCD_WRAPPER_LIB}
    mooncake_common
    glog
    gflags
    pthread
)

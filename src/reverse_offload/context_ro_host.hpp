/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_RO_HOST_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_RO_HOST_HPP_

#include "../context.hpp"

namespace rocshmem {

class ROContextWindowInfo {
 public:
  /**
   * @brief Constructor with default members
   */
  ROContextWindowInfo() = default;

  /**
   * @brief Constructor with initialized members
   *
   * @param[in] team pointer used to track team info
   * @param[in] team_info information about participating PEs
   */
  ROContextWindowInfo(MPI_Comm comm_world, SymmetricHeap *heap);

  /**
   * @brief Destructor
   */
  __host__ ~ROContextWindowInfo();

  /**
   * @brief Retrieve a pointer to the internal WindowInfo
   *
   * @return WindowInfo pointer
   */
  WindowInfo *get() { return window_info_; }

 private:
  /**
   * @brief Pointer to the WindowInfo object that manages the MPI Window for
   * this context
   */
  WindowInfo *window_info_{nullptr};
};

class ROHostContext : public Context {
 public:
  __host__ void register_memory(void *ptr, size_t size);

  __host__ void deregister_memory(void *ptr);

 public:
  __host__ ROHostContext(Backend *b, int64_t options);

  __host__ ~ROHostContext();

  /* Pointer to the backend's host interface */
  HostInterface *host_interface = nullptr;

  /* An MPI Window implements a context */
  WindowInfo *context_window_info = nullptr;

  /**************************************************************************
   ****************************** HOST METHODS ******************************
   *************************************************************************/
  template <typename T>
  __host__ void p(T *dest, T value, int pe);

  template <typename T>
  __host__ T g(const T *source, int pe);

  template <typename T>
  __host__ void put(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __host__ void get(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __host__ void put_nbi(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __host__ void get_nbi(T *dest, const T *source, size_t nelems, int pe);

  __host__ void putmem(void *dest, const void *source, size_t nelems, int pe);

  __host__ void getmem(void *dest, const void *source, size_t nelems, int pe);

  __host__ void putmem_nbi(void *dest, const void *source, size_t nelems,
                           int pe);

  __host__ void getmem_nbi(void *dest, const void *source, size_t size, int pe);

  template <typename T>
  __host__ void amo_add(void *dst, T value, int pe);

  template <typename T>
  __host__ void amo_cas(void *dst, T value, T cond, int pe);

  template <typename T>
  __host__ T amo_fetch_add(void *dst, T value, int pe);

  template <typename T>
  __host__ T amo_fetch_cas(void *dst, T value, T cond, int pe);

  __host__ void fence();

  __host__ void quiet();

  __host__ void barrier_all();

  __host__ void sync_all();

  template <typename T>
  __host__ void broadcast(T *dest, const T *source, int nelems, int pe_root,
                          int pe_start, int log_pe_stride, int pe_size,
                          long *p_sync);  // NOLINT(runtime/int)

  template <typename T>
  __host__ void broadcast(roc_shmem_team_t team, T *dest, const T *source,
                          int nelems, int pe_root);

  template <typename T, ROC_SHMEM_OP Op>
  __host__ void to_all(T *dest, const T *source, int nreduce, int pe_start,
                       int log_pe_stride, int pe_size, T *p_wrk,
                       long *p_sync);  // NOLINT(runtime/int)

  template <typename T, ROC_SHMEM_OP Op>
  __host__ void to_all(roc_shmem_team_t team, T *dest, const T *source,
                       int nreduce);

  template <typename T>
  __host__ void wait_until(T *ptr, roc_shmem_cmps cmp, T val);

  template <typename T>
  __host__ void wait_until_all(T* ptr, size_t nelems,
                               const int *status,
                               roc_shmem_cmps cmp, T val);

  template <typename T>
  __host__ size_t wait_until_any(T* ptr, size_t nelems,
                                 const int *status,
                                 roc_shmem_cmps cmp, T val);

  template <typename T>
  __host__ size_t wait_until_some(T* ptr, size_t nelems,
                                size_t* indices,
                                const int *status,
                                roc_shmem_cmps cmp, T val);

  template <typename T>
  __host__ void wait_until_all_vector(T* ptr, size_t nelems,
                                      const int *status,
                                      roc_shmem_cmps cmp, T* vals);

  template <typename T>
  __host__ size_t wait_until_any_vector(T* ptr, size_t nelems,
                                        const int *status,
                                        roc_shmem_cmps cmp, T* vals);

  template <typename T>
  __host__ size_t wait_until_some_vector(T* ptr, size_t nelems,
                                       size_t* indices,
                                       const int *status,
                                       roc_shmem_cmps cmp, T* vals);

  template <typename T>
  __host__ int test(T *ptr, roc_shmem_cmps cmp, T val);
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_RO_HOST_HPP_
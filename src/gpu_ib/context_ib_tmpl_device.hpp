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

#ifndef LIBRARY_SRC_GPU_IB_CONTEXT_IB_TMPL_DEVICE_HPP_
#define LIBRARY_SRC_GPU_IB_CONTEXT_IB_TMPL_DEVICE_HPP_

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem.hpp"
#include "context_ib_device.hpp"
#include "gpu_ib_team.hpp"
#include "queue_pair.hpp"
#include "../util.hpp"
#include "../rocshmem_calc.hpp"

namespace rocshmem {

template <typename T, ROCSHMEM_OP Op>
__device__ void compute_reduce(T *src, T *dst, int size, int wg_id,
                               int wg_size) {
  for (size_t i = wg_id; i < size; i += wg_size) {
    OpWrap<Op>::Calc(src, dst, i);
  }
  __syncthreads();
}

template <typename T>
__device__ void GPUIBContext::p(T *dest, T value, int pe) {
  putmem_nbi(dest, &value, sizeof(T), pe);
}

template <typename T, ROCSHMEM_OP Op>
__device__ void GPUIBContext::internal_ring_allreduce(
    T *dst, const T *src, int nelems, [[maybe_unused]] int PE_start,
    [[maybe_unused]] int logPE_stride, [[maybe_unused]] int PE_size, T *pWrk,
    long *pSync,  // NOLINT(runtime/int)
    int n_seg, int seg_size, int chunk_size) {
  int off_seg, off_send, off_recv;
  int send_pe = (my_pe + 1) % num_pes;
  long wait_val;  // NOLINT(runtime/int)

  int wg_size = get_flat_block_size();
  int wg_id = get_flat_block_id();

  for (size_t i = wg_id; i < nelems; i += wg_size) {
    dst[i] = src[i];
  }
  __syncthreads();

  for (size_t seg = 0; seg < n_seg; seg++) {
    off_seg = seg * seg_size;
    for (int round = 0; round < num_pes - 1; round++) {
      off_send = (((my_pe + 1 - round + 2 * num_pes) % num_pes) * chunk_size);
      off_recv = (((my_pe - round + 2 * num_pes) % num_pes) * chunk_size);

      putmem_nbi_wg(reinterpret_cast<void *>(&pWrk[off_send]),
                    reinterpret_cast<void *>(&dst[off_send + off_seg]),
                    chunk_size * sizeof(T), send_pe);

      if (is_thread_zero_in_block()) {
        fence();

        wait_val = seg + 100;
        p(&pSync[round], wait_val, send_pe);

        wait_until(&pSync[round], ROCSHMEM_CMP_EQ, wait_val);
        __threadfence();
      }
      __syncthreads();
      compute_reduce<T, Op>(&pWrk[off_recv], &dst[off_seg + off_recv],
                            chunk_size, wg_id, wg_size);
    }
    for (size_t round = num_pes - 1; round < 2 * num_pes - 2; round++) {
      int off_send2 =
          (((my_pe + 1 - round + 2 * num_pes) % num_pes) * chunk_size);
      putmem_nbi_wg(reinterpret_cast<void *>(&dst[off_send2 + off_seg]),
                    reinterpret_cast<void *>(&dst[off_send2 + off_seg]),
                    chunk_size * sizeof(T), send_pe);

      if (is_thread_zero_in_block()) {
        fence();
        wait_val = seg + 100;
        p(&pSync[round], wait_val, send_pe);
        wait_until(&pSync[round], ROCSHMEM_CMP_EQ, wait_val);
      }
      __syncthreads();
    }
  }
  __syncthreads();
  for (size_t i = wg_id; i < 2 * num_pes - 2; i += wg_size) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }
  __syncthreads();
}

template <typename T, ROCSHMEM_OP Op>
__device__ void GPUIBContext::internal_direct_allreduce(
    T *dst, const T *src, int nelems, int PE_start, int logPE_stride,
    int PE_size, T *pWrk,
    long *pSync) {  // NOLINT(runtime/int)
  int stride = 1 << logPE_stride;
  int finish = PE_start + stride * PE_size;
  int pe = my_pe;

  int wg_id = get_flat_block_id();
  int wg_size = get_flat_block_size();

  for (int i = wg_id; i < nelems; i += wg_size) {
    dst[i] = src[i];
  }
  __syncthreads();

  for (int i = PE_start; i < finish; i += stride) {
    if (i != pe) {
      putmem_nbi_wg(&pWrk[pe * nelems], reinterpret_cast<const void *>(src),
                    nelems * sizeof(T), i);

      if (is_thread_zero_in_block()) {
        fence();
        p(&pSync[pe], 1L, i);
      }
      __syncthreads();
    }
  }

  // Do the compute and pSync reset in parallel.

  for (int i = PE_start; i < finish; i += stride) {
    if (i != pe) {
      // Wait for leader thread to see that the buffer is ready.
      if (is_thread_zero_in_block()) {
        wait_until(&pSync[i], ROCSHMEM_CMP_EQ, 1L);
      }
      __syncthreads();

      T *ptr = &pWrk[i * nelems];
      compute_reduce<T, Op>(ptr, dst, nelems, wg_id, wg_size);
    }
  }

  __syncthreads();

  for (int i = wg_id; i < num_pes; i += wg_size) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }

  __syncthreads();
}

template <typename T, ROCSHMEM_OP Op>
__device__ void GPUIBContext::to_all(rocshmem_team_t team, T *dest,
                                     const T *source, int nreduce) {
  GPUIBTeam *team_obj = reinterpret_cast<GPUIBTeam *>(team);

  double dbl_log_pe_stride = team_obj->tinfo_wrt_world->log_stride;
  int log_pe_stride = static_cast<int>(dbl_log_pe_stride);
  /**
   * Ensure that the stride is a multiple of 2 for GPU_IB.
   * TODO(bpotter): enable GPU_IB to work with non-power-of-2 strides
   * and remove this assert.
   */
  assert((dbl_log_pe_stride - log_pe_stride) == 0);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;

  long *p_sync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);

  to_all<T, Op>(dest, source, nreduce, pe_start, log_pe_stride, pe_size, pWrk,
                p_sync);
}

template <typename T, ROCSHMEM_OP Op>
__device__ void GPUIBContext::to_all(T *dest, const T *source, int nreduce,
                                     int PE_start, int logPE_stride,
                                     int PE_size, T *pWrk,
                                     long *pSync) {  // NOLINT(runtime/int)
  size_t direct_pWrk = num_pes * nreduce;
  size_t direct_pSync = num_pes;

  size_t ring_pSync = 2 * num_pes;

  size_t provided_pWrk =
      max(nreduce / 2 + 1, ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE);
  size_t provided_pSync = ROCSHMEM_REDUCE_SYNC_SIZE;

  // TODO(bpotter):
  // We basically do a direct reduce if pWrk is big enough, else we
  // give up. In the future we will want to design algorithms to work
  // with nreduce/2 + 1 space, which would cover every case per the
  // standard.
  if (provided_pWrk >= direct_pWrk && provided_pSync >= direct_pSync) {
    internal_direct_allreduce<T, Op>(dest, source, nreduce, PE_start,
                                     logPE_stride, PE_size, pWrk, pSync);
  } else {
    if (ring_pSync <= ROCSHMEM_REDUCE_SYNC_SIZE) {
      int chunk_size = 1024;
      size_t ring_pWrk = chunk_size * num_pes;
      if (provided_pWrk < ring_pWrk) {
        ring_pWrk = max(nreduce / 2,  // NOLINT
                        ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE);
        chunk_size = ring_pWrk / num_pes;
      }
      int seg_size = ring_pWrk;
      int n_seg = nreduce / seg_size;
      if (n_seg == 0) {
        n_seg = 1;
        seg_size = nreduce;
        chunk_size = seg_size / num_pes;
      }
      internal_ring_allreduce<T, Op>(dest, source, nreduce, PE_start,
                                     logPE_stride, PE_size, pWrk, pSync, n_seg,
                                     seg_size, chunk_size);
    } else {
      GPU_DPRINTF("Unsupported reduction size for gpu_ib.\n");
    }
  }
}

template <typename T>
__device__ void GPUIBContext::put(T *dest, const T *source, size_t nelems,
                                  int pe) {
  putmem(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ T GPUIBContext::g(const T *source, int pe) {
  T ret;
  auto *src_const_cast = reinterpret_cast<const char *>(source);
  uint64_t L_offset = const_cast<char *>(src_const_cast) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    ipcImpl_.ipcCopy(&ret, ipcImpl_.ipc_bases[pe] + L_offset, sizeof(T));
    return ret;
  } else {
    int thread_id = get_flat_block_id();
    int block_size = get_flat_block_size();
    int offset = ctx_idx * block_size + thread_id;

    char *base_dest = g_ret;
    char *dest = &base_dest[offset * sizeof(int64_t)];
    size_t nelems = sizeof(T);

    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return ret;
    }
    getQueuePair(pe)->get_nbi<THREAD>(base_heap[pe] + L_offset, dest, nelems,
                                      pe, true);
    getQueuePair(pe)->quiet_single<THREAD>();
    getQueuePair(my_pe)->hdp_policy->hdp_flush();
    __threadfence();
    ret = *(reinterpret_cast<T *>(dest));
    return ret;
  }
  return ret;
}

template <typename T>
__device__ void GPUIBContext::put_nbi(T *dest, const T *source, size_t nelems,
                                      int pe) {
  putmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ void GPUIBContext::get(T *dest, const T *source, size_t nelems,
                                  int pe) {
  getmem(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ void GPUIBContext::get_nbi(T *dest, const T *source, size_t nelems,
                                      int pe) {
  getmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ T GPUIBContext::amo_fetch_add(void *dst, T value, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    return ipcImpl_.ipcAMOFetchAdd(
        reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
  } else {
    auto *qp = getQueuePair(pe);
    return qp->atomic_fetch(base_heap[pe] + L_offset, value, 0, pe, true,
                            MLX5_OPCODE_ATOMIC_FA);
  }
}

template <typename T>
__device__ T GPUIBContext::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    return ipcImpl_.ipcAMOFetchCas(
        reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), cond, value);
  } else {
    auto *qp = getQueuePair(pe);
    return qp->atomic_fetch(base_heap[pe] + L_offset, value, cond, pe, true,
                            MLX5_OPCODE_ATOMIC_CS);
  }
}

template <typename T>
__device__ void GPUIBContext::amo_add(void *dst, T value, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    ipcImpl_.ipcAMOAdd(reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset),
                       value);
  } else {
    auto *qp = getQueuePair(pe);
    qp->atomic_nofetch(base_heap[pe] + L_offset, value, 0, pe, true,
                       MLX5_OPCODE_ATOMIC_FA);
  }
}

template <typename T>
__device__ void GPUIBContext::amo_set(void *dst, T value, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];

  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    ipcImpl_.ipcAMOSet(reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset),
                       value);
  } else {
    auto *qp = getQueuePair(pe);

    // Guess that the remote memory is zero by setting condition to zero.
    // The compare-and-swap loop will execute at least twice if wrong.
    // It may run additional times if contention on memory location.
    T ret_val;
    T cond = 0;
    while ((ret_val = qp->atomic_fetch(base_heap[pe] + L_offset, value, cond,
                                       pe, true, MLX5_OPCODE_ATOMIC_CS))) {
      if (ret_val == cond) {
        break;
      }
      cond = ret_val;
    }
  }
}

template <typename T>
__device__ T GPUIBContext::amo_swap(void *dst, T value, int pe) {
  assert(false);
  return 0;
}

template <typename T>
__device__ T GPUIBContext::amo_fetch_and(void *dst, T value, int pe) {
  assert(false);
  return 0;
}

template <typename T>
__device__ void GPUIBContext::amo_and(void *dst, T value, int pe) {
  assert(false);
}

template <typename T>
__device__ T GPUIBContext::amo_fetch_or(void *dst, T value, int pe) {
  assert(false);
  return 0;
}

template <typename T>
__device__ void GPUIBContext::amo_or(void *dst, T value, int pe) {
  assert(false);
}

template <typename T>
__device__ T GPUIBContext::amo_fetch_xor(void *dst, T value, int pe) {
  assert(false);
  return 0;
}

template <typename T>
__device__ void GPUIBContext::amo_xor(void *dst, T value, int pe) {
  assert(false);
}

template <typename T>
__device__ void GPUIBContext::amo_cas(void *dst, T value, T cond, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    ipcImpl_.ipcAMOCas(reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset),
                       cond, value);
  } else {
    auto *qp = getQueuePair(pe);
    qp->atomic_nofetch(base_heap[pe] + L_offset, value, cond, pe, true,
                       MLX5_OPCODE_ATOMIC_CS);
  }
}

template <typename T>
__device__ void GPUIBContext::internal_put_broadcast(
    T *dst, const T *src, int nelems, int pe_root, int pe_start,
    int log_pe_stride, int pe_size,
    [[maybe_unused]] long *p_sync) {  // NOLINT(runtime/int)
  if (my_pe == pe_root) {
    int stride = 1 << log_pe_stride;
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
      if (i != my_pe) {
        put_nbi_wg(dst, src, nelems, i);
      }
    }
  }
}

template <typename T>
__device__ void GPUIBContext::internal_get_broadcast(
    T *dst, const T *src, int nelems, int pe_root,
    [[maybe_unused]] long *pSync) {  // NOLINT(runtime/int)
  if (my_pe != pe_root) {
    get_wg(dst, src, nelems, pe_root);
  }
}

template <typename T>
__device__ void GPUIBContext::broadcast(rocshmem_team_t team, T *dst,
                                        const T *src, int nelems, int pe_root) {
  GPUIBTeam *team_obj = reinterpret_cast<GPUIBTeam *>(team);

  double dbl_log_pe_stride = team_obj->tinfo_wrt_world->log_stride;
  int log_pe_stride = static_cast<int>(dbl_log_pe_stride);
  /**
   * Ensure that the stride is a multiple of 2 for GPU_IB.
   * TODO(bpotter): enable GPU_IB to work with non-powers-of-2 strides
   * and remove this assert.
   */
  assert((dbl_log_pe_stride - log_pe_stride) == 0);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;

  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(pe_root);

  broadcast<T>(dst, src, nelems, pe_root_world, pe_start, log_pe_stride,
               pe_size, p_sync);
}

template <typename T>
__device__ void GPUIBContext::broadcast(T *dst, const T *src, int nelems,
                                        int pe_root, int pe_start,
                                        int log_pe_stride, int pe_size,
                                        long *p_sync) {  // NOLINT(runtime/int)
  if (num_pes < 4) {
    internal_put_broadcast(dst, src, nelems, pe_root, pe_start, log_pe_stride,
                           pe_size, p_sync);
  } else {
    internal_get_broadcast(dst, src, nelems, pe_root, p_sync);
  }
  // Synchronize on completion of broadcast
  internal_sync(my_pe, pe_start, (1 << log_pe_stride), pe_size, p_sync);
}

template <typename T>
__device__ void GPUIBContext::alltoall(rocshmem_team_t team, T *dst,
                                       const T *src, int nelems) {
  // Currently broadcast implementation performs the best
  alltoall_broadcast(team, dst, src, nelems);
}

template <typename T>
__device__ void GPUIBContext::alltoall_broadcast(rocshmem_team_t team, T *dst,
                                                 const T *src, int nelems) {
  // Broadcast implementation of alltoall collective
  GPUIBTeam *team_obj = reinterpret_cast<GPUIBTeam *>(team);

  double dbl_log_pe_stride = team_obj->tinfo_wrt_world->log_stride;
  int log_pe_stride = static_cast<int>(dbl_log_pe_stride);
  /**
   * Ensure that the stride is a multiple of 2 for GPU_IB.
   * TODO(bpotter): enable GPU_IB to work with non-powers-of-2 strides
   * and remove this assert.
   */
  assert((dbl_log_pe_stride - log_pe_stride) == 0);
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = 1 << log_pe_stride;

  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  // Have each PE put their designated data to the other PEs
  for (int j = 0; j < pe_size; j++) {
    int dest_pe = team_obj->get_pe_in_world(j);
    put_nbi_wg(&dst[my_pe_in_team * nelems], &src[j * nelems], nelems, dest_pe);
  }
  if (is_thread_zero_in_block()) {
    quiet();
  }
  // wait until everyone has obtained their designated data
  internal_sync(my_pe, pe_start, stride, pe_size, pSync);
}

template <typename T>
__device__ void GPUIBContext::alltoall_brucks(rocshmem_team_t team, T *dst,
                                              const T *src, int nelems) {
  // Brucks implementation of alltoall collective
  GPUIBTeam *team_obj = reinterpret_cast<GPUIBTeam *>(team);

  double dbl_log_pe_stride = team_obj->tinfo_wrt_world->log_stride;
  int log_pe_stride = static_cast<int>(dbl_log_pe_stride);
  /**
   * Ensure that the stride is a multiple of 2 for GPU_IB.
   * TODO(bpotter): enable GPU_IB to work with non-powers-of-2 strides
   * and remove this assert.
   */
  assert((dbl_log_pe_stride - log_pe_stride) == 0);
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = 1 << log_pe_stride;

  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  int tid = get_flat_block_id();
  int blk_size = get_flat_block_size();

  // Check if we have enough buffer space. If not, fail.
  if (pe_size * nelems * 2 > ROCSHMEM_ATA_MAX_WRKDATA_SIZE) {
    GPU_DPRINTF("Unsupported alltoall size for gpu_ib.\n");
    assert(false);
  }

  T *pAta1 = reinterpret_cast<T *>(team_obj->pAta);
  T *pAta2 = &pAta1[pe_size * nelems];

  // Phase 1: Shift all data by (pe_size * nelems) elements
  for (size_t i = tid; i < pe_size * nelems; i += blk_size) {
    size_t index = (i + my_pe_in_team * nelems) % (pe_size * nelems);
    pAta1[i] = src[index];
  }
  __syncthreads();

  // Phase 2: Perform packing and data transfers
  for (int64_t shift = 0; ((int64_t)1 << shift) < pe_size; shift++) {
    int64_t shift_decimal = ((int64_t)1 << shift);
    // Step 1: Pack data to be sent
    for (int64_t i = tid; i < pe_size * nelems; i += blk_size) {
      int64_t pos = i / nelems;
      int64_t offset = i % nelems;
      // If bit is set in index, insert in data to be sent
      if ((pos >> shift) & 1) {
        int64_t index =
            ((pos >> (shift + 1)) << shift) + (pos & (shift_decimal - 1));
        pAta2[index * nelems + offset] = pAta1[i];
      }
    }
    threadfence_system();
    __syncthreads();

    // Calculate how much data to be sent
    int64_t region_size = shift_decimal * 2;
    int64_t data_size = nelems * (pe_size / region_size * shift_decimal);
    if (pe_size % region_size > shift_decimal)
      data_size += pe_size % region_size - shift_decimal;

    // Step 2: Send data
    int dest_pe =
        team_obj->get_pe_in_world((my_pe_in_team + shift_decimal) % pe_size);
    put_wg(dst, pAta2, data_size, dest_pe);
    if (is_thread_zero_in_block()) {
      quiet();
    }
    threadfence_system();
    // Need to synchronize with both receiver and sender. So just sync all.
    internal_sync(my_pe, pe_start, stride, pe_size, pSync);
    // Step 3: Unpack received data
    for (int i = tid; i < pe_size * nelems; i += blk_size) {
      int64_t pos = i / nelems;
      int64_t offset = i % nelems;
      // If bit is set in index, insert in data to be sent
      if ((pos >> shift) & 1) {
        int64_t index =
            ((pos >> (shift + 1)) << shift) + (pos & (shift_decimal - 1));
        pAta1[i] = dst[index * nelems + offset];
      }
    }
    threadfence_system();
    __syncthreads();
  }

  // Phase 3: Inverse rotation, shift data by (pe_size * nelems) elements
  for (size_t i = tid; i < pe_size * nelems; i += blk_size) {
    size_t offset = i % nelems;
    size_t index = ((pe_size + my_pe_in_team - i / nelems) % pe_size) * nelems;
    dst[index + offset] = pAta1[i];
  }

  // wait until everyone has sent the data
  internal_sync(my_pe, pe_start, stride, pe_size, pSync);
}

template <typename T>
__device__ void GPUIBContext::alltoall_gcen(rocshmem_team_t team, T *dst,
                                            const T *src, int nelems) {
  // GPU-centric implementation of alltoall collective
  GPUIBTeam *team_obj = reinterpret_cast<GPUIBTeam *>(team);

  double dbl_log_pe_stride = team_obj->tinfo_wrt_world->log_stride;
  int log_pe_stride = static_cast<int>(dbl_log_pe_stride);
  /**
   * Ensure that the stride is a multiple of 2 for GPU_IB.
   * TODO(bpotter): enable GPU_IB to work with non-powers-of-2 strides
   * and remove this assert.
   */
  assert((dbl_log_pe_stride - log_pe_stride) == 0);
  int pe_size = team_obj->num_pes;
  int stride = 1 << log_pe_stride;

  long *pSync = team_obj->alltoall_pSync;
  int64_t *pSync2 = &team_obj->alltoall_pSync[ROCSHMEM_BARRIER_SYNC_SIZE];
  int my_pe_in_team = team_obj->my_pe;

  // Check if we have enough buffer space. If not, fail.
  T *pAta = reinterpret_cast<T *>(team_obj->pAta);
  if (pe_size * nelems > ROCSHMEM_ATA_MAX_WRKDATA_SIZE) {
    GPU_DPRINTF("Unsupported alltoall size for gpu_ib.\n");
    assert(false);
  }

  // Works when number of PEs divisible by root(PE_size)
  int num_clust = sqrt(pe_size);
  int clust_size = (pe_size + num_clust - 1) / num_clust;
  // TODO(bpotter): Allow any size of cluster
  assert(num_clust * clust_size == pe_size);
  int clust_id = my_pe_in_team / clust_size;

  int64_t flag_val = 1;
  // Step 1: Send data to PEs in cluster
  for (int i = 0; i < pe_size; ++i) {
    int src_pe =
        team_obj->get_pe_in_world(clust_id * clust_size + (i % clust_size));
    int src_loc = (i / clust_size) * clust_size + (my_pe_in_team % clust_size);
    get_nbi_wg(&pAta[i * nelems], &src[src_loc * nelems], nelems, src_pe);
  }
  if (is_thread_zero_in_block()) {
    quiet();
  }
  __syncthreads();
  // Step 2: Send final data to PEs outside cluster
  for (int i = 0; i < num_clust; i++) {
    int dest_pe = team_obj->get_pe_in_world((my_pe_in_team % clust_size) +
                                            i * clust_size);
    int j = clust_id;
    put_nbi_wg(&dst[j * nelems * clust_size], &pAta[i * nelems * clust_size],
               nelems * clust_size, dest_pe);
  }
  if (is_thread_zero_in_block()) {
    quiet();

    // Now sync PEs in cluster and ring. Ideally, we overlap this.
    int dest_pe = team_obj->get_pe_in_world(clust_id * clust_size);
    if (dest_pe != my_pe) amo_add<int64_t>(pSync2, flag_val, dest_pe);

    int dest_pe2 = team_obj->get_pe_in_world(my_pe_in_team % clust_size);
    if (dest_pe2 != my_pe) amo_add<int64_t>(&pSync[0], flag_val, dest_pe2);

    if (my_pe == dest_pe) {
      wait_until(pSync2, ROCSHMEM_CMP_EQ, flag_val * (clust_size - 1));
      pSync2[0] = ROCSHMEM_SYNC_VALUE;
      __threadfence_system();
      for (int i = 1; i < clust_size; ++i)
        put_nbi(&pSync2[0], &flag_val, 1,
                team_obj->get_pe_in_world(my_pe_in_team + i));
    } else {
      wait_until(pSync2, ROCSHMEM_CMP_EQ, flag_val);
      pSync2[0] = ROCSHMEM_SYNC_VALUE;
      __threadfence_system();
    }

    if (my_pe == dest_pe2) {
      wait_until(&pSync[0], ROCSHMEM_CMP_EQ, (int64_t)(num_clust - 1));
      pSync[0] = ROCSHMEM_SYNC_VALUE;
      threadfence_system();
      for (size_t i = 1, j = dest_pe2 + clust_size * stride; i < num_clust;
           ++i, j += clust_size * stride) {
        put_nbi(&pSync[0], &flag_val, 1, j);
      }
    } else {
      wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
      pSync[0] = ROCSHMEM_SYNC_VALUE;
      threadfence_system();
    }
  }
  __syncthreads();
}

template <typename T>
__device__ void GPUIBContext::alltoall_gcen2(rocshmem_team_t team, T *dst,
                                             const T *src, int nelems) {
  // GPU-centric implementation of alltoall collective
  // Uses in-place blocking sync
  GPUIBTeam *team_obj = reinterpret_cast<GPUIBTeam *>(team);

  double dbl_log_pe_stride = team_obj->tinfo_wrt_world->log_stride;
  int log_pe_stride = static_cast<int>(dbl_log_pe_stride);
  /**
   * Ensure that the stride is a multiple of 2 for GPU_IB.
   * TODO(bpotter): enable GPU_IB to work with non-powers-of-2 strides
   * and remove this assert.
   */
  assert((dbl_log_pe_stride - log_pe_stride) == 0);
  int pe_size = team_obj->num_pes;
  int stride = 1 << log_pe_stride;

  long *pSync = team_obj->alltoall_pSync;
  int64_t *pSync2 = &team_obj->alltoall_pSync[ROCSHMEM_BARRIER_SYNC_SIZE];
  int my_pe_in_team = team_obj->my_pe;

  // Check if we have enough buffer space. If not, fail.
  T *pAta = reinterpret_cast<T *>(team_obj->pAta);
  if (pe_size * nelems > ROCSHMEM_ATA_MAX_WRKDATA_SIZE) {
    GPU_DPRINTF("Unsupported alltoall size for gpu_ib.\n");
    assert(false);
  }

  // Works when number of PEs divisible by root(PE_size)
  int num_clust = sqrt(pe_size);
  int clust_size = (pe_size + num_clust - 1) / num_clust;
  // TODO(bpotter): Allow any size of cluster
  assert(num_clust * clust_size == pe_size);
  int clust_id = my_pe_in_team / clust_size;

  int64_t flag_val = 1;
  // Step 1: Send data to PEs in cluster
  for (int i = 0; i < pe_size; ++i) {
    int src_pe =
        team_obj->get_pe_in_world(clust_id * clust_size + (i % clust_size));
    int src_loc = (i / clust_size) * clust_size + (my_pe_in_team % clust_size);
    get_nbi_wg(&pAta[i * nelems], &src[src_loc * nelems], nelems, src_pe);
  }

  if (is_thread_zero_in_block()) {
    int dest_pe = team_obj->get_pe_in_world(clust_id * clust_size);
    if (dest_pe != my_pe) amo_add<int64_t>(pSync2, flag_val, dest_pe);
    quiet();
  }
  __syncthreads();

  // Step 2: Send final data to PEs outside cluster
  // Have each PE put their designated data to the other PEs
  for (int i = 0; i < num_clust; i++) {
    int dest_pe = team_obj->get_pe_in_world((my_pe_in_team % clust_size) +
                                            i * clust_size);
    int j = clust_id;
    put_nbi_wg(&dst[j * nelems * clust_size], &pAta[i * nelems * clust_size],
               nelems * clust_size, dest_pe);
  }

  if (is_thread_zero_in_block()) {
    quiet();
    if ((my_pe_in_team % clust_size) == 0) {
      wait_until(pSync2, ROCSHMEM_CMP_EQ, flag_val * (clust_size - 1));
      pSync2[0] = ROCSHMEM_SYNC_VALUE;
      __threadfence_system();
      for (int i = 1; i < clust_size; ++i)
        put_nbi(&pSync2[0], &flag_val, 1,
                team_obj->get_pe_in_world(my_pe_in_team + i));
    } else {
      wait_until(pSync2, ROCSHMEM_CMP_EQ, flag_val);
      pSync2[0] = ROCSHMEM_SYNC_VALUE;
      __threadfence_system();
    }
  }

  // wait until everyone in ring has sent the data
  internal_sync(my_pe, team_obj->get_pe_in_world(my_pe_in_team % clust_size),
                clust_size * stride, num_clust, pSync);
}

template <typename T>
__device__ void GPUIBContext::fcollect(rocshmem_team_t team, T *dst,
                                       const T *src, int nelems) {
  // Main function for fcollect
  // Broadcast version performs moderately well
  // But there still seems to be scope for optimisation
  fcollect_broadcast(team, dst, src, nelems);
}

template <typename T>
__device__ void GPUIBContext::fcollect_broadcast(rocshmem_team_t team, T *dst,
                                                 const T *src, int nelems) {
  // Broadcast implementation of fcollect collective
  GPUIBTeam *team_obj = reinterpret_cast<GPUIBTeam *>(team);

  double dbl_log_pe_stride = team_obj->tinfo_wrt_world->log_stride;
  int log_pe_stride = static_cast<int>(dbl_log_pe_stride);
  /**
   * Ensure that the stride is a multiple of 2 for GPU_IB.
   * TODO(bpotter): enable GPU_IB to work with non-powers-of-2 strides
   * and remove this assert.
   */
  assert((dbl_log_pe_stride - log_pe_stride) == 0);
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = 1 << log_pe_stride;

  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  // Have each PE put their designated data to the other PEs
  for (int j = 0; j < pe_size; j++) {
    int dest_pe = team_obj->get_pe_in_world(j);
    put_nbi_wg(&dst[my_pe_in_team * nelems], src, nelems, dest_pe);
  }

  if (is_thread_zero_in_block()) {
    quiet();
  }
  // wait until everyone has obtained their designated data
  internal_sync(my_pe, pe_start, stride, pe_size, pSync);
}

template <typename T>
__device__ void GPUIBContext::fcollect_brucks(rocshmem_team_t team, T *dst,
                                              const T *src, int nelems) {
  // Brucks implementation of fcollect collective
  GPUIBTeam *team_obj = reinterpret_cast<GPUIBTeam *>(team);

  double dbl_log_pe_stride = team_obj->tinfo_wrt_world->log_stride;
  int log_pe_stride = static_cast<int>(dbl_log_pe_stride);
  /**
   * Ensure that the stride is a multiple of 2 for GPU_IB.
   * TODO(bpotter): enable GPU_IB to work with non-powers-of-2 strides
   * and remove this assert.
   */
  assert((dbl_log_pe_stride - log_pe_stride) == 0);
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = 1 << log_pe_stride;

  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  int tid = get_flat_block_id();
  int blk_size = get_flat_block_size();

  // Check if we have enough buffer space. If not, fail.
  if (pe_size * nelems > ROCSHMEM_ATA_MAX_WRKDATA_SIZE) {
    GPU_DPRINTF("Unsupported fcollect size for gpu_ib.\n");
    assert(false);
  }

  T *pAta = reinterpret_cast<T *>(team_obj->pAta);

  // Initial src transfer
  put_wg(pAta, src, nelems, team_obj->get_pe_in_world(my_pe_in_team));

  // Phase 1: Perform data transfers
  for (int64_t shift = 0; ((int64_t)1 << shift) < pe_size; shift++) {
    int64_t shift_decimal = ((int64_t)1 << shift);

    // Calculate how much data to be sent
    int64_t data_size =
        min(shift_decimal, pe_size - shift_decimal) * nelems;  // NOLINT

    // Send data
    int dest_pe =
        team_obj->get_pe_in_world((my_pe_in_team + shift_decimal) % pe_size);
    put_wg(&pAta[shift_decimal * nelems], pAta, data_size, dest_pe);

    // Need to synchronize with both receiver and sender. So just sync all.
    internal_sync(my_pe, pe_start, stride, pe_size, pSync);
  }

  // Phase 2: Inverse rotation, shift data by (pe_size * nelems) elements
  for (size_t i = tid; i < pe_size * nelems; i += blk_size) {
    size_t offset = i % nelems;
    size_t index =
        ((pe_size + my_pe_in_team - i / nelems) % (pe_size)) * nelems;
    dst[index + offset] = pAta[i];
  }

  // wait until everyone has sent the data
  internal_sync(my_pe, pe_start, stride, pe_size, pSync);
}

template <typename T>
__device__ void GPUIBContext::fcollect_gcen(rocshmem_team_t team, T *dst,
                                            const T *src, int nelems) {
  // GPU-centric implementation of fcollect collective
  GPUIBTeam *team_obj = reinterpret_cast<GPUIBTeam *>(team);

  double dbl_log_pe_stride = team_obj->tinfo_wrt_world->log_stride;
  int log_pe_stride = static_cast<int>(dbl_log_pe_stride);
  /**
   * Ensure that the stride is a multiple of 2 for GPU_IB.
   * TODO(bpotter): enable GPU_IB to work with non-powers-of-2 strides
   * and remove this assert.
   */
  assert((dbl_log_pe_stride - log_pe_stride) == 0);
  int pe_size = team_obj->num_pes;
  int stride = 1 << log_pe_stride;

  long *pSync = team_obj->alltoall_pSync;
  long *pSync2 = &team_obj->alltoall_pSync[ROCSHMEM_BARRIER_SYNC_SIZE];
  int my_pe_in_team = team_obj->my_pe;

  // Check if we have enough buffer space. If not, fail.
  T *pAta = reinterpret_cast<T *>(team_obj->pAta);
  if (pe_size * nelems > ROCSHMEM_ATA_MAX_WRKDATA_SIZE) {
    GPU_DPRINTF("Unsupported fcollect size for gpu_ib.\n");
    assert(false);
  }

  // Works when number of PEs divisible by root(PE_size)
  int num_clust = sqrt(pe_size);
  int clust_size = (pe_size + num_clust - 1) / num_clust;
  // TODO(bpotter): Allow any size of cluster
  assert(num_clust * clust_size == pe_size);
  int clust_id = my_pe_in_team / clust_size;

  int64_t flag_val = 1;
  // Step 1: Send data to PEs in cluster
  for (int i = 0; i < clust_size; ++i) {
    int src_pe =
        team_obj->get_pe_in_world(clust_id * clust_size + (i % clust_size));
    get_nbi_wg(&pAta[i * nelems], src, nelems, src_pe);
  }

  if (is_thread_zero_in_block()) {
    int dest_pe = team_obj->get_pe_in_world(clust_id * clust_size);
    if (dest_pe != my_pe) amo_add<int64_t>(pSync2, flag_val, dest_pe);
    quiet();
  }
  __syncthreads();

  // Step 2: Send final data to PEs outside cluster
  // Have each PE put their designated data to the other PEs
  for (int i = 0; i < num_clust; i++) {
    int dest_pe = team_obj->get_pe_in_world((my_pe_in_team % clust_size) +
                                            i * clust_size);
    int j = clust_id;
    put_nbi_wg(&dst[j * nelems * clust_size], pAta, nelems * clust_size,
               dest_pe);
  }

  if (is_thread_zero_in_block()) {
    quiet();
    if ((my_pe_in_team % clust_size) == 0) {
      wait_until(pSync2, ROCSHMEM_CMP_EQ, flag_val * (clust_size - 1));
      pSync2[0] = ROCSHMEM_SYNC_VALUE;
      threadfence_system();
      for (int i = 1; i < clust_size; ++i)
        put_nbi(&pSync2[0], &flag_val, 1,
                team_obj->get_pe_in_world(my_pe_in_team + i));
    } else {
      wait_until(pSync2, ROCSHMEM_CMP_EQ, flag_val);
      pSync2[0] = ROCSHMEM_SYNC_VALUE;
      threadfence_system();
    }
  }

  // wait until everyone in ring has sent the data
  internal_sync(my_pe, team_obj->get_pe_in_world(my_pe_in_team % clust_size),
                clust_size * stride, num_clust, pSync);
}

template <typename T>
__device__ void GPUIBContext::fcollect_gcen2(rocshmem_team_t team, T *dst,
                                             const T *src, int nelems) {
  // GPU-centric implementation of fcollect collective
  // Uses in-place blocking sync
  GPUIBTeam *team_obj = reinterpret_cast<GPUIBTeam *>(team);

  double dbl_log_pe_stride = team_obj->tinfo_wrt_world->log_stride;
  int log_pe_stride = static_cast<int>(dbl_log_pe_stride);
  /**
   * Ensure that the stride is a multiple of 2 for GPU_IB.
   * TODO(bpotter): enable GPU_IB to work with non-powers-of-2 strides
   * and remove this assert.
   */
  assert((dbl_log_pe_stride - log_pe_stride) == 0);
  int pe_size = team_obj->num_pes;
  int stride = 1 << log_pe_stride;

  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Check if we have enough buffer space. If not, fail.
  T *pAta = reinterpret_cast<T *>(team_obj->pAta);
  if (pe_size * nelems > ROCSHMEM_ATA_MAX_WRKDATA_SIZE) {
    GPU_DPRINTF("Unsupported fcollect size for gpu_ib.\n");
    assert(false);
  }

  // Works when number of PEs divisible by root(PE_size)
  int num_clust = sqrt(pe_size);
  int clust_size = (pe_size + num_clust - 1) / num_clust;
  // TODO(bpotter): Allow any size of cluster
  assert(num_clust * clust_size == pe_size);
  int clust_id = my_pe_in_team / clust_size;

  // Step 1: Send data to PEs in cluster
  for (int i = 0; i < clust_size; ++i) {
    int src_pe =
        team_obj->get_pe_in_world(clust_id * clust_size + (i % clust_size));
    get_nbi_wg(&pAta[i * nelems], src, nelems, src_pe);
  }

  if (is_thread_zero_in_block()) {
    quiet();
  }
  internal_sync(my_pe, team_obj->get_pe_in_world(clust_id * clust_size), stride,
                clust_size, pSync);

  // Step 2: Send final data to PEs outside cluster
  // Have each PE put their designated data to the other PEs
  for (int i = 0; i < num_clust; i++) {
    int dest_pe = team_obj->get_pe_in_world((my_pe_in_team % clust_size) +
                                            i * clust_size);
    int j = clust_id;
    put_nbi_wg(&dst[j * nelems * clust_size], pAta, nelems * clust_size,
               dest_pe);
  }

  if (is_thread_zero_in_block()) quiet();

  // wait until everyone in ring has sent the data
  internal_sync(my_pe, team_obj->get_pe_in_world(my_pe_in_team % clust_size),
                clust_size * stride, num_clust, pSync);
}

/******************************************************************************
 ***************** SHMEM X API EXTENSION FOR BLOCK/WAVE LEVEL *****************
 *****************************************************************************/

template <typename T>
__device__ void GPUIBContext::put_wg(T *dest, const T *source, size_t nelems,
                                     int pe) {
  putmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GPUIBContext::put_wave(T *dest, const T *source, size_t nelems,
                                       int pe) {
  putmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GPUIBContext::put_nbi_wg(T *dest, const T *source,
                                         size_t nelems, int pe) {
  putmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GPUIBContext::put_nbi_wave(T *dest, const T *source,
                                           size_t nelems, int pe) {
  putmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GPUIBContext::get_wg(T *dest, const T *source, size_t nelems,
                                     int pe) {
  getmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GPUIBContext::get_wave(T *dest, const T *source, size_t nelems,
                                       int pe) {
  getmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GPUIBContext::get_nbi_wg(T *dest, const T *source,
                                         size_t nelems, int pe) {
  getmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GPUIBContext::get_nbi_wave(T *dest, const T *source,
                                           size_t nelems, int pe) {
  getmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_CONTEXT_IB_TMPL_DEVICE_HPP_

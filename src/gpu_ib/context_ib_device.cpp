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

#include "context_ib_device.hpp"

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

#include "context_incl.hpp"
#include "gda_device.hpp"
#include "queue_pair.hpp"

namespace rocshmem {

GPUIBContext::GPUIBContext(GDADevice *device, int idx)
    : Context(device) {
  base_heap = device->heap.get_heap_bases().data();
  barrier_sync = device->barrier_sync;
  device->initialize_context(this, idx);
  size_t barrier_sync_offset = idx * ROCSHMEM_BARRIER_SYNC_SIZE;
  barrier_sync = device->barrier_sync + barrier_sync_offset;
}

__device__ void GPUIBContext::quiet() {
  for (int k = 0; k < num_pes; k++) {
    qps[k].quiet();
  }
}

__device__ void *GPUIBContext::shmem_ptr(const void *dest, int pe) {
  return nullptr;
}

__device__ void GPUIBContext::putmem(void *dest, const void *source, size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[pe].put_nbi(base_heap[pe] + L_offset, source, nelems, pe);
      qps[pe].quiet();
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

__device__ void GPUIBContext::putmem_nbi(void *dest, const void *source, size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[pe].put_nbi(base_heap[pe] + L_offset, source, nelems, pe);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

static __device__ __forceinline__ void store_asm(uint8_t* val, uint8_t* dst, int size) {
  switch (size) {
    case 2: {
      int16_t val16{*(reinterpret_cast<int16_t*>(val))};
      asm volatile("flat_store_short %0 %1 sc0 sc1" : : "v"(dst), "v"(val16));
      break;
    }
    case 4: {
      int32_t val32{*(reinterpret_cast<int32_t*>(val))};
      asm volatile("flat_store_dword %0 %1 sc0 sc1" : : "v"(dst), "v"(val32));
      break;
    }
    case 8: {
      int64_t val64{*(reinterpret_cast<int64_t*>(val))};
      asm volatile("flat_store_dwordx2 %0 %1 sc0 sc1" : : "v"(dst), "v"(val64));
      break;
    }
    default:
      break;
  }
}

static __device__ __forceinline__ void memcpy_wave(void* dst, void* src, size_t size) {
  int wave_tid = get_flat_block_id() % 64;
  int wave_size{wave_SZ()};

  int cpy_size{};
  uint8_t* dst_bytes{nullptr};
  uint8_t* dst_def{nullptr};
  uint8_t* src_bytes{nullptr};
  uint8_t* src_def{nullptr};

  dst_def = reinterpret_cast<uint8_t*>(dst);
  src_def = reinterpret_cast<uint8_t*>(src);
  dst_bytes = dst_def;
  src_bytes = src_def;

  for (int j{8}; j > 1; j >>= 1) {
    cpy_size = size / j;
    for (int i{wave_tid}; i < cpy_size; i += wave_size) {
      dst_bytes = dst_def;
      src_bytes = src_def;

      src_bytes += i * j;
      dst_bytes += i * j;

      store_asm(src_bytes, dst_bytes, j);
    }
    size -= cpy_size * j;
    dst_def += cpy_size * j;
    src_def += cpy_size * j;
  }

  if (size == 1) {
    if (is_thread_zero_in_wave()) {
      *dst_bytes = *src_bytes;
    }
  }
}

__device__ void GPUIBContext::putmem_wave(void *dest, const void *source, size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  if (pe == my_pe) {
    memcpy_wave(dest, (void*)source, nelems);
  } else if (is_thread_zero_in_wave()) {
    qps[pe].put_nbi(base_heap[pe] + L_offset, source, nelems, pe);
    qps[pe].quiet();
  }
}

__device__ void GPUIBContext::putmem_nbi_wave(void *dest, const void *source, size_t nelems, int pe) {
  uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
  if (pe == my_pe) {
    memcpy_wave(dest, (void*)source, nelems);
  } else if (is_thread_zero_in_wave()) {
    qps[pe].put_nbi(base_heap[pe] + L_offset, source, nelems, pe);
  }
}

__device__ void GPUIBContext::fence() {
  for (int i{0}; i < num_pes; i++) {
    qps[i].quiet();
  }
  __threadfence_system();
}

}  // namespace rocshmem

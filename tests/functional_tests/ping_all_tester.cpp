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

#include "ping_all_tester.hpp"

#include <roc_shmem/roc_shmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void PingAllTest(int loop, int skip, uint64_t *timer, int *r_buf,
                            ShmemContextType ctx_type) {
  __shared__ roc_shmem_ctx_t ctx;

  roc_shmem_wg_init();
  roc_shmem_wg_ctx_create(ctx_type, &ctx);

  int pe = roc_shmem_ctx_my_pe(ctx);
  int num_pe = roc_shmem_ctx_n_pes(ctx);
  int status[1024];
  for (int j{0}; j < num_pe; j++) {
    status[j] = 0;
  }

  if (hipThreadIdx_x == 0) {
    uint64_t start;
    auto blk_pe_off {hipBlockIdx_x * num_pe};

    for (int i = 0; i < loop + skip; i++) {
      if (i == skip) {
        start = roc_shmem_timer();
      }

      for (int j{0}; j < num_pe; j++) {
        roc_shmem_ctx_int_p(ctx, &r_buf[blk_pe_off + pe], 1, j);
      }
      roc_shmem_int_wait_until_all(&r_buf[blk_pe_off], num_pe, status, ROC_SHMEM_CMP_EQ, 1);
    }
    timer[hipBlockIdx_x] = roc_shmem_timer() - start;
  }
  roc_shmem_wg_ctx_destroy(&ctx);
  roc_shmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
PingAllTester::PingAllTester(TesterArguments args) : Tester(args) {
  int num_pes {roc_shmem_n_pes()};
  r_buf = (int *)roc_shmem_malloc(sizeof(int) * args.wg_size * num_pes);
}

PingAllTester::~PingAllTester() { roc_shmem_free(r_buf); }

void PingAllTester::resetBuffers(uint64_t size) {
  int num_pes {roc_shmem_n_pes()};
  memset(r_buf, 0, sizeof(int) * args.wg_size * num_pes);
}

void PingAllTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                  uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(PingAllTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, timer, r_buf, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop;
}

void PingAllTester::verifyResults(uint64_t size) {}
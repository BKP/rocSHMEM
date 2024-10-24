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

#ifndef LIBRARY_SRC_ROC_SHMEM_CALC_HPP_
#define LIBRARY_SRC_ROC_SHMEM_CALC_HPP_

namespace rocshmem {

// clang-format off
NOWARN(-Wunused-parameter,
template <ROC_SHMEM_OP Op>
struct OpWrap {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    static_assert(true, "Unimplemented ipc collective.");
  }
};
)
// clang-format on

/******************************************************************************
 ************************** TEMPLATE SPECIALIZATIONS **************************
 *****************************************************************************/
template <>
struct OpWrap<ROC_SHMEM_SUM> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] += src[i];
  }
};

template <>
struct OpWrap<ROC_SHMEM_MAX> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] = max(dst[i], src[i]);
  }
};

template <>
struct OpWrap<ROC_SHMEM_MIN> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] = min(dst[i], src[i]);
  }
};

template <>
struct OpWrap<ROC_SHMEM_PROD> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] *= src[i];
  }
};

template <>
struct OpWrap<ROC_SHMEM_AND> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] &= src[i];
  }
};

template <>
struct OpWrap<ROC_SHMEM_OR> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] |= src[i];
  }
};

template <>
struct OpWrap<ROC_SHMEM_XOR> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] ^= src[i];
  }
};

}
#endif // LIBRARY_SRC_ROC_SHMEM_CALC_HPP_

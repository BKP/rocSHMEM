/*
hipcc -c -fgpu-rdc -x hip rocshmem_put_signal_test.cc \
  -I/opt/rocm/include \
  -I$ROCSHMEM_SRC_DIR/include \
  -I$ROCSHMEM_INSTALL_DIR/include \
  -I$OPENMPI_UCX_INSTALL_DIR/include/

hipcc -fgpu-rdc --hip-link rocshmem_put_signal_test.o -o rocshmem_getmem_test \
  $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a \
  $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so \
  -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

ROC_SHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 ./rocshmem_put_signal_test
*/

#include <iostream>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <roc_shmem/roc_shmem.hpp>

#define CHECK_HIP(condition) {                                            \
        hipError_t error = condition;                                     \
        if(error != hipSuccess){                                          \
            fprintf(stderr,"HIP error: %d line: %d\n", error,  __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, error);                             \
        }                                                                 \
    }

using namespace rocshmem;

__global__ void simple_put_signal_test(uint64_t *data, uint64_t *message, size_t nelem,
                                       uint64_t *sig_addr, int my_pe, int dst_pe)
{
    roc_shmem_wg_init();

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId == 0) {
        if (my_pe == 0) {
            roc_shmem_ulong_put_signal(data, message, nelem, sig_addr, 1, ROC_SHMEM_SIGNAL_SET, dst_pe);
        }
        else {
            roc_shmem_ulong_wait_until(sig_addr, ROC_SHMEM_CMP_EQ, 1);
            roc_shmem_ulong_put_signal(data, data, nelem, sig_addr, 1, ROC_SHMEM_SIGNAL_SET, dst_pe);
        }
    }

    __syncthreads();
    roc_shmem_wg_finalize();
}

#define MAX_ELEM 256

int main (int argc, char **argv)
{
    int rank = roc_shmem_my_pe();
    int ndevices, my_device = 0;
    CHECK_HIP(hipGetDeviceCount(&ndevices));
    my_device = rank % ndevices;
    CHECK_HIP(hipSetDevice(my_device));
    int nelem = MAX_ELEM;

    if (argc > 1) {
        nelem = atoi(argv[1]);
    }

    roc_shmem_init();
    int npes =  roc_shmem_n_pes();
    int dst_pe = (rank + 1) % npes;
    uint64_t *message = (uint64_t*)roc_shmem_malloc(nelem * sizeof(uint64_t));
    uint64_t *data = (uint64_t*)roc_shmem_malloc(nelem * sizeof(uint64_t));
    uint64_t *sig_addr = (uint64_t*)roc_shmem_malloc(sizeof(uint64_t));
    if (NULL == data || NULL == message || NULL == sig_addr) {
        std::cout << "Error allocating memory from symmetric heap" << std::endl;
        std::cout << "data: " << data
                  << ", message: " << message
                  << ", size: " << sizeof(uint64_t) * nelem
                  << ", sig_addr: " << sig_addr
                  << std::endl;
        roc_shmem_global_exit(1);
    }

    for (int i=0; i<nelem; i++) {
        message[i] = rank;
    }

    CHECK_HIP(hipMemset(data, 0, (nelem * sizeof(uint64_t))));
    CHECK_HIP(hipDeviceSynchronize());

    int threadsPerBlock=256;
    simple_put_signal_test<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(data, message, nelem, sig_addr, rank, dst_pe);
    roc_shmem_barrier_all();
    CHECK_HIP(hipDeviceSynchronize());

    bool pass = true;
    for (int i=0; i<nelem; i++) {
        if (data[i] != 0) {
            pass = false;
#if VERBOSE
            printf("[%d] Error in element %d expected 0 got %d\n", rank, i, dst[i]);
#endif
        }
    }
    printf("[%d] Test %s \t %s\n", rank, argv[0], pass ? "[PASS]" : "[FAIL]");

    roc_shmem_free(data);
    roc_shmem_free(message);
    roc_shmem_finalize();
    return 0;
}

/**
 * @file include/Matrix/MatrixHandler/MatrixTrace/CUDA/src/MatrixTraceCUDAHandler.cu
 *
 * @copyright 2023-2026 Karthik Murali Madhavan Rathai
 */
/*
 * This file is part of CoolDiff library.
 *
 * You can redistribute it and/or modify it under the terms of the GNU
 * General Public License version 3 as published by the Free Software
 * Foundation.
 *
 * Licensees holding a valid commercial license may use this software
 * in accordance with the commercial license agreement provided in
 * conjunction with the software.  The terms and conditions of any such
 * commercial license agreement shall govern, supersede, and render
 * ineffective any application of the GPLv3 license to this software,
 * notwithstanding of any reference thereto in the software or
 * associated repository.
 */

#include <cuda_runtime.h>

#include "MatTraceCUDAHandler.cuh"
#include "GlobalParameters.hpp"

// Matrix trace kernel
template<typename T>
__global__ void MatrixTrace(const T* A, T* B, const size_t N) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(B, A[i * N + i]);
    }
}

// CUDA matrix trace kernel
template<typename T>
void TraceKernel(const dim3 blocks, const dim3 threads, const T* A, T* B, const size_t N) {
    // Launch kernel
    MatrixTrace <<<blocks, threads>>> (A, B, N);
    // Device synchronize
    cudaDeviceSynchronize();
}

// CUDA matrix trace kernel (float)
template void TraceKernel<float>(const dim3, const dim3, const float*, float*, const size_t);
// CUDA matrix trace kernel (double)
template void TraceKernel<double>(const dim3, const dim3, const double*, double*, const size_t);
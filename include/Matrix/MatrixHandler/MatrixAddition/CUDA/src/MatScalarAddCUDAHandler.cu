/**
 * @file include/Matrix/MatrixHandler/MatrixAddition/CUDA/src/MatScalarAddCUDAHandler.cu
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

#include "MatScalarAddCUDAHandler.cuh"

// Scalar-Matrix addition kernel
template<typename T>
__global__ void MatrixScalarAdd(const T* A, T* B, const size_t M, const size_t N, T val) {
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < M) && (col < N)) {
        const auto idx = row * N + col;
        B[idx] = val + A[idx];
    }
}

// CUDA matrix scalar addition kernel
template<typename T>
void AddScalarKernel(const dim3 blocks, const dim3 threads, const T* A, T* B, const size_t M, const size_t N, T val) {
    // Launch kernel
    MatrixScalarAdd <<<blocks, threads>>> (A, B, M, N, val);
    // Device synchronize
    cudaDeviceSynchronize();
}

// CUDA matrix scalar addition kernel (float)
template void AddScalarKernel<float>(const dim3, const dim3, const float*, float*, const size_t, const size_t, float);
// CUDA matrix scalar addition kernel (double)
template void AddScalarKernel<double>(const dim3, const dim3, const double*, double*, const size_t, const size_t, double);
/**
 * @file include/Matrix/MatrixHandler/MatrixKronProduct/CUDA/src/MatKronCUDAHandler.cu
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


#include "MatKronCUDAHandler.cuh"
#include "GlobalParameters.hpp"

// Matrix-Matrix Hadamard product kernel
template<typename T>
__global__ void MatrixKronecker(const T* A, const T* B, T* C, const size_t m, const size_t n, const size_t k, const size_t l) {
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;

    const auto M = m * k;
    const auto N = n * l;

    if ((row < M) && (col < N)) {
        const auto i = row / k; const auto p = row % k;
        const auto j = col / l; const auto q = col % l;

        const T a = A[i * n + j]; 
        const T b = B[p * l + q];

        C[row * N + col] = a * b;
    }
}

// CUDA matrix Kronecker kernel
template<typename T>
void KronKernel(const dim3 blocks, const dim3 threads, const T* A, const T* B, T* C, const size_t M, const size_t N, const size_t K, const size_t L) {
    // Launch kernel
    MatrixKronecker <<<blocks, threads>>> (A, B, C, M, N, K, L);
    // Device synchronize
    cudaDeviceSynchronize();
}

// CUDA matrix Kronecker product kernel (float)
template void KronKernel<float>(const dim3, const dim3, const float*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
// CUDA matrix Kronecker product kernel (double)
template void KronKernel<double>(const dim3, const dim3, const double*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
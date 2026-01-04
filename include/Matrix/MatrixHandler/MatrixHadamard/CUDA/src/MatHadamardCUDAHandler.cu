/**
 * @file include/Matrix/MatrixHandler/MatrixHadamard/CUDA/src/MatHadamardCUDAHandler.cu
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


#include "MatHadamardCUDAHandler.cuh"

// Matrix-Matrix Hadamard product kernel
template<typename T>
__global__ void MatrixHadamard(const T* A, const T* B, T* C, const size_t M, const size_t N) {
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < M) && (col < N)) {
        const auto idx = row * N + col;
        C[idx] = A[idx]*B[idx];
    }
}

// CUDA matrix Hadamard kernel
template<typename T>
void HadamardKernel(const dim3 blocks, const dim3 threads, const T* A, const T* B, T* C, const size_t M, const size_t N) {
    // Launch kernel
    MatrixHadamard <<<blocks, threads>>> (A, B, C, M, N);
    // Device synchronize
    cudaDeviceSynchronize();
}

// CUDA matrix Hadamard kernel (float)
template void HadamardKernel<float>(const dim3, const dim3, const float*, const float*, float*, const size_t, const size_t);
// CUDA matrix Hadamard kernel (double)
template void HadamardKernel<double>(const dim3, const dim3, const double*, const double*, double*, const size_t, const size_t);
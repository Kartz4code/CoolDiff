/**
 * @file include/Matrix/MatrixHandler/MatrixTranspose/CUDA/src/MatTransposeCUDAHandler.cu
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

#include "MatTransposeCUDAHandler.cuh"

// Matrix transpose kernel
template<typename T>
__global__ void MatrixTranspose(const T* input, T* output, const size_t M, const size_t N) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    auto x = blockIdx.x * TILE_DIM + threadIdx.x;
    auto y = blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int i{}; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < N && (y + i) < M) {
            tile[threadIdx.y + i][threadIdx.x] =
                input[(y + i) * N + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int i{}; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < M && (y + i) < N) {
            output[(y + i) * M + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

// Custom unary function kernel
template<typename T>
void TransposeKernel(const dim3 blocks, const dim3 threads, const T* A, T* B, const size_t M, const size_t N) {
    // Launch kernel
    MatrixTranspose<T> <<<blocks, threads>>> (A, B, M, N);
    // Device synchronize
    cudaDeviceSynchronize();
}

// Custom unary function kernel (float)
template void TransposeKernel<float>(const dim3, const dim3, const float*, float*, const size_t, const size_t);
// Custom unary function kernel (double)
template void TransposeKernel<double>(const dim3, const dim3, const double*, double*, const size_t, const size_t);
/**
 * @file include/Matrix/MatrixHandler/MatrixMultiplication/CUDA/src/MatMulCUDAHandler.cu
 *
 * @copyright 2023-2025 Karthik Murali Madhavan Rathai
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


#include "MatMulCUDAHandler.cuh"

template<typename T>
__global__ void MatrixMul(const T* A, const T* B, T* C, const size_t M, const size_t K, const size_t N) {
    // Shared memory tiles
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];

    auto row = blockIdx.y * TILE_SIZE + threadIdx.y;
    auto col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Intermediate value
    T value = 0.0f;

    // Loop over tiles
    for (int t{}; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A tile
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Load B tile
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        }
        else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int i{}; i < TILE_SIZE; ++i) {
            value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if ((row < M) && (col < N)) {
        C[row * N + col] = value;
    }
}

// CUDA matrix multiplication kernel
template<typename T>
void MulKernel(const dim3 blocks, const dim3 threads, const T* A, const T* B, T* C, const size_t M, const size_t K, const size_t N) {
    // Launch kernel
    MatrixMul <<<blocks, threads>>> (A, B, C, M, K, N);
    // Device synchronize
    cudaDeviceSynchronize();
}

// CUDA matrix multiplication kernel (float)
template void MulKernel<float>(const dim3, const dim3, const float*, const float*, float*, const size_t, const size_t, const size_t);
// CUDA matrix multiplication kernel (double)
template void MulKernel<double>(const dim3, const dim3, const double*, const double*, double*, const size_t, const size_t, const size_t);
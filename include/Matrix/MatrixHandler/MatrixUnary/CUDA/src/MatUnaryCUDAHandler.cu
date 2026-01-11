/**
 * @file include/Matrix/MatrixHandler/MatrixUnary/CUDA/src/MatUnaryCUDAHandler.cu
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

#include "MatUnaryCUDAHandler.cuh"
#include "UnaryFunctions.hpp"
#include "GlobalParameters.hpp"

// First time initalization
static bool g_init{false};

// Matrix element-wise application of unary function
template<typename T>
__global__ void MatrixUnary(const T* A, T* B, int func_num, const size_t M, const size_t N) {
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < M) && (col < N)) {
        const auto idx = row * N + col;
        B[idx] = (*device_functions<T>[func_num])(A[idx]);
    }
}

// Custom unary function kernel
template<typename T>
void CustomUnaryKernel(const dim3 blocks, const dim3 threads, const T* A, T* B, FunctionTypeCuda<T> func, const size_t M, const size_t N) {
    // First time initalization
    if(false == g_init) {
        DeviceFunctionsInit<T> <<<1,1>>>();
        g_init = true;
    }
    // Launch kernel
    MatrixUnary<T> <<<blocks, threads>>> (A, B, device_functions_map<T>[func], M, N);
    // Device synchronize
    cudaDeviceSynchronize();
}

// Custom unary function kernel (float)
template void CustomUnaryKernel<float>(const dim3, const dim3, const float*, float*, FunctionTypeCuda<float>, const size_t, const size_t);
// Custom unary function kernel (double)
template void CustomUnaryKernel<double>(const dim3, const dim3, const double*, double*, FunctionTypeCuda<double>, const size_t, const size_t);
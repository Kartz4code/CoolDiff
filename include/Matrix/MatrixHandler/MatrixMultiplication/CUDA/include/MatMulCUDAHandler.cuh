/**
 * @file include/Matrix/MatrixHandler/MatrixMultiplication/CUDA/include/MatMulCUDAHandler.cuh
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

#pragma once

#if defined(USE_CUDA_BACKEND)
    #include <cuda.h>
    #include <cuda_runtime.h>

    #define TILE_SIZE 32

    // CUDA matrix multiplication kernel
    template<typename T>
    void MulKernel(const dim3, const dim3, const T*, const T*, T*, const size_t, const size_t, const size_t);
#endif
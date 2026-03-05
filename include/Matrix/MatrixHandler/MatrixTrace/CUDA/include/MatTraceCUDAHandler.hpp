/**
 * @file include/Matrix/MatrixHandler/MatrixTrace/CUDA/include/MatTraceCUDAHandler.hpp
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

#include "MatrixStaticHandler.hpp"
#include "Matrix.hpp"

#include "MatTraceCUDAHandler.cuh"

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatTraceCUDAHandler : public T {
    public:
        /* Matrix-Matrix numerical Hadamard product */
        void handle(const Matrix<Type>* rhs, Matrix<Type>*& result) {
            #if !defined(USE_CUDA_BACKEND)
                // If USE_CUDA_BACKEND is undefined
                T::handle(rhs, result);
                return;
            #else     
                // Dimensions of mat matrix
                const size_t nrows{rhs->getNumRows()};
                const size_t ncols{rhs->getNumColumns()};

                // Mat memory strategy
                const auto& rhs_strategy = rhs->allocatorType();

                // Assert squareness
                ASSERT((nrows == ncols), "Matrix is not square for trace computation");

                // CUDA handler
                CUDA_BACKEND_HANDLER(T::handle(rhs, result), rhs_strategy);

                // Pool matrix
                MemoryManager::MatrixPool(result, 1, 1, rhs_strategy);
            
                Type* rhs_ptr = const_cast<Matrix<Type>*>(rhs)->getMatrixPtr();
                Type* result_ptr = result->getMatrixPtr();

                // Define the block and grid sizes
                const dim3 threads(THREAD_SIZE, THREAD_SIZE);
                const dim3 blocks(  (ncols + threads.x - 1) / threads.x,
                                    (nrows + threads.y - 1) / threads.y );

                // Launch the kernel
                TraceKernel(blocks, threads, rhs_ptr, result_ptr, nrows);

                return;
            #endif
        }
};
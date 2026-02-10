/**
 * @file include/Matrix/MatrixHandler/MatrixHadamard/CUDA/include/MatHadamardCUDAHandler.hpp
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

#include "MatHadamardCUDAHandler.cuh"

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatHadamardCUDAHandler : public T {
    public:
        /* Matrix-Matrix numerical Hadamard product */
        void handle(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            #if !defined(USE_CUDA_BACKEND)
                // If USE_CUDA_BACKEND is undefined
                T::handle(lhs, rhs, result);
                return;
            #else     
                // Dimensions of LHS and RHS matrices
                const size_t lrows{lhs->getNumRows()};
                const size_t lcols{lhs->getNumColumns()};
                const size_t rcols{rhs->getNumColumns()};
                const size_t rrows{rhs->getNumRows()};

                // LHS/RHS memory strategies
                const auto& lhs_strategy = lhs->allocatorType();
                const auto& rhs_strategy = rhs->allocatorType();

                // Assert dimensions
                ASSERT((lrows == rrows) && (lcols == rcols), "Matrix addition dimensions mismatch");
                // Assert allocator
                ASSERT((lhs_strategy == rhs_strategy), "LHS and RHS matrices are in different memory spaces");

                // CUDA handler
                CUDA_BACKEND_HANDLER(T::handle(lhs, rhs, result), rhs_strategy);

                // Pool matrix
                MemoryManager::MatrixPool(result, lrows, lcols, rhs_strategy);

                // Get raw GPU pointers to result, left and right matrices
                const Type* left = const_cast<Matrix<Type>*>(lhs)->getMatrixPtr();
                const Type* right = const_cast<Matrix<Type>*>(rhs)->getMatrixPtr();
                Type* result_ptr = result->getMatrixPtr();

                // Define the block and grid sizes
                const dim3 threads(THREAD_SIZE, THREAD_SIZE);
                const dim3 blocks(  (lcols + threads.x - 1) / threads.x,
                                    (lrows + threads.y - 1) / threads.y );

                // Launch the kernel
                HadamardKernel(blocks, threads, left, right, result_ptr, lrows, lcols);

                return;
            #endif
        }
};
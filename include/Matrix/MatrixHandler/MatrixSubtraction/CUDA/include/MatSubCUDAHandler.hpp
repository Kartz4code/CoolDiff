/**
 * @file include/Matrix/MatrixHandler/MatrixSubtraction/CUDA/include/MatSubCUDAHandler.hpp
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

#include "MatSubCUDAHandler.cuh"

// CUDA subtraction handler
template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatSubCUDAHandler : public T {
    public:
        void handle(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            /* Matrix-Matrix numerical subtraction */

            // Dimensions of LHS and RHS matrices
            const size_t nrows{lhs->getNumRows()};
            const size_t ncols{rhs->getNumColumns()};
            const size_t lcols{lhs->getNumColumns()};
            const size_t rrows{rhs->getNumRows()};

            // Assert dimensions
            ASSERT((nrows == rrows) && (ncols == lcols), "Matrix subtraction dimensions mismatch");

            // Pool matrix
            MemoryManager::MatrixPool(result, nrows, ncols);

            // Get raw GPU pointers to result, left and right matrices
            const Type* left = const_cast<Matrix<Type>*>(lhs)->getMatrixPtr();
            const Type* right = const_cast<Matrix<Type>*>(rhs)->getMatrixPtr();
            Type* result_ptr = result->getMatrixPtr();

            // Define the block and grid sizes
            const dim3 threads(THREAD_SIZE, THREAD_SIZE);
            const dim3 blocks(  (ncols + threads.x - 1) / threads.x,
                                (nrows + threads.y - 1) / threads.y );

            // Launch the kernel
            SubKernel(blocks, threads, left, right, result_ptr, nrows, ncols);
            
            return;
        }
};
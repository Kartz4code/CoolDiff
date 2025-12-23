/**
 * @file include/Matrix/MatrixHandler/MatrixMultiplication/CUDA/include/MatScalarMulCUDAHandler.hpp
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

#pragma once

#include "MatrixStaticHandler.hpp"
#include "Matrix.hpp"

#include "MatScalarMulCUDAHandler.cuh"

// CUDA Scalar-Matrix Multiplication handler
template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatScalarMulCUDAHandler : public T {
    public:
        void handle(Type lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            /* Matrix-Scalar numerical multiplication */
            
            // Dimensions of RHS matrices
            const size_t nrows{rhs->getNumRows()};
            const size_t ncols{rhs->getNumColumns()};

            // Pool matrix
            MemoryManager::MatrixPool(result, nrows, ncols);

            // Get raw pointers to result, left and right matrices
            Type* right = const_cast<Matrix<Type>*>(rhs)->getMatrixPtr();
            Type* result_ptr = result->getMatrixPtr();

            // Define the block and grid sizes
            const dim3 threads(THREAD_SIZE, THREAD_SIZE);
            const dim3 blocks(  (ncols + threads.x - 1) / threads.x, 
                                (nrows + threads.y - 1) / threads.y  );

            
            // RHS matrix data size in bytes
            const size_t size_bytes = (rhs->getNumElem()*sizeof(Type));

            // Launch the kernel
            MulScalarKernel(blocks, threads, right, result_ptr, nrows, ncols, lhs);
            return;
        }
};
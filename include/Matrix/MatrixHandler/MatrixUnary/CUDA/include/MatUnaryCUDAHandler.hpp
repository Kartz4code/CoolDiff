/**
 * @file include/Matrix/MatrixHandler/MatrixUnary/CUDA/include/MatUnaryCUDAHandler.hpp
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

#include "MatUnaryCUDAHandler.cuh"

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatUnaryCUDAHandler : public T {
    public:
        void handle(const Matrix<Type>* mat, const FunctionType& func, Matrix<Type>*& result) {
            const size_t nrows{mat->getNumRows()};
            const size_t ncols{mat->getNumColumns()};

            // Pool matrix
            MemoryManager::MatrixPool(result, nrows, ncols);

            // Get raw pointers to result, left and right matrices
            Type* mat_ptr = const_cast<Matrix<Type>*>(mat)->getMatrixPtr();
            Type* result_ptr = result->getMatrixPtr();

            // Kernel launch parameters
            const dim3 threads(THREAD_SIZE, THREAD_SIZE);
            const dim3 blocks(  ((ncols + threads.x - 1) / threads.x), 
                                ((nrows + threads.y - 1) / threads.y)  );
                                
            // Launch the kernel
            CustomUnaryKernel<Type>(blocks, threads, mat_ptr, result_ptr, func, nrows, ncols);

            return;
        }
};
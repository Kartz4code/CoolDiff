/**
 * @file include/Matrix/MatrixHandler/MatrixKronProduct/CUDA/MatKronCUDAHandler.hpp
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

#include "MatKronCUDAHandler.cuh"

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatKronCUDAHandler : public T {
  public:
  void handle(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
    /* Matrix-Matrix numerical Kronocker product */

    // Dimensions of LHS and RHS matrices
    const size_t lr{lhs->getNumRows()};
    const size_t lc{lhs->getNumColumns()};
    const size_t rr{rhs->getNumRows()};
    const size_t rc{rhs->getNumColumns()};

    // Pool matrix
    MemoryManager::MatrixPool(result, (lr * rr), (lc * rc));

    // Get raw pointers to result, left and right matrices
    Type* left = const_cast<Matrix<Type>*>(lhs)->getMatrixPtr();
    Type* right = const_cast<Matrix<Type>*>(rhs)->getMatrixPtr();
    Type* result_ptr = result->getMatrixPtr();

    // Define the block and grid sizes
    const dim3 threads(THREAD_SIZE, THREAD_SIZE);
    const dim3 blocks(  ((lc * rc) + threads.x - 1) / threads.x,
                        ((lr * rr) + threads.y - 1) / threads.y );

    // Launch the kernel
    KronKernel(blocks, threads, left, right, result_ptr, lr, lc, rr, rc);

    return;
  }
};
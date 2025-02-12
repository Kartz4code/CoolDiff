/**
 * @file include/Matrix/MatrixHandler/MatrixHadamard/NaiveCPU/MatHadamardNaiveHandler.hpp
 *
 * @copyright 2023-2024 Karthik Murali Madhavan Rathai
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

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatHadamardNaiveHandler : public T {
    public:
      void handle(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
        // Dimensions of LHS and RHS matrices
        const size_t nrows{lhs->getNumRows()};
        const size_t ncols{rhs->getNumColumns()};
        const size_t lcols{lhs->getNumColumns()};
        const size_t rrows{rhs->getNumRows()};

        // Assert dimensions
        ASSERT((nrows == rrows) && (ncols == lcols), "Matrix Hadamard product dimensions mismatch");

        // Pool matrix
        MemoryManager::MatrixPool(nrows, ncols, result);

        // Get raw pointers to result, left and right matrices
        Type* res = result->getMatrixPtr();
        const Type* left = lhs->getMatrixPtr();
        const Type* right = rhs->getMatrixPtr();

        // For each element, perform addition
        const size_t size{nrows * ncols};
        std::transform(EXECUTION_PAR left, left + size, right, res,
                      [](const Type a, const Type b) { return a * b; });

        return;
    }
};
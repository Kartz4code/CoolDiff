/**
 * @file include/Matrix/MatrixHandler/MatrixDet/Eigen/MatDetEigenHandler.hpp
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

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatDetEigenHandler : public T {
    public:
      void handle(const Matrix<Type>* mat, Matrix<Type>*& result) {
        const size_t nrows{mat->getNumRows()};
        const size_t ncols{mat->getNumColumns()};
        // Assert squareness
        ASSERT((nrows == ncols), "Matrix is not square for determinant computation");
      
        // Pool matrix
        MemoryManager::MatrixPool(result, 1, 1);
      
        Type* mat_ptr = const_cast<Matrix<Type>*>(mat)->getMatrixPtr();
      
        // Eigen inverse
        const Eigen::Map<EigenMatrix> A(mat_ptr, nrows, ncols);
        const auto det_A = A.determinant();
      
        // Store result
        (*result)(0,0) = (Type)det_A;
      }
};
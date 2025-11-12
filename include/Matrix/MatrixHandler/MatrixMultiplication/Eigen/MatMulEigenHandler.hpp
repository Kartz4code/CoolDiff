/**
 * @file include/Matrix/MatrixHandler/MatrixMultiplication/Eigen/MatMulEigenHandler.hpp
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
class MatMulEigenHandler : public T {
    public:
        void handle(const Matrix<Type> *lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
            // Dimensions of LHS and RHS matrices
            const size_t lrows{lhs->getNumRows()};
            const size_t lcols{lhs->getNumColumns()};
            const size_t rcols{rhs->getNumColumns()};
            const size_t rrows{rhs->getNumRows()};

            // Assert dimensions
            ASSERT(lcols == rrows, "Matrix multiplication dimensions mismatch");

            // Pool matrix
            MemoryManager::MatrixPool(lrows, rcols, result);

            // Get raw pointers to result, left and right matrices
            Type* left = const_cast<Matrix<Type>*>(lhs)->getMatrixPtr();
            Type* right = const_cast<Matrix<Type>*>(rhs)->getMatrixPtr();

            
            const Eigen::Map<EigenMatrix> left_eigen(left, lrows, lcols);
            const Eigen::Map<EigenMatrix> right_eigen(right, rrows, rcols);

            const auto& result_eigen = left_eigen*right_eigen; 

            Eigen::Map<EigenMatrix>(result->getMatrixPtr(), 
                                    result_eigen.rows(), 
                                    result_eigen.cols()) = result_eigen;

            return;
        }
};

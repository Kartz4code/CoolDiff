/**
 * @file include/Matrix/MatrixHandler/MatrixMultiplication/Eigen/MatScalarMulEigenHandler.hpp
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

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatScalarMulEigenHandler : public T {
    public:
        void handle(Type lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            /* Matrix-Scalar numerical multiplication */
            
            // Dimensions of RHS matrices
            const size_t nrows{rhs->getNumRows()};
            const size_t ncols{rhs->getNumColumns()};
            const size_t nelems{rhs->getNumElem()};

            // Pool matrix
            MemoryManager::MatrixPool(result, nrows, ncols);

            // Get raw pointers to result, left and right matrices
            Type* right = const_cast<Matrix<Type>*>(rhs)->getMatrixPtr();

            const Eigen::Map<EigenMatrix> right_eigen(right, nrows, ncols);
            const auto& result_eigen = (right_eigen.array() * lhs).matrix();
            Eigen::Map<EigenMatrix>(result->getMatrixPtr(), 
                                    result_eigen.rows(), 
                                    result_eigen.cols()) = result_eigen;
            return;
        }
};
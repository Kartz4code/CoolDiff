/**
 * @file include/Matrix/MatrixHandler/MatTranspose/Eigen/MatTransposeEigenHandler.hpp
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
class MatTransposeEigenHandler : public T {
    public:
        /* Matrix transpose operation */
        void handle(const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Dimensions of mat matrix
            const size_t nrows{rhs->getNumRows()};
            const size_t ncols{rhs->getNumColumns()};

            // Mat memory strategy
            const auto& rhs_strategy = rhs->allocatorType();

            // Eigen handler
            EIGEN_BACKEND_HANDLER(T::handle(rhs, result), rhs_strategy);

            // Pool matrix
            MemoryManager::MatrixPool(result, ncols, nrows, rhs_strategy);

            // Get raw pointers to result, left and right matrices
            Type* rhs_get = const_cast<Matrix<Type>*>(rhs)->getMatrixPtr();

            const Eigen::Map<EigenMatrix> rhs_eigen(rhs_get, nrows, ncols);
            const auto& result_eigen = rhs_eigen.transpose(); 

            Eigen::Map<EigenMatrix>(result->getMatrixPtr(), 
                                    result_eigen.rows(), 
                                    result_eigen.cols()) = result_eigen;

            return;
        }
};
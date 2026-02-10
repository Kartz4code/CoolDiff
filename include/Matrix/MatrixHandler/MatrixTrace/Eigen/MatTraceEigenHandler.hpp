/**
 * @file include/Matrix/MatrixHandler/MatrixTrace/Eigen/MatTraceEigenHandler.hpp
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
class MatTraceEigenHandler : public T {
    public:
      /* Matrix trace computation */
      void handle(const Matrix<Type>* rhs, Matrix<Type>*& result) {
        // Dimensions of mat matrix
        const size_t nrows{rhs->getNumRows()};
        const size_t ncols{rhs->getNumColumns()};

        // Mat memory strategy
        const auto& rhs_strategy = rhs->allocatorType();

        // Assert squareness
        ASSERT((nrows == ncols), "Matrix is not square for trace computation");
      
        // Eigen handler
        EIGEN_BACKEND_HANDLER(T::handle(rhs, result), rhs_strategy);

        // Pool matrix
        MemoryManager::MatrixPool(result, 1, 1, rhs_strategy);
      
        Type* rhs_ptr = const_cast<Matrix<Type>*>(rhs)->getMatrixPtr();
      
        // Eigen inverse
        const Eigen::Map<EigenMatrix> A(rhs_ptr, nrows, ncols);
        const auto trace_A = A.trace();
      
        // Store result
        (*result)(0,0) = (Type)trace_A;
      }
};
/**
 * @file include/Matrix/MatrixHandler/MatrixUnary/NaiveCPU/MatUnaryNaiveHandler.hpp
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
class MatUnaryNaiveHandler : public T {
    public:
      /* Matrix unary operation */
      void handle(const Matrix<Type>* rhs, const FunctionType& func, Matrix<Type>*& result) {
        // Dimensions of mat matrix
        const size_t nrows{rhs->getNumRows()};
        const size_t ncols{rhs->getNumColumns()};

        // Mat memory strategy
        const auto& rhs_strategy = rhs->allocatorType();

        // Pool matrix
        MemoryManager::MatrixPool(result, nrows, ncols, rhs_strategy);

        // Get raw pointers to result and right matrix
        Type* res = result->getMatrixPtr();
        const Type* right = rhs->getMatrixPtr();

        const size_t size{nrows * ncols};

        // For each element, perform operation
        std::transform(EXECUTION_PAR right, right + size, res, [func](const Type a) { return func(a); });
      }
};
/**
 * @file include/Matrix/MatrixHandler/MatrixAddHandler/NaiveCPU/ZeroMatAddHandler.hpp
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
#include "MatrixZeroOps.hpp"

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class ZeroMatAddHandler : public T {
    public:
        void handle(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Dimensions of LHS and RHS matrices
            const size_t nrows{lhs->getNumRows()};
            const size_t ncols{rhs->getNumColumns()};
            const size_t lcols{lhs->getNumColumns()};
            const size_t rrows{rhs->getNumRows()};

            // Assert dimensions
            ASSERT((nrows == rrows) && (ncols == lcols), "Matrix addition dimensions mismatch");

            #if defined(NAIVE_IMPL)
                /* Zero matrix special check */
                if (auto* it = ZeroMatAdd(lhs, rhs); nullptr != it) {
                    result = const_cast<Matrix<Type>*>(it);
                    return;
                }
                /* Zero matrix numerical check */
                #if defined(NUMERICAL_CHECK)
                    else if (auto* it = ZeroMatAddNum(lhs, rhs); nullptr != it) {
                        result = const_cast<Matrix<Type>*>(it);
                        return;
                    }
                #endif
            #endif

            // Chain of responsibility
            T::handle(lhs, rhs, result);
        }
};

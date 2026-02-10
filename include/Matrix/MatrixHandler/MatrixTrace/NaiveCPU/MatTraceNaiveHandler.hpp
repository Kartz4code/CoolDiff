/**
 * @file include/Matrix/MatrixHandler/MatrixTrace/NaiveCPU/MatTraceNaiveHandler.hpp
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
class MatTraceNaiveHandler : public T {
    public:
        /* Matrix trace operation */
        void handle(const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Dimensions of mat matrix
            const size_t nrows{rhs->getNumRows()};
            const size_t ncols{rhs->getNumColumns()};

            // Mat memory strategy
            const auto& rhs_strategy = rhs->allocatorType();
            
            // Assert squareness
            ASSERT((nrows == ncols), "Matrix is not square for trace computation");
        
            // Pool matrix
            MemoryManager::MatrixPool(result, 1, 1, rhs_strategy);
        
            // Indices for outer loop and inner loop
            const auto outer_idx = CoolDiff::Common::Range<size_t>(0, nrows);

            // Naive matrix-matrix multiplication
            Type tmp{};
            std::for_each(EXECUTION_PAR outer_idx.begin(), outer_idx.end(), [&](const size_t n) {
                    tmp = tmp + (*rhs)(n,n);
            });
            (*result)[0] = tmp;
        }
};
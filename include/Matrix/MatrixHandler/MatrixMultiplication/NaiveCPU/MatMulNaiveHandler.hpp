/**
 * @file include/Matrix/MatrixHandler/MatrixMultiplication/NaiveCPU/MatMulNaiveHandler.hpp
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
class MatMulNaiveHandler : public T {
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
            MemoryManager::MatrixPool(result, lrows, rcols);
            ASSERT(lhs->m_nidx != result->m_nidx, "Matrix multiplication result matrix and lhs matrix are the same");
            ASSERT(rhs->m_nidx != result->m_nidx, "Matrix multiplication result matrix and lhs matrix are the same");

            // Transpose of rhs
            #if defined(MATRIX_TRANSPOSED_MUL)
                static MatTransposeNaiveHandler transpose_handler{nullptr};
                transpose_handler.handle(rhs, mp_rhs_transpose);
            #endif

            // Indices for outer loop and inner loop
            const auto outer_idx = CoolDiff::Common::Range<size_t>(0, lrows * rcols);
            const auto inner_idx = CoolDiff::Common::Range<size_t>(0, rrows);

            // Naive matrix-matrix multiplication
            std::for_each(EXECUTION_PAR outer_idx.begin(), outer_idx.end(), [&](const size_t n) {
                    // Row and column index
                    const size_t j = (n % rcols);
                    const size_t i = ((n - j) / rcols);

                    // Inner product
                    Type tmp{};
                    std::for_each(EXECUTION_SEQ inner_idx.begin(), inner_idx.end(),
                                [&](const size_t m) {
                                    #if defined(MATRIX_TRANSPOSED_MUL)
                                        tmp += ((*lhs)(i, m) * (*mp_rhs_transpose)(j, m));
                                    #else
                                        tmp += ((*lhs)(i, m) * (*rhs)(m,j));
                                    #endif
                                });

                    // Store result
                    (*result)(i, j) = std::exchange(tmp, (Type)(0));
                });
        }
};

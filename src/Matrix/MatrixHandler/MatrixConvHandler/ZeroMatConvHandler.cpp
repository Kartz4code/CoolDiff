/**
 * @file include/Matrix/MatrixHandler/MatrixAddition/MatAddNaiveHandler.cpp
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

#include "ZeroMatConvHandler.hpp"
#include "Matrix.hpp"
#include "MatrixZeroOps.hpp"

void ZeroMatConvHandler::handle(const size_t stride_x, const size_t stride_y,
                                const size_t pad_x, const size_t pad_y,
                                const Matrix<Type>* lhs, const Matrix<Type>* rhs,
                                Matrix<Type>*& result) {

    // Stride must be strictly non-negative
    ASSERT(((int)stride_x > 0) && ((int)stride_y > 0), "Stride is not strictly non-negative");
    // Padding must be positive
    ASSERT(((int)pad_x >= 0) && ((int)pad_y >= 0), "Stride is not positive");

    // Convolution matrix dimensions (RHS)
    const size_t crows = rhs->getNumRows();
    const size_t ccols = rhs->getNumColumns();

    // Result matrix dimensions
    const size_t rows = (((lhs->getNumRows() + (2 * pad_x) - crows) / stride_x) + 1);
    const size_t cols = (((lhs->getNumColumns() + (2 * pad_y) - ccols) / stride_y) + 1);

    // Matrix-Matrix convolution result dimensions must be strictly non-negative
    ASSERT(((int)rows > 0 || (int)cols > 0), "Matrix-Matrix convolution dimensions invalid");       

    #if defined(NAIVE_IMPL)
        /* Zero matrix special check */
        if (auto *it = ZeroMatConv(rows, cols, lhs, rhs); nullptr != it) {
            result = const_cast<Matrix<Type>*>(it);
            return;
        }
        /* Zero matrix numerical check */
        else if (auto *it = ZeroMatConvNum(rows, cols, lhs, rhs); nullptr != it) {
            result = const_cast<Matrix<Type>*>(it);
            return;
        }
    #endif

    // Chain of responsibility
    MatrixHandler::handle(stride_x, stride_y, pad_x, pad_y, lhs, rhs, result);
}
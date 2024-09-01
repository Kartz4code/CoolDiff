/**
 * @file src/Matrix/MatrixHandler/MatMulNaiveHandler.cpp
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

#include "MatMulNaiveHandler.hpp"
#include "Matrix.hpp"

void MatMulNaiveHandler::handle(Matrix<Type>* lhs, 
                                Matrix<Type>* rhs, 
                                Matrix<Type>*& result) {
    // Null pointer check
    NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
    NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

    // If result is nullptr, then create a new resource
    const size_t lrows = lhs->getNumRows();
    const size_t rcols = rhs->getNumColumns();
    const size_t rrows = rhs->getNumRows();
    if (nullptr == result) {
        result = CreateMatrixPtr<Type>(lrows, rcols);
    }

    // Get raw pointers to result, left and right matrices
    Matrix<Type>& res = *result;
    Matrix<Type>& left = *lhs;
    Matrix<Type>& right = *rhs;   
    
    // Indices for outer loop
    Vector<size_t> elemM(lrows*rcols);
    std::iota(elemM.begin(), elemM.end(), 0);
    // Indices for inner loop
    Vector<size_t> elemR(rrows);
    std::iota(elemR.begin(), elemR.end(), 0);
                    
    // Naive matrix-matrix multiplication
    std::for_each(EXECUTION_PAR 
                  elemM.begin(), elemM.end(), 
                  [&](const size_t n) {
                    // Row and column index
                    const size_t j = n%rcols;
                    const size_t i = (n-j)/rcols;

                    // Inner product
                    Type tmp{};
                    std::for_each(EXECUTION_SEQ 
                                    elemR.begin(), elemR.end(), 
                                    [&](const size_t m) {
                                    tmp += left(i,m)*right(m,j);
                  });

                  // Store result
                  res(i,j) = std::exchange(tmp, (Type)(0));
    });

    return;
}
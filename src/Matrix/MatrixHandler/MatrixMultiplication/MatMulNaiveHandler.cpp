/**
 * @file src/Matrix/MatrixHandler/MatrixMultiplication/MatMulNaiveHandler.cpp
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

void MatMulNaiveHandler::handle(const Matrix<Type> *lhs, const Matrix<Type> *rhs,
                                Matrix<Type> *&result) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  // If result is nullptr, then create a new resource
  const size_t lrows = lhs->getNumRows();
  const size_t rcols = rhs->getNumColumns();
  const size_t rrows = rhs->getNumRows();
  if (nullptr == result) {
    result = CreateMatrixPtr<Type>(lrows, rcols);
  } else if ((lrows != result->getNumRows()) ||
             (rcols != result->getNumColumns())) {
    result = CreateMatrixPtr<Type>(lrows, rcols);
  }

  // Indices for outer loop and inner loop
  auto outer_idx = Range<size_t>(0, lrows * rcols);
  auto inner_idx = Range<size_t>(0, rrows);

  // Naive matrix-matrix multiplication
  std::for_each(
      EXECUTION_PAR outer_idx.begin(), outer_idx.end(), [&](const size_t n) {
        // Row and column index
        const size_t j = n % rcols;
        const size_t i = (n - j) / rcols;

        // Inner product
        Type tmp{};
        std::for_each(EXECUTION_SEQ inner_idx.begin(), inner_idx.end(),
                      [&](const size_t m) { tmp += (*lhs)(i, m) * (*rhs)(m, j); });

        // Store result
        (*result)(i, j) = std::exchange(tmp, (Type)(0));
      });

  return;
}
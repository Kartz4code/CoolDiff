/**
 * @file include/Matrix/MatrixHelper.hpp
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

#include "Matrix.hpp"

template <typename T>
void DevalR(T &exp, const Matrix<Variable> &X, Matrix<Type> *&result) {
  const size_t nrows_x = X.getNumRows();
  const size_t ncols_x = X.getNumColumns();

  if (nullptr == result) {
    result = Matrix<Type>::MatrixFactory::CreateMatrixPtr(nrows_x, ncols_x);
  } else if ((nrows_x != result->getNumRows()) ||
            (ncols_x != result->getNumColumns())) {
    result = Matrix<Type>::MatrixFactory::CreateMatrixPtr(nrows_x, ncols_x);
  }

  const size_t n_size = X.getNumElem();
  // Precompute (By design, the operation is serial)
  if constexpr (true == std::is_same_v<Expression, T>) {
    PreComp(exp);
    std::transform(EXECUTION_SEQ X.getMatrixPtr(), X.getMatrixPtr() + n_size,
                  result->getMatrixPtr(),
                  [&exp](const auto &v) { return CoolDiff::TensorR1::DevalR(exp, v); });
  } else {
    // Create a new expression
    Expression exp2{exp};
    CoolDiff::TensorR1::PreComp(exp2);
    std::transform(EXECUTION_SEQ X.getMatrixPtr(), X.getMatrixPtr() + n_size,
                  result->getMatrixPtr(),
                  [&exp2](const auto &v) { return CoolDiff::TensorR1::DevalR(exp2, v); });
  }
}
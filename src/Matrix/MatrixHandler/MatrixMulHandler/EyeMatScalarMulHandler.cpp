/**
 * @file src/Matrix/MatrixHandler/MatrixMulHandler/EyeMatScalarMulHandler.cpp
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

#include "EyeMatScalarMulHandler.hpp"
#include "Matrix.hpp"
#include "MatrixEyeOps.hpp"

void MulEye(Type val, const Matrix<Type> *it, Matrix<Type> *&result) {
  /*
    Rows and columns of result matrix and if result is nullptr or if dimensions
    mismatch, then create a new matrix resource
  */
  const size_t nrows{it->getNumRows()};
  const size_t ncols{it->getNumColumns()};

  // Pool matrix
  MemoryManager::MatrixPool(nrows, ncols, result);

  // Diagonal indices (Modification)
  const auto diag_idx = Range<size_t>(0, nrows);
  std::for_each(EXECUTION_PAR diag_idx.begin(), diag_idx.end(),
                [&](const size_t i) { (*result)(i, i) = val; });
}

void EyeMatScalarMulHandler::handle(Type lhs, const Matrix<Type> *rhs,
                                    Matrix<Type> *&result) {
#if defined(NAIVE_IMPL)
  /* Eye matrix special check */
  if (auto *it = EyeMatScalarMul(lhs, rhs); nullptr != it) {
    MulEye(lhs, rhs, result);
    return;
  }

  /* Eye matrix numerical check */
  else if (auto *it = EyeMatScalarMulNum(lhs, rhs); nullptr != it) {
    MulEye(lhs, rhs, result);
    return;
  }
#endif

  // Chain of responsibility
  MatrixHandler::handle(lhs, rhs, result);
}
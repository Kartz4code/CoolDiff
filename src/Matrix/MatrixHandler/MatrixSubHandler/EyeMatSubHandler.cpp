/**
 * @file src/Matrix/MatrixHandler/MatrixSubHandler/EyeMatSubHandler.cpp
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

#include "EyeMatSubHandler.hpp"
#include "Matrix.hpp"
#include "MatrixEyeOps.hpp"

void SubEyeRHS(const Matrix<Type> *it, Matrix<Type> *&result) {
  /*
    Rows and columns of result matrix and if result is nullptr or if dimensions
    mismatch, then create a new matrix resource
  */
  const size_t nrows{it->getNumRows()};
  const size_t ncols{it->getNumColumns()};
  
  // Pool matrix
  MemoryManager::MatrixPool(nrows, ncols, result);

  // Copy all LHS matrix value into result
  *result = *it;

  // Iteration elements (Along the diagonal)
  const auto idx = Range<size_t>(0, nrows);
  // For each execution
  std::for_each(EXECUTION_PAR idx.begin(), idx.end(), [&](const size_t i) {
    (*result)(i, i) = (*it)(i, i) - (Type)(1);
  });
}

void SubEyeLHS(const Matrix<Type> *it, Matrix<Type> *&result) {
  /*
    Rows and columns of result matrix and if result is nullptr or if dimensions
    mismatch, then create a new matrix resource
  */
  const size_t nrows{it->getNumRows()};
  const size_t ncols{it->getNumColumns()};
  
  // Pool matrix
  MemoryManager::MatrixPool(nrows, ncols, result);

  // Iteration elements
  const auto idx = Range<size_t>(0, nrows * ncols);
  // For each execution
  std::for_each(EXECUTION_PAR idx.begin(), idx.end(), [&](const size_t n) {
    const size_t j = n % ncols;
    const size_t i = (n - j) / ncols;
    (*result)(i, j) =
        ((i == j) ? ((Type)(1) - (*it)(i, j)) : ((Type)(-1) * (*it)(i, j)));
  });
}

void EyeMatSubHandler::handle(const Matrix<Type> *lhs, 
                              const Matrix<Type> *rhs,
                              Matrix<Type> *&result) {

// Rows and columns of result matrix and if result is nullptr, then create a
// new resource
const size_t nrows{lhs->getNumRows()};
const size_t ncols{rhs->getNumColumns()};
const size_t lcols{lhs->getNumColumns()};
const size_t rrows{rhs->getNumRows()};

// Assert dimensions
ASSERT((nrows == rrows) && (ncols == lcols), "Matrix subtraction dimensions mismatch");

#if defined(NAIVE_IMPL)
  /* Zero matrix special check */
  if (auto *it = EyeMatSub(lhs, rhs); nullptr != it) {
    if (it == lhs) {
      SubEyeRHS(it, result);
    } else if (it == rhs) {
      SubEyeLHS(it, result);
    } else {
      result = const_cast<Matrix<Type> *>(it);
    }
    return;
  }
  /* Zero matrix numerical check */
  else if (auto *it = EyeMatSubNum(lhs, rhs); nullptr != it) {
    if (it == lhs) {
      SubEyeRHS(it, result);
    } else if (it == rhs) {
      SubEyeLHS(it, result);
    } else {
      result = const_cast<Matrix<Type> *>(it);
    }
    return;
  }
#endif

  // Chain of responsibility
  MatrixHandler::handle(lhs, rhs, result);
}
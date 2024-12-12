/**
 * @file src/Matrix/MatrixHandler/MatrixSubHandler/ZeroMatSubHandler.cpp
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

#include "ZeroMatSubHandler.hpp"
#include "Matrix.hpp"
#include "MatrixZeroOps.hpp"

void SubZero(const Matrix<Type>* it, Matrix<Type>*& result) {
  /*
    Rows and columns of result matrix and if result is nullptr or if dimensions
    mismatch, then create a new matrix resource
  */
  const size_t nrows{it->getNumRows()};
  const size_t ncols{it->getNumColumns()};

  // Pool matrix
  MemoryManager::MatrixPool(nrows, ncols, result);

  std::transform(EXECUTION_PAR it->getMatrixPtr(),
                 it->getMatrixPtr() + it->getNumElem(), result->getMatrixPtr(),
                 [](const auto &i) { return ((Type)(-1) * i); });
}

void ZeroMatSubHandler::handle(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
  // Rows and columns of result matrix and if result is nullptr, then create a new resource
  const size_t nrows{lhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};
  const size_t lcols{lhs->getNumColumns()};
  const size_t rrows{rhs->getNumRows()};

  // Assert dimensions
  ASSERT((nrows == rrows) && (ncols == lcols), "Matrix subtraction dimensions mismatch");

#if defined(NAIVE_IMPL)
  /* Zero matrix special check */
  if (auto* it = ZeroMatSub(lhs, rhs); nullptr != it) {
    if (it == lhs) {
      result = const_cast<Matrix<Type>*>(lhs);
    } else if (it == rhs) {
      if(-1 == it->getMatType()) {
        SubZero(it, result);
      } else {
        MatrixHandler::handle(lhs, rhs, result);
      }
    } else {
      result = const_cast<Matrix<Type>*>(it);
    }
    return;
  }
  /* Zero matrix numerical check */
  #if defined(NUMERICAL_CHECK)
    else if (auto* it = ZeroMatSubNum(lhs, rhs); nullptr != it) {
      if (it == lhs) {
        result = const_cast<Matrix<Type>*>(lhs);
      } else if (it == rhs) {
        if(-1 == it->getMatType()) {
          SubZero(it, result);
        } else {
          MatrixHandler::handle(lhs, rhs, result);
        }
      } else {
        result = const_cast<Matrix<Type>*>(it);
      }
      return;
    }
  #endif
#endif

  // Chain of responsibility
  MatrixHandler::handle(lhs, rhs, result);
}
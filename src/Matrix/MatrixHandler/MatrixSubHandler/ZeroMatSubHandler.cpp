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

void SubZero(const Matrix<Type> *it, Matrix<Type> *&result) {
  /*
    Rows and columns of result matrix and if result is nullptr or if dimensions
    mismatch, then create a new matrix resource
  */
  const size_t nrows{it->getNumRows()};
  const size_t ncols{it->getNumColumns()};
  if (nullptr == result) {
    result = CreateMatrixPtr<Type>(nrows, ncols);
  } else if ((nrows != result->getNumRows()) ||
             (ncols != result->getNumColumns())) {
    result = CreateMatrixPtr<Type>(nrows, ncols);
  }

  std::transform(EXECUTION_PAR 
                 it->getMatrixPtr(), it->getMatrixPtr() + it->getNumElem(), 
                 result->getMatrixPtr(), 
                 [](const auto& i) {
                    return (Type)(-1)*i;
                 });
}

void ZeroMatSubHandler::handle(const Matrix<Type> *lhs, 
                               const Matrix<Type> *rhs,
                               Matrix<Type> *&result) {
#if defined(NAIVE_IMPL)
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Zero matrix special check */
  if (auto *it = ZeroMatSub(lhs, rhs); nullptr != it) {
    if(it == lhs) {
        result = const_cast<Matrix<Type>*>(lhs);
    } else if(it == rhs) {
        SubZero(it, result);
    } else {
        result = const_cast<Matrix<Type>*>(it);
    }
    return;
  }
  /* Zero matrix numerical check */
  else if (auto *it = ZeroMatSubNum(lhs, rhs); nullptr != it) {
    if(it == lhs) {
        result = const_cast<Matrix<Type>*>(lhs);
    } else if(it == rhs) {
        SubZero(it, result);
    } else {
        result = const_cast<Matrix<Type>*>(it);
    }
    return;
  }
#endif

  // Chain of responsibility
  MatrixHandler::handle(lhs, rhs, result);
}
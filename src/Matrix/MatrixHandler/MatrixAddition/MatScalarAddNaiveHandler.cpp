/**
 * @file src/Matrix/MatrixHandler/MatrixAddition/MatScalarAddNaiveHandler.cpp
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

#include "MatScalarAddNaiveHandler.hpp"
#include "Matrix.hpp"

void MatScalarAddNaiveHandler::handle(Type lhs,
                                      const Matrix<Type> *rhs,
                                      Matrix<Type> *&result) {
  // Null pointer check   
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Matrix-Scalar numerical addition */
  // Rows and columns of result matrix and if result is nullptr, then create a
  // new resource
  const size_t nrows = rhs->getNumRows();
  const size_t ncols = rhs->getNumColumns();
  
  if (nullptr == result) {
    result = CreateMatrixPtr<Type>(nrows, ncols);
  } else if ((nrows != result->getNumRows()) ||
             (ncols != result->getNumColumns())) {
    result = CreateMatrixPtr<Type>(nrows, ncols);
  }

  // Get raw pointers to result, left and right matrices
  const Type *res = result->getMatrixPtr();
  const Type *right = rhs->getMatrixPtr();

  const size_t size{nrows * ncols};
  // For each element, perform addition
  std::transform(EXECUTION_PAR right, right + size, res,
                [&lhs](const Type value) { return lhs + value; });
}
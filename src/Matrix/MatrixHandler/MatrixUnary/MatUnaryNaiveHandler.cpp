/**
 * @file
 * src/Matrix/MatrixHandler/MatrixUnary/MatUnaryNaiveHandler.cpp
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

#include "MatUnaryNaiveHandler.hpp"
#include "Matrix.hpp"
#include "MatrixEyeOps.hpp"

void MatUnaryNaiveHandler::handle(const Matrix<Type>* mat, const FunctionType1& func, Matrix<Type>*& result) {

  const size_t nrows{mat->getNumRows()};
  const size_t ncols{mat->getNumColumns()};

  // Pool matrix
  MemoryManager::MatrixPool(nrows, ncols, result);

  // Get raw pointers to result and right matrix
  Type *res = result->getMatrixPtr();
  const Type* right = mat->getMatrixPtr();

  const size_t size{nrows * ncols};

  // For each element, perform operation
  std::transform(EXECUTION_PAR right, right + size, res, [func](const Type a) { return func(a); });
}
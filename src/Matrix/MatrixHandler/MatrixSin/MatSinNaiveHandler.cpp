/**
 * @file src/Matrix/MatrixHandler/MatrixSin/MatSinNaiveHandler.cpp
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

#include "MatSinNaiveHandler.hpp"
#include "Matrix.hpp"

void MatSinNaiveHandler::handle(const Matrix<Type>* mat, Matrix<Type>*& result) {

  /* Matrix sin */
  // Rows and columns of result matrix and if result is nullptr, then create a
  // new resource

  const size_t nrows{mat->getNumRows()};
  const size_t ncols{mat->getNumColumns()};

  // Pool matrix
  MemoryManager::MatrixPool(nrows, ncols, result);

  // Get raw pointers to result and right matrix
  Type *res = result->getMatrixPtr();
  const Type* right = mat->getMatrixPtr();

  const size_t size{nrows * ncols};
  // For each element, perform operation
  std::transform(EXECUTION_PAR right, right + size, res, [](const Type a) { return std::sin(a); });
  
  return;  
}

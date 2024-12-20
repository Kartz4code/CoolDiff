/**
 * @file
 * src/Matrix/MatrixHandler/MatrixUnaryHandler/ZeroMatUnaryHandler.cpp
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

#include "ZeroMatUnaryHandler.hpp"
#include "Matrix.hpp"
#include "MatrixZeroOps.hpp"

void ZeroMatUnaryHandler::handle(const Matrix<Type>* mat, const FunctionType1& func, Matrix<Type>*& result) {
#if defined(NAIVE_IMPL)
  /* Zero matrix special check */
  if (true == IsZeroMatrix(mat)) {
    // Rows and columns of result matrix
    const size_t nrows{mat->getNumRows()};
    const size_t ncols{mat->getNumColumns()};

    // Result matrix is transposed zero matrix
    MemoryManager::MatrixPool(ncols, nrows, result);

    // Zero matrix fill
    std::fill_n(EXECUTION_PAR result->getMatrixPtr(), result->getNumElem(), func((Type)0));
    return;
  }
#endif
  // Chain of responsibility
  MatrixHandler::handle(mat, func, result);
}
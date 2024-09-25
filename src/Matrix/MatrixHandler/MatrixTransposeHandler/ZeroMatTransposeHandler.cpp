/**
 * @file src/Matrix/MatrixHandler/MatrixTransposeHandler/ZeroMatTransposeHandler.cpp
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

#include "ZeroMatTransposeHandler.hpp"
#include "Matrix.hpp"
#include "MatrixZeroOps.hpp"

void ZeroMatTransposeHandler::handle(const Matrix<Type> * mat, Matrix<Type> *& result) {
#if defined(NAIVE_IMPL)
  /* Zero matrix special check */
  if(true == IsZeroMatrix(mat)) {
    // Rows and columns of result matrix
    const size_t nrows{mat->getNumRows()};
    const size_t ncols{mat->getNumColumns()};

    // Result matrix is transposed zero matrix
    if (nullptr == result) {
      result = CreateMatrixPtr<Type>(ncols, nrows, MatrixSpl::ZEROS);
      return;
    } else if ((ncols != result->getNumRows()) ||
               (nrows != result->getNumColumns())) {  
      result = CreateMatrixPtr<Type>(ncols, nrows, MatrixSpl::ZEROS);
      return;
    } else if(-1 != result->getMatType()) {
      result = CreateMatrixPtr<Type>(ncols, nrows, MatrixSpl::ZEROS);
      return;
    }
    return;
  }
#endif
  // Chain of responsibility
  MatrixHandler::handle(mat, result);
}
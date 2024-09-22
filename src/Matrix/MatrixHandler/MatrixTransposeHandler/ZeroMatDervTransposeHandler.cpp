/**
 * @file src/Matrix/MatrixHandler/MatrixTransposeHandler/ZeroMatDervTransposeHandler.cpp
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

#include "ZeroMatDervTransposeHandler.hpp"
#include "Matrix.hpp"
#include "MatrixZeroOps.hpp"

void ZeroMatDervTransposeHandler::handle(const size_t nrows_f, const size_t ncols_f, 
                                         const size_t nrows_x, const size_t ncols_x, 
                                         const Matrix<Type> * mat, Matrix<Type>*& result) {
#if defined(NAIVE_IMPL)
  /* Zero matrix special check */
  if(true == IsZeroMatrix(mat)) {
    // Result matrix dimensions
    const size_t nrows = ncols_f*nrows_x;
    const size_t ncols = nrows_f*ncols_x;

    // Result transposed derivative matrix   
    if (nullptr == result) {
        result = CreateMatrixPtr<Type>(nrows, ncols, MatrixSpl::ZEROS);
        return;
    } else if ((nrows != result->getNumRows()) ||
               (ncols != result->getNumColumns())) {          
        result = CreateMatrixPtr<Type>(nrows, ncols, MatrixSpl::ZEROS);
        return;
    }
    return;
  }
#endif
  // Chain of responsibility
  MatrixHandler::handle(nrows_f, ncols_f, nrows_x, ncols_x, mat, result);
}
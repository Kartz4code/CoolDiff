/**
 * @file src/Matrix/MatrixHandler/MatrixTranspose/MatTransposeNaiveHandler.cpp
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

#include "MatTransposeNaiveHandler.hpp"
#include "Matrix.hpp"

void MatTransposeNaiveHandler::handle(const Matrix<Type> * mat, Matrix<Type> *& result) {
  const size_t nrows{mat->getNumRows()};
  const size_t ncols{mat->getNumColumns()};

  MatrixPool(ncols, nrows, result);

  const auto idx = Range<size_t>(0, nrows * ncols);
  std::for_each(EXECUTION_PAR idx.begin(), idx.end(),
                [&](const size_t n) {
                    // Row and column index
                    const size_t j = n % ncols;
                    const size_t i = (n - j) / ncols;
                    (*result)(j,i) = std::conj((*mat)(i,j));
                }
  );
}
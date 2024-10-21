/**
 * @file src/Matrix/Matrix.cpp
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

#include "Matrix.hpp"
#include "MemoryManager.hpp"

Matrix<Type> *DervMatrix(const size_t frows, const size_t fcols,
                         const size_t xrows, const size_t xcols) {
                         const size_t drows = frows * xrows;
                         const size_t dcols = fcols * xcols;
  Matrix<Type> *dresult = Matrix<Type>::MatrixFactory::CreateMatrixPtr(drows, dcols);

  // Vector of indices in X matrix
  const auto idx = Range<size_t>(0, xrows * xcols);
  // Logic for Kronecker product (With ones)
  std::for_each(EXECUTION_PAR idx.begin(), idx.end(), [&](const size_t n) {
    const size_t j = n % xcols;
    const size_t i = (n - j) / xcols;
    // Inner loop
    (*dresult)(i * xrows + i, j * xcols + j) = (Type)(1);
  });

  return dresult;
}
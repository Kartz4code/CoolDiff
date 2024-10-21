/**
 * @file src/Matrix/MatrixHandler/MatrixConvolution/MatConvNaiveHandler.cpp
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

#include "MatConvNaiveHandler.hpp"
#include "MatOperators.hpp"
#include "Matrix.hpp"
#include "MatrixBasics.hpp"

void MatConvNaiveHandler::handle(const size_t stride_x, const size_t stride_y,
                                 const size_t pad_x, const size_t pad_y,
                                 const Matrix<Type>* lhs,
                                 const Matrix<Type>* rhs,
                                 Matrix<Type>*& result) {

  // Stride must be strictly non-negative
  ASSERT(((int)stride_x > 0) && ((int)stride_y > 0), "Stride is not strictly non-negative");
  // Padding must be positive
  ASSERT(((int)pad_x >= 0) && ((int)pad_y >= 0), "Stride is not positive");

  // One time initialization
  if (false == m_initialized) {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
    m_initialized = true;
  }

  // Convolution matrix dimensions (RHS)
  const size_t crows = rhs->getNumRows();
  const size_t ccols = rhs->getNumColumns();

  // Result matrix dimensions
  const size_t rows = (((lhs->getNumRows() + (2 * pad_x) - crows) / stride_x) + 1);
  const size_t cols = (((lhs->getNumColumns() + (2 * pad_y) - ccols) / stride_y) + 1);

  // Matrix-Matrix convolution result dimensions must be strictly non-negative
  ASSERT(((int)rows > 0 || (int)cols > 0), "Matrix-Matrix convolution dimensions invalid");

  // Pad left matrix with the required padding
  lhs->pad(pad_x, pad_y, mp_arr[0]);

  // Get result matrix from pool
  MemoryManager::MatrixPool(rows, cols, result);

  // Fill the elements of result matrix
  for (size_t i{}; i < rows; ++i) {
    for (size_t j{}; j < cols; ++j) {

      // Reset to zero for the recurring matrices (#1 - #4)
      for (size_t k{1}; k <= 4; ++k) {
        ResetZero(mp_arr[k]);
      }

      // Get block matrix
      mp_arr[0]->getBlockMat({(i * stride_x), (i * stride_x) + crows - 1},
                             {(j * stride_y), (j * stride_y) + ccols - 1},
                             mp_arr[1]);

      // Hadamard product
      MatrixHadamard(mp_arr[1], rhs, mp_arr[2]);

      // Sigma function
      MatrixMul(Ones(1, crows), mp_arr[2], mp_arr[3]);
      MatrixMul(mp_arr[3], Ones(ccols, 1), mp_arr[4]);

      // Set block matrix
      result->setBlockMat({i, i}, {j, j}, mp_arr[4]);
    }
  }

  // Free resources
  std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [](Matrix<Type>* m) {
    if (nullptr != m) {
      m->free();
    }
  });
}
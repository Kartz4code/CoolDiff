/**
 * @file src/Matrix/MatrixHandler/MatInverse/MatInverseEigenHandler.cpp
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

#include "MatInverseEigenHandler.hpp"
#include "Matrix.hpp"

#include <eigen3/Eigen/Dense>

#if defined(USE_COMPLEX_MATH)
  #if COOLDIFF_SCALAR_TYPE == 2
    using EigenMatrix = Eigen::MatrixXcd;
  #elif COOLDIFF_SCALAR_TYPE == 1
    using EigenMatrix = Eigen::MatrixXcf;
  #else
    ASSERT(false, "Unknown type");
  #endif
#else
  #if COOLDIFF_SCALAR_TYPE == 2
    using EigenMatrix = Eigen::MatrixXd;
  #elif COOLDIFF_SCALAR_TYPE == 1
    using EigenMatrix = Eigen::MatrixXf;
  #else
    ASSERT(false, "Unknown type");
  #endif
#endif

void MatInverseEigenHandler::handle(const Matrix<Type>* mat, Matrix<Type>*& result) {
  const size_t nrows{mat->getNumRows()};
  const size_t ncols{mat->getNumColumns()};

  // Pool matrix
  MemoryManager::MatrixPool(nrows, ncols, result);

  Type* res_ptr = result->getMatrixPtr();
  Type* mat_ptr = const_cast<Matrix<Type>*>(mat)->getMatrixPtr();

  // Eigen inverse
  const Eigen::Map<EigenMatrix> A(mat_ptr, nrows, ncols);
  const auto inv_A = A.inverse();

  // Store result
  Eigen::Map<EigenMatrix>(res_ptr, nrows, ncols) = inv_A;
}
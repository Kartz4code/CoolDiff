/**
 * @file src/Matrix/MatrixHandler/MatrixDetHandler/EyeMatDetHandler.cpp
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

#include "EyeMatDetHandler.hpp"
#include "Matrix.hpp"
#include "MatrixEyeOps.hpp"

void EyeMatDetHandler::handle(const Matrix<Type>* mat, Matrix<Type>*& result) {
#if defined(NAIVE_IMPL)
  /* Zero matrix special check */
  if (true == IsEyeMatrix(mat)) {
    // Result matrix is transposed identity matrix
    result = MemoryManager::MatrixSplPool(1, 1, MatrixSpl::EYE);
    return;
  }
#endif
  // Chain of responsibility
  MatrixHandler::handle(mat, result);
}
/**
 * @file src/Matrix/MatrixHandler/MatrixMulHandler/DynamicHandlers/EyeMatScalarMulHandler.cpp
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

#include "EyeMatScalarMulHandler.hpp"
#include "Matrix.hpp"
#include "MatrixEyeOps.hpp"


void EyeMatScalarMulHandler::handle(Type lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
  #if defined(NAIVE_IMPL)
    /* Eye matrix special check */
    if (auto *it = EyeMatScalarMul(lhs, rhs); nullptr != it) {
      MulEye(lhs, rhs, result);
      return;
    }

  /* Eye matrix numerical check */
    #if defined(NUMERICAL_CHECK)
      else if (auto *it = EyeMatScalarMulNum(lhs, rhs); nullptr != it) {
        MulEye(lhs, rhs, result);
        return;
      }
    #endif
  #endif

  // Chain of responsibility
  MatrixHandler::handle(lhs, rhs, result);
}
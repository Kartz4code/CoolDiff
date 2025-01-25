/**
 * @file src/Matrix/MatrixHandler/MatrixAddHandler/EyeMatAddHandler.cpp
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

#include "EyeMatAddHandler.hpp"
#include "Matrix.hpp"
#include "MatrixEyeOps.hpp"

void EyeMatAddHandler::handle(const Matrix<Type> *lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
  // LHS and RHS dimensions
  const size_t nrows{lhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};
  const size_t lcols{lhs->getNumColumns()};
  const size_t rrows{rhs->getNumRows()};

  // Assert dimensions
  ASSERT((nrows == rrows) && (ncols == lcols),
         "Matrix addition dimensions mismatch");

  #if defined(NAIVE_IMPL)
      /* Eye matrix special check */
      if (auto *it = EyeMatAdd(lhs, rhs); nullptr != it) {
        if (it == lhs || it == rhs) {
          if (-1 == it->getMatType()) {
            AddEye(it, result);
          } else {
            MatrixHandler::handle(lhs, rhs, result);
          }
        } else {
          Add2Eye(it, result);
        }
        return;
      }

    /* Eye matrix numerical check */
    #if defined(NUMERICAL_CHECK)
      else if (auto *it = EyeMatAddNum(lhs, rhs); nullptr != it) {
        if (it == lhs || it == rhs) {
          if (-1 == it->getMatType()) {
            AddEye(it, result);
          } else {
            MatrixHandler::handle(lhs, rhs, result);
          }
        } else {
          Add2Eye(it, result);
        }
        return;
      }
    #endif
  #endif

  // Chain of responsibility
  MatrixHandler::handle(lhs, rhs, result);
}
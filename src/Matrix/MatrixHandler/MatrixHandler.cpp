/**
 * @file src/Matrix/MatrixHandler/MatrixHandler.cpp
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

#include "MatrixHandler.hpp"

void MatrixHandler::handle(const Matrix<Type> *lhs, const Matrix<Type> *rhs,
                           Matrix<Type> *&result) {
  ASSERT(this != mp_handler, "The handle is recursive");
  if (nullptr != mp_handler) {
    mp_handler->handle(lhs, rhs, result);
  }
}


void MatrixHandler::handle(Type lhs, const Matrix<Type> * rhs, Matrix<Type> *& result) {
  ASSERT(this != mp_handler, "The handle is recursive");
  if (nullptr != mp_handler) {
    mp_handler->handle(lhs, rhs, result);
  }
}
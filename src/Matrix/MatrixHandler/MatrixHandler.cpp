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

void MatrixHandler::handle(Type lhs, const Matrix<Type> *rhs,
                           Matrix<Type> *&result) {
  ASSERT(this != mp_handler, "The handle is recursive");
  if (nullptr != mp_handler) {
    mp_handler->handle(lhs, rhs, result);
  }
}

void MatrixHandler::handle(const Matrix<Type> *mat, Matrix<Type> *&result) {
  ASSERT(this != mp_handler, "The handle is recursive");
  if (nullptr != mp_handler) {
    mp_handler->handle(mat, result);
  }
}

void MatrixHandler::handle(const size_t nrows_f, const size_t ncols_f,
                           const size_t nrows_x, const size_t ncols_x,
                           const Matrix<Type> *mat, Matrix<Type> *&result) {
  ASSERT(this != mp_handler, "The handle is recursive");
  if (nullptr != mp_handler) {
    mp_handler->handle(nrows_f, ncols_f, nrows_x, ncols_x, mat, result);
  }
}

void MatrixHandler::handle(const size_t stride_x, const size_t stride_y,
                           const size_t pad_x, const size_t pad_y,
                           const Matrix<Type> *left, const Matrix<Type> *right,
                           Matrix<Type> *&result) {
  ASSERT(this != mp_handler, "The handle is recursive");
  if (nullptr != mp_handler) {
    mp_handler->handle(stride_x, stride_y, pad_x, pad_y, left, right, result);
  }
}
/**
 * @file include/Matrix/MatrixHandler/MatrixStaticHandler.hpp
 *
 * @copyright 2023-2026 Karthik Murali Madhavan Rathai
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

#pragma once

#include "CommonHeader.hpp"

#define THREAD_SIZE 32

class MatrixStaticHandler {
  public:
    void handle(const Matrix<Type>*, const Matrix<Type>*, Matrix<Type>*&) {
      ASSERT(false, "Invalid Handler - Entered base case");
    }

    void handle(Type, const Matrix<Type>*, Matrix<Type>*&) {
      ASSERT(false, "Invalid Handler - Entered base case");
    }

    void handle(const Matrix<Type>*, Matrix<Type>*&) {
      ASSERT(false, "Invalid Handler - Entered base case");
    }

    void handle(const size_t, const size_t, const size_t, const size_t, const Matrix<Type>*, Matrix<Type>*&) {
      ASSERT(false, "Invalid Handler - Entered base case");
    }

    void handle(const Matrix<Type>*, const FunctionType&, Matrix<Type>*&) {
      ASSERT(false, "Invalid Handler - Entered base case");
    }

    void handle(const size_t, const size_t, const size_t, const size_t, const Matrix<Type>*, const Matrix<Type>*, Matrix<Type>*&) {
      ASSERT(false, "Invalid Handler - Entered base case");
    }

    void handle(const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const Matrix<Type>*, const Matrix<Type>*, 
                const Matrix<Type>*, const Matrix<Type>*, Matrix<Type>*&) {
      ASSERT(false, "Invalid Handler - Entered base case");
    }
};
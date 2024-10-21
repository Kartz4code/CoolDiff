/**
 * @file include/Matrix/MatrixHandler/MatrixHandler.hpp
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

#pragma once

#include "CommonHeader.hpp"

// Matrix handler for different types
class MatrixHandler {
private:
  MatrixHandler *mp_handler{nullptr};

public:
  // Constructor
  constexpr MatrixHandler(MatrixHandler *h = nullptr) : mp_handler{h} {}

  V_UNPURE(void handle(const Matrix<Type> *, const Matrix<Type> *,
                       Matrix<Type> *&));

  V_UNPURE(void handle(Type, const Matrix<Type> *, Matrix<Type> *&));

  V_UNPURE(void handle(const Matrix<Type> *, Matrix<Type> *&));

  V_UNPURE(void handle(const size_t, const size_t, 
                       const size_t, const size_t,
                       const Matrix<Type> *, Matrix<Type> *&));

  V_UNPURE(void handle(const size_t, const size_t, 
                       const size_t, const size_t,
                       const Matrix<Type> *, const Matrix<Type> *,
                       Matrix<Type> *&));

  V_UNPURE(void handle(const size_t, const size_t, 
                       const size_t, const size_t,
                       const size_t, const size_t, 
                       const Matrix<Type> *, const Matrix<Type> *, 
                       const Matrix<Type> *, const Matrix<Type> *, 
                       Matrix<Type> *&));

  // Destructor
  V_DTR(~MatrixHandler() = default);
};

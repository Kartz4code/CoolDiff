/**
 * @file
 * include/Matrix/MatrixHandler/MatrixConvolution/MatDervConvNaiveHandler.hpp
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

#include "MatrixHandler.hpp"

class MatDervConvNaiveHandler : public MatrixHandler {
private:
  // All matrices
  inline static constexpr const size_t m_size{13};
  Matrix<Type> *mp_arr[m_size]{};

  // Boolean check for initialization
  bool m_initialized{false};

public:
  using MatrixHandler::MatrixHandler;

  V_OVERRIDE(void handle(const size_t, const size_t, const size_t, const size_t,
                         const size_t, const size_t, const Matrix<Type> *,
                         const Matrix<Type> *, const Matrix<Type> *,
                         const Matrix<Type> *, Matrix<Type> *&));

  // Destructor
  V_DTR(~MatDervConvNaiveHandler() = default);
};
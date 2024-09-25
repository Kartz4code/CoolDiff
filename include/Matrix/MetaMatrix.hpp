/**
 * @file include/Matrix/MetaMatrix.hpp
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

class MetaMatrix {
protected:
  // Index counter (A counter to count the number of matrix operations)
  inline static size_t m_idx_count{0};

public:
  // Visited flag
  bool m_visited{false};

  // Default constructor
  MetaMatrix() = default;

  // Evaluate run-time
  V_PURE(Matrix<Type> *eval());

  // Forward derivative
  V_PURE(Matrix<Type> *devalF(Matrix<Variable> &));
  
  // Get number of rows and columns
  V_PURE(size_t getNumRows() const);

  V_PURE(size_t getNumColumns() const);

  // Reset all visited flags
  V_PURE(void reset());

  // Get type
  V_PURE(std::string_view getType() const);

  // Destructor
  V_DTR(~MetaMatrix()) = default;
};
/**
 * @file include/Matrix/IMatrix.hpp
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

#include "MatOperators.hpp"
#include "MemoryManager.hpp"

// IVariable class to enforce expression templates for lazy evaluation
template <typename T>
class IMatrix : public MetaMatrix {
 private:
  inline constexpr const T &derived() const {
    return static_cast<const T &>(*this);
  }

  inline constexpr T &derived() { return static_cast<T &>(*this); }

 protected:
  // Protected constructor
  IMatrix() = default;

 public:
  /* Type implementations */
  // Get number of rows and columns
  size_t getNumRows() const { return derived().getNumRows(); }

  size_t getNumColumns() const { return derived().getNumColumns(); }

  // Find me
  bool findMe(void *v) const { return derived().findMe(v); }

  // Protected destructor
  V_DTR(~IMatrix()) = default;
};

// Special matrices
enum MatrixSpl : size_t {
  ZEROS = 1 << 1,
  EYE = 1 << 2,
  ONES = 1 << 3,
  DIAG = 1 << 4,
  ROW_MAT = 1 << 5,
  COL_MAT = 1 << 6
};

// Operation type (Order matters!)
#define OpMatType void (*)(Type *, Type *, Type *, size_t, size_t)
// Operation objects (Order matters!)
#define OpMatObj MatrixAddition
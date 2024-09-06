/**
 * @file src/Matrix/MatrixBasics.cpp
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

#include "MatrixBasics.hpp"
#include "Matrix.hpp"

// Numerical Eye matrix
Matrix<Type> *Eye(const size_t n) {

  // Eye matrix registry
  static UnOrderedMap<size_t, Matrix<Type> *> eye_register;

  // Find in registry of special matrices
  if (auto it = eye_register.find(n); it != eye_register.end()) {
    return it->second;
  } else {
    Matrix<Type> *result = CreateMatrixPtr<Type>(n, n);

    // Vector of indices
    auto idx = Range<size_t>(0, n);
    std::for_each(EXECUTION_PAR idx.begin(), idx.end(),
                  [&](const size_t i) { (*result)(i, i) = (Type)(1); });

    // Register and return result
    eye_register[n] = result;
    return result;
  }
}
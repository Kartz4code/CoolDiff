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
const Matrix<Type> *Eye(const size_t n) {
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

// Numerical Zero matrix
const Matrix<Type> *Zeros(const size_t n, const size_t m) {
  // Zeros matrix registry
  static UnOrderedMap<Pair<size_t, size_t>, Matrix<Type> *> zeros_register;

  // Find in registry of special matrices
  if (auto it = zeros_register.find({n, m}); it != zeros_register.end()) {
    return it->second;
  } else {
    Matrix<Type> *result = CreateMatrixPtr<Type>(n, m);
    // Register and return result
    zeros_register[{n, m}] = result;
    return result;
  }
}

// Numerical Zeros square matrix
const Matrix<Type> *Zeros(const size_t n) { return Zeros(n, n); }

// Numerical Ones matrix
const Matrix<Type> *Ones(const size_t n, const size_t m) {
  // Zeros matrix registry
  static UnOrderedMap<Pair<size_t, size_t>, Matrix<Type> *> ones_register;

  // Find in registry of special matrices
  if (auto it = ones_register.find({n, m}); it != ones_register.end()) {
    return it->second;
  } else {
    // Result matrix
    Matrix<Type> *result = CreateMatrixPtr<Type>(n, m, (Type)(1));
    // Register and return result
    ones_register[{n, m}] = result;
    return result;
  }
}

// Numerical Ones square matrix
const Matrix<Type> *Ones(const size_t n) { return Ones(n, n); }
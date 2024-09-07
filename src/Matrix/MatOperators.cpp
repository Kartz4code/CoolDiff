/**
 * @file src/Matrix/MatOperators.cpp
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

#include "MatOperators.hpp"
#include "Matrix.hpp"

// Special matrix addition
#include "EyeMatAddHandler.hpp"
#include "ZeroMatAddHandler.hpp"

// Special matrix subtraction
#include "EyeMatSubHandler.hpp"
#include "ZeroMatSubHandler.hpp"

// Special matrix multiplication
#include "EyeMatMulHandler.hpp"
#include "ZeroMatMulHandler.hpp"

// Special matrix Kronocker product
#include "EyeMatKronHandler.hpp"
#include "ZeroMatKronHandler.hpp"

// Special matrix Hadamard product
#include "EyeMatHadamardHandler.hpp"
#include "ZeroMatHadamardHandler.hpp"

// Matrix operations
#include "MatAddNaiveHandler.hpp"
#include "MatSubNaiveHandler.hpp"
#include "MatKronNaiveHandler.hpp"
#include "MatMulNaiveHandler.hpp"
#include "MatHadamardNaiveHandler.hpp"

// Matrix-Matrix addition - Left, Right, Result matrix pointer
void MatrixAdd(const Matrix<Type> *lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix-Matrix addition
  */

  MatAddNaiveHandler h1{nullptr};
  ZeroMatAddHandler h2{&h1};
  EyeMatAddHandler h3{&h2};

  // Handle matrix addition
  h3.handle(lhs, rhs, result);
}


// Matrix-Matrix subtraction - Left, Right, Result matrix pointer
void MatrixSub(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
   // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix-Matrix subtraction
  */

  MatSubNaiveHandler h1{nullptr};
  ZeroMatSubHandler h2{&h1};
  EyeMatSubHandler h3{&h2};

  // Handle matrix subtraction
  h3.handle(lhs, rhs, result);
}

// Matrix-Matrix multiplication - Left, Right, Result matrix pointer
void MatrixMul(const Matrix<Type> *lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix-Matrix multiplication
  */
  MatMulNaiveHandler h1{nullptr};
  ZeroMatMulHandler h2{&h1};
  EyeMatMulHandler h3{&h2};

  // Handle matrix multiplication
  h3.handle(lhs, rhs, result);
}

// Matrix-Matrix Kronocker product - Left, Right, Result matrix pointer
void MatrixKron(const Matrix<Type> *lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix-Matrix Kronocker product
  */

  MatKronNaiveHandler h1{nullptr};
  ZeroMatKronHandler h2{&h1};
  EyeMatKronHandler h3{&h2};

  // Handle Kronocker product
  h3.handle(lhs, rhs, result);
}

// Matrix-Matrix Hadamard product - Left, Right, Result matrix pointer
void MatrixHadamard(const Matrix<Type> * lhs, const Matrix<Type> * rhs, Matrix<Type> *& result) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix-Matrix Hadamard product
  */

  MatHadamardNaiveHandler h1{nullptr};
  ZeroMatHadamardHandler h2{&h1};
  EyeMatHadamardHandler h3{&h2};

  // Handle Hadamard product
  h3.handle(lhs, rhs, result);
}

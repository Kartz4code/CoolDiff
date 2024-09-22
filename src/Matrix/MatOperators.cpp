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
#include "ZeroMatScalarAddHandler.hpp"
#include "EyeMatScalarAddHandler.hpp"

// Special matrix subtraction
#include "EyeMatSubHandler.hpp"
#include "ZeroMatSubHandler.hpp"

// Special matrix multiplication
#include "EyeMatMulHandler.hpp"
#include "ZeroMatMulHandler.hpp"
#include "ZeroMatScalarMulHandler.hpp"
#include "EyeMatScalarMulHandler.hpp"

// Special matrix Kronocker product
#include "EyeMatKronHandler.hpp"
#include "ZeroMatKronHandler.hpp"

// Special matrix Hadamard product
#include "EyeMatHadamardHandler.hpp"
#include "ZeroMatHadamardHandler.hpp"

// Special matrix transpose product
#include "EyeMatTransposeHandler.hpp"
#include "ZeroMatTransposeHandler.hpp"
#include "EyeMatDervTransposeHandler.hpp"
#include "ZeroMatDervTransposeHandler.hpp"


// Matrix operations
#include "MatAddNaiveHandler.hpp"
#include "MatScalarAddNaiveHandler.hpp"
#include "MatHadamardNaiveHandler.hpp"
#include "MatKronNaiveHandler.hpp"
#include "MatMulNaiveHandler.hpp"
#include "MatScalarMulNaiveHandler.hpp"
#include "MatSubNaiveHandler.hpp"
#include "MatTransposeNaiveHandler.hpp"
#include "MatDervTransposeNaiveHandler.hpp"

// Matrix-Matrix addition - Left, Right, Result matrix pointer
void MatrixAdd(const Matrix<Type> *lhs, const Matrix<Type> *rhs,
               Matrix<Type> *&result) {
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
void MatrixSub(const Matrix<Type> *lhs, const Matrix<Type> *rhs,
               Matrix<Type> *&result) {
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
void MatrixMul(const Matrix<Type> *lhs, const Matrix<Type> *rhs,
               Matrix<Type> *&result) {
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
void MatrixKron(const Matrix<Type> *lhs, const Matrix<Type> *rhs,
                Matrix<Type> *&result) {
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
void MatrixHadamard(const Matrix<Type> *lhs, const Matrix<Type> *rhs,
                    Matrix<Type> *&result) {
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


// Matrix-Scalar addition
void MatrixScalarAdd(Type lhs, const Matrix<Type> * rhs, Matrix<Type> *& result) {
  // Null pointer check
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix-Matrix Hadamard product
  */

  MatScalarAddNaiveHandler h1{nullptr};
  ZeroMatScalarAddHandler h2{&h1};
  EyeMatScalarAddHandler h3{&h2};

  // Handle Matrix-Scalar addition
  h3.handle(lhs, rhs, result);
}


// Matrix-Scalar multiplication
void MatrixScalarMul(Type lhs, const Matrix<Type> * rhs, Matrix<Type> *& result) {
  // Null pointer check
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

   /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix-Matrix Hadamard product
  */

  MatScalarMulNaiveHandler h1{nullptr};
  ZeroMatScalarMulHandler h2{&h1};
  EyeMatScalarMulHandler h3{&h2};

  // Handle Matrix-Scalar multiplication
  h3.handle(lhs, rhs, result);
}


// Matrix transpose
void MatrixTranspose(const Matrix<Type> * mat, Matrix<Type> *& result) {
  // Null pointer check
  NULL_CHECK(mat, "Matrix (mat) is a nullptr");

   /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix transpose
  */

  MatTransposeNaiveHandler h1{nullptr};
  ZeroMatTransposeHandler h2{&h1};
  EyeMatTransposeHandler h3{&h2};

  // Handle Matrix transpose
  h3.handle(mat, result);
}

// Matrix derivative transpose
void MatrixDervTranspose(const size_t nrows_f, const size_t ncols_f, const size_t nrows_x, const size_t ncols_x, 
                         const Matrix<Type> * mat, Matrix<Type>*& result) {
    // Null pointer check
  NULL_CHECK(mat, "Matrix (mat) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix transpose
  */

  MatDervTransposeNaiveHandler h1{nullptr};
  ZeroMatDervTransposeHandler h2{&h1};                        
  EyeMatDervTransposeHandler h3{&h2};

  // Handle Matrix transpose
  h3.handle(nrows_f, ncols_f, nrows_x, ncols_x, mat, result);
}
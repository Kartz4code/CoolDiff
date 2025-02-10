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



// Special matrix Hadamard product
#include "EyeMatHadamardHandler.hpp"
#include "ZeroMatHadamardHandler.hpp"

// Special matrix transpose product
#include "EyeMatDervTransposeHandler.hpp"
#include "EyeMatTransposeHandler.hpp"
#include "ZeroMatDervTransposeHandler.hpp"
#include "ZeroMatTransposeHandler.hpp"

// Special matrix convolution
#include "ZeroMatConvHandler.hpp"
#include "ZeroMatDervConvHandler.hpp"

// Special matrix unary
#include "EyeMatUnaryHandler.hpp"
#include "ZeroMatUnaryHandler.hpp"

// Matrix operations
#include "MatConvNaiveHandler.hpp"
#include "MatDervConvNaiveHandler.hpp"
#include "MatDervTransposeNaiveHandler.hpp"
#include "MatHadamardNaiveHandler.hpp"

#include "MatTransposeNaiveHandler.hpp"
#include "MatUnaryNaiveHandler.hpp"

#include "MatInverseEigenHandler.hpp"
#include "EyeMatInvHandler.hpp"
#include "ZeroMatInvHandler.hpp"



// Static handlers
// Matrix-Matrix addition
#include "EyeMatAddHandler.hpp"
#include "ZeroMatAddHandler.hpp"
#include "MatAddNaiveHandler.hpp"

// Matrix-scalar addition
#include "EyeMatScalarAddHandler.hpp"
#include "ZeroMatScalarAddHandler.hpp"
#include "MatScalarAddNaiveHandler.hpp"

// Matrix-Matrix multiplication
#include "EyeMatMulHandler.hpp"
#include "ZeroMatMulHandler.hpp"
#include "MatMulNaiveHandler.hpp"

// Matrix-scalar multiplication
#include "EyeMatScalarMulHandler.hpp"
#include "ZeroMatScalarMulHandler.hpp"
#include "MatScalarMulNaiveHandler.hpp"

// Matrix-Matrix subtraction
#include "EyeMatSubHandler.hpp"
#include "ZeroMatSubHandler.hpp"
#include "MatSubNaiveHandler.hpp"

// Matrix-Matrix Kronocker product
#include "EyeMatKronHandler.hpp"
#include "ZeroMatKronHandler.hpp"
#include "MatKronNaiveHandler.hpp"

// Matrix determinant product
#include "MatDetEigenHandler.hpp"
#include "EyeMatDetHandler.hpp"
#include "ZeroMatDetHandler.hpp"

// Handler of sizes 2,3,4,5,6,7
#define HANDLER2(X1,X2) X1<X2<MatrixStaticHandler>>
#define HANDLER3(X1,X2,X3) X1<X2<X3<MatrixStaticHandler>>>
#define HANDLER4(X1,X2,X3,X4) X1<X2<X3<X4<MatrixStaticHandler>>>>
#define HANDLER5(X1,X2,X3,X4,X5) X1<X2<X3<X4<X5<MatrixStaticHandler>>>>>
#define HANDLER6(X1,X2,X3,X4,X5,X6) X1<X2<X3<X4<X5<X6<MatrixStaticHandler>>>>>>
#define HANDLER7(X1,X2,X3,X4,X5,X6,X7) X1<X2<X3<X4<X5<X6<X7<MatrixStaticHandler>>>>>>>

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

  static HANDLER3(EyeMatAddHandler, 
                  ZeroMatAddHandler,
                  MatAddNaiveHandler) handler;

  // Handle matrix addition
  handler.handle(lhs, rhs, result);
}

// Matrix-Scalar addition
void MatrixScalarAdd(Type lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
  // Null pointer check
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix-Matrix Hadamard product
  */

  static HANDLER3(EyeMatScalarAddHandler, 
                  ZeroMatScalarAddHandler,
                  MatScalarAddNaiveHandler) handler;

  // Handle Matrix-Scalar addition
  handler.handle(lhs, rhs, result);
}

// Matrix-Matrix subtraction - Left, Right, Result matrix pointer
void MatrixSub(const Matrix<Type> *lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix-Matrix subtraction
  */

  static HANDLER3(EyeMatSubHandler, 
                  ZeroMatSubHandler,
                  MatSubNaiveHandler) handler;

  // Handle matrix addition
  handler.handle(lhs, rhs, result);
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

  static HANDLER3(EyeMatMulHandler,
                  ZeroMatMulHandler,
                  MatMulNaiveHandler) handler;

  // Handle matrix multiplication
  handler.handle(lhs, rhs, result);
}

// Matrix-Scalar multiplication
void MatrixScalarMul(Type lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
  // Null pointer check
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
     1) Eye matrix check
     2) Zero matrix check
     3) Matrix-Matrix Hadamard product
 */

  static HANDLER3(EyeMatScalarMulHandler,
                  ZeroMatScalarMulHandler,
                  MatScalarMulNaiveHandler) handler;

  // Handle Matrix-Scalar multiplication
  handler.handle(lhs, rhs, result);
}

// Matrix-Matrix Kronocker product - Left, Right, Result matrix pointer
void MatrixKron(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix-Matrix Kronocker product
  */

  static HANDLER3(EyeMatKronHandler, 
                  ZeroMatKronHandler,
                  MatKronNaiveHandler) handler;

  // Handle Kronocker product
  handler.handle(lhs, rhs, result);
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

  static MatHadamardNaiveHandler h1{nullptr};
  static ZeroMatHadamardHandler h2{&h1};
  static EyeMatHadamardHandler h3{&h2};

  // Handle Hadamard product
  h3.handle(lhs, rhs, result);
}

// Matrix transpose
void MatrixTranspose(const Matrix<Type> *mat, Matrix<Type> *&result) {
  // Null pointer check
  NULL_CHECK(mat, "Matrix (mat) is a nullptr");

  /* Chain of responsibility (Order matters)
     1) Eye matrix check
     2) Zero matrix check
     3) Matrix transpose
 */

  static MatTransposeNaiveHandler h1{nullptr};
  static ZeroMatTransposeHandler h2{&h1};
  static EyeMatTransposeHandler h3{&h2};

  // Handle Matrix transpose
  h3.handle(mat, result);
}

// Matrix derivative transpose
void MatrixDervTranspose(const size_t nrows_f, const size_t ncols_f,
                         const size_t nrows_x, const size_t ncols_x,
                         const Matrix<Type> *mat, Matrix<Type> *&result) {
  // Null pointer check
  NULL_CHECK(mat, "Matrix (mat) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix transpose
  */

  static MatDervTransposeNaiveHandler h1{nullptr};
  static ZeroMatDervTransposeHandler h2{&h1};
  static EyeMatDervTransposeHandler h3{&h2};

  // Handle Matrix transpose
  h3.handle(nrows_f, ncols_f, nrows_x, ncols_x, mat, result);
}

// Matrix convolution
void MatrixConv(const size_t stride_x, const size_t stride_y,
                const size_t pad_x, const size_t pad_y, const Matrix<Type> *lhs,
                const Matrix<Type> *rhs, Matrix<Type> *&result) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix convolution
  */
  static MatConvNaiveHandler h1{nullptr};
  static ZeroMatConvHandler h2{&h1};

  // Handle Matrix convolution
  h2.handle(stride_x, stride_y, pad_x, pad_y, lhs, rhs, result);
}

// Matrix derivative convolution
void MatrixDervConv(const size_t nrows_x, const size_t ncols_x,
                    const size_t stride_x, const size_t stride_y,
                    const size_t pad_x, const size_t pad_y,
                    const Matrix<Type> *lhs, const Matrix<Type> *dlhs,
                    const Matrix<Type> *rhs, const Matrix<Type> *drhs,
                    Matrix<Type> *&result) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  NULL_CHECK(dlhs, "LHS Derivative Matrix (lhs) is a nullptr");
  NULL_CHECK(drhs, "RHS Derivative Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Eye matrix check
      2) Zero matrix check
      3) Matrix convolution derivative
  */

  static MatDervConvNaiveHandler h1{nullptr};
  static ZeroMatDervConvHandler h2{&h1};

  // Handle Matrix convolution derivative
  h2.handle(nrows_x, ncols_x, stride_x, stride_y, pad_x, pad_y, lhs, dlhs, rhs,
            drhs, result);
}

// Matrix unary
void MatrixUnary(const Matrix<Type> *mat, const FunctionType1 &func, Matrix<Type> *&result) {
  NULL_CHECK(mat, "Matrix mat is a nullptr");

  static MatUnaryNaiveHandler h1{nullptr};
  static EyeMatUnaryHandler h2{&h1};
  static ZeroMatUnaryHandler h3{&h2};

  // Handle Unary Matrix
  h3.handle(mat, func, result);
}

// MatrixInverse
void MatrixInverse(const Matrix<Type>* mat, Matrix<Type>*& result) {
  NULL_CHECK(mat, "Matrix mat is a nullptr");

  static MatInverseEigenHandler h1{nullptr};
  static EyeMatInvHandler h2{&h1};
  static ZeroMatInvHandler h3{&h2};

  // Handle Matrix Inverse
  h3.handle(mat, result);
}

// MatrixDeterminant
void MatrixDet(const Matrix<Type>* mat, Matrix<Type>*& result) {
  // Null pointer check
  NULL_CHECK(mat, "Matrix mat is a nullptr");

  /* Chain of responsibility (Order matters)
    1) Eye matrix check
    2) Zero matrix check
    3) Matrix convolution derivative
  */

  static HANDLER3(EyeMatDetHandler, 
                  ZeroMatDetHandler,
                  MatDetEigenHandler) handler;

  // Handle matrix determinant
  handler.handle(mat, result);
}
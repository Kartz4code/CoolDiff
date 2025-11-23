/**
 * @file src/Matrix/MatOperators.cpp
 *
 * @copyright 2023-2025 Karthik Murali Madhavan Rathai
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

// Static handlers
// Matrix-Matrix addition
#include "MatAddNaiveHandler.hpp"

// Eigen implementation
#include "MatAddEigenHandler.hpp"

// Matrix-scalar addition
#include "MatScalarAddNaiveHandler.hpp"

// Eigen implementation
#include "MatScalarAddEigenHandler.hpp"

// Matrix-Matrix multiplication
#include "MatMulNaiveHandler.hpp"

// Eigen implementation
#include "MatMulEigenHandler.hpp"

// Matrix-scalar multiplication
#include "MatScalarMulNaiveHandler.hpp"

// Eigen implementation
#include "MatScalarMulEigenHandler.hpp"

// Matrix-Matrix subtraction
#include "MatSubNaiveHandler.hpp"

// Eigen implementation
#include "MatSubEigenHandler.hpp"

// Matrix-Matrix Kronocker product
#include "MatKronNaiveHandler.hpp"

// Eigen implementation
#include "MatKronEigenHandler.hpp"

// Matrix determinant
#include "MatDetEigenHandler.hpp"

// Matrix trace
#include "MatTraceNaiveHandler.hpp"

// Eigen trace
#include "MatTraceEigenHandler.hpp" 

// Matrix Hadamard product
#include "MatHadamardNaiveHandler.hpp"

// Eigen implementation
#include "MatHadamardEigenHandler.hpp"

// Matrix transpose
#include "MatDervTransposeNaiveHandler.hpp"
#include "MatTransposeNaiveHandler.hpp"

// Eigen implementation
#include "MatTransposeEigenHandler.hpp"

// Matrix unary
#include "MatUnaryNaiveHandler.hpp"

// Eigen implementation
#include "MatUnaryEigenHandler.hpp"

// Matrix convolution
#include "MatConvNaiveHandler.hpp"
#include "MatConvEigenHandler.hpp"
#include "MatDervConvNaiveHandler.hpp"

// Matrix inverse
#include "MatInverseEigenHandler.hpp"

// Handler of sizes {1,2,3,4,5,6,7}
#define HANDLER1(X) X<MatrixStaticHandler>
#define HANDLER2(X1,X2) X1<X2<MatrixStaticHandler>>
#define HANDLER3(X1,X2,X3) X1<X2<X3<MatrixStaticHandler>>>
#define HANDLER4(X1,X2,X3,X4) X1<X2<X3<X4<MatrixStaticHandler>>>>
#define HANDLER5(X1,X2,X3,X4,X5) X1<X2<X3<X4<X5<MatrixStaticHandler>>>>>
#define HANDLER6(X1,X2,X3,X4,X5,X6) X1<X2<X3<X4<X5<X6<MatrixStaticHandler>>>>>>
#define HANDLER7(X1,X2,X3,X4,X5,X6,X7) X1<X2<X3<X4<X5<X6<X7<MatrixStaticHandler>>>>>>>

namespace CoolDiff {
    namespace TensorR2 {
        namespace MatOperators {
          // Matrix-Matrix addition - Left, Right, Result matrix pointer
          void MatrixAdd(const Matrix<Type> *lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            #ifdef NAIVE_HANDLER
              static HANDLER1(MatAddNaiveHandler) handler;
            #endif

            static HANDLER1(MatAddEigenHandler) handler;

            // Handle matrix addition
            handler.handle(lhs, rhs, result);
          }

          // Matrix-Scalar addition
          void MatrixScalarAdd(Type lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
            // Null pointer check
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            #ifdef NAIVE_HANDLER
              static HANDLER1(MatScalarAddNaiveHandler) handler;
            #endif

            static HANDLER1(MatScalarAddEigenHandler) handler;
                            
            // Handle Matrix-Scalar addition
            handler.handle(lhs, rhs, result);
          }

          // Matrix-Matrix subtraction - Left, Right, Result matrix pointer
          void MatrixSub(const Matrix<Type> *lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            #ifdef NAIVE_HANDLER
              static HANDLER1(MatSubNaiveHandler) handler;
            #endif
            
            static HANDLER1(MatSubEigenHandler) handler;
            
            // Handle matrix addition
            handler.handle(lhs, rhs, result);
          }

          // Matrix-Matrix multiplication - Left, Right, Result matrix pointer
          void MatrixMul(const Matrix<Type> *lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            #ifdef NAIVE_HANDLER
              static HANDLER1(MatMulNaiveHandler) handler;
            #endif

            static HANDLER1(MatMulEigenHandler) handler;

            // Handle matrix multiplication
            handler.handle(lhs, rhs, result);
          }

          // Matrix-Scalar multiplication
          void MatrixScalarMul(Type lhs, const Matrix<Type> *rhs, Matrix<Type> *&result) {
            // Null pointer check
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            #ifdef NAIVE_HANDLER
              static HANDLER1(MatScalarMulNaiveHandler) handler;
            #endif

            static HANDLER1(MatScalarMulEigenHandler) handler;
            
            // Handle Matrix-Scalar multiplication
            handler.handle(lhs, rhs, result);
          }

          // Matrix-Matrix Kronocker product - Left, Right, Result matrix pointer
          void MatrixKron(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            #ifdef NAIVE_HANDLER
              static HANDLER1(MatKronNaiveHandler) handler;
            #endif

            static HANDLER1(MatKronEigenHandler) handler;
            
            // Handle Kronocker product
            handler.handle(lhs, rhs, result);
          }

          // Matrix-Matrix Hadamard product - Left, Right, Result matrix pointer
          void MatrixHadamard(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            #ifdef NAIVE_HANDLER
              static HANDLER1(MatHadamardNaiveHandler) handler;
            #endif
            
            static HANDLER1(MatHadamardEigenHandler) handler;
            
            // Handle Hadamard product
            handler.handle(lhs, rhs, result);
          }

          // Matrix transpose
          void MatrixTranspose(const Matrix<Type>* mat, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(mat, "Matrix (mat) is a nullptr");

            #if NAIVE_HANDLER
              static HANDLER1(MatTransposeNaiveHandler) handler;
            #endif

            static HANDLER1(MatTransposeEigenHandler) handler;
              
            // Handle Matrix transpose
            handler.handle(mat, result);
          }

          // Matrix derivative transpose
          void MatrixDervTranspose(const size_t nrows_f, const size_t ncols_f,
                                  const size_t nrows_x, const size_t ncols_x,
                                  const Matrix<Type>* mat, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(mat, "Matrix (mat) is a nullptr");

            static HANDLER1(MatDervTransposeNaiveHandler) handler;

            // Handle Matrix transpose
            handler.handle(nrows_f, ncols_f, nrows_x, ncols_x, mat, result);
          }

          // MatrixInverse
          void MatrixInverse(const Matrix<Type>* mat, Matrix<Type>*& result) {
            NULL_CHECK(mat, "Matrix mat is a nullptr");

            static HANDLER1(MatInverseEigenHandler) handler;

            // Handle Matrix Inverse
            handler.handle(mat, result);
          }

          // Matrix determinant
          void MatrixDet(const Matrix<Type>* mat, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(mat, "Matrix mat is a nullptr");

            static HANDLER1(MatDetEigenHandler) handler;

            // Handle matrix determinant
            handler.handle(mat, result);
          }

          // Matrix trace
          void MatrixTrace(const Matrix<Type>* mat, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(mat, "Matrix mat is a nullptr");

            #if NAIVE_HANDLER
              static HANDLER1(MatTraceNaiveHandler) handler;
            #endif  
            static HANDLER1(MatTraceEigenHandler) handler;

            // Handle matrix determinant
            handler.handle(mat, result);
          }

          // Matrix unary
          void MatrixUnary(const Matrix<Type>* mat, const FunctionType1& func, Matrix<Type>*& result) {
            NULL_CHECK(mat, "Matrix mat is a nullptr");

            #if NAIVE_HANDLER
              static HANDLER1(MatUnaryNaiveHandler) handler;
            #endif
            static HANDLER1(MatUnaryEigenHandler) handler;

            // Handle Unary Matrix
            handler.handle(mat, func, result);
          }

          // Matrix convolution
          void MatrixConv(const size_t stride_x, const size_t stride_y,
                          const size_t pad_x, const size_t pad_y, const Matrix<Type> *lhs,
                          const Matrix<Type> *rhs, Matrix<Type> *&result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            #if NAIVE_HANDLER
              static HANDLER1(MatConvNaiveHandler) handler;
            #endif  
            static HANDLER1(MatConvEigenHandler) handler;
            
            // Handle Matrix convolution
            handler.handle(stride_x, stride_y, pad_x, pad_y, lhs, rhs, result);
          }

          // Matrix derivative convolution
          void MatrixDervConv(const size_t nrows_x, const size_t ncols_x,
                              const size_t stride_x, const size_t stride_y,
                              const size_t pad_x, const size_t pad_y,
                              const Matrix<Type>* lhs, const Matrix<Type>* dlhs,
                              const Matrix<Type>* rhs, const Matrix<Type>* drhs,
                              Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            NULL_CHECK(dlhs, "LHS Derivative Matrix (lhs) is a nullptr");
            NULL_CHECK(drhs, "RHS Derivative Matrix (rhs) is a nullptr");

            static HANDLER1(MatDervConvNaiveHandler) handler;

            // Handle Matrix convolution derivative
            handler.handle(nrows_x, ncols_x, stride_x, stride_y, pad_x, pad_y, lhs, dlhs, rhs, drhs, result);
          }
        }
    }
}


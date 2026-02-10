/**
 * @file src/Matrix/MatOperators.cpp
 *
 * @copyright 2023-2026 Karthik Murali Madhavan Rathai
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

// Handler of sizes {1,2,3,4,5,6,7}
#define HANDLER1(X) X<MatrixStaticHandler>
#define HANDLER2(X1,X2) X1<X2<MatrixStaticHandler>>
#define HANDLER3(X1,X2,X3) X1<X2<X3<MatrixStaticHandler>>>
#define HANDLER4(X1,X2,X3,X4) X1<X2<X3<X4<MatrixStaticHandler>>>>
#define HANDLER5(X1,X2,X3,X4,X5) X1<X2<X3<X4<X5<MatrixStaticHandler>>>>>
#define HANDLER6(X1,X2,X3,X4,X5,X6) X1<X2<X3<X4<X5<X6<MatrixStaticHandler>>>>>>
#define HANDLER7(X1,X2,X3,X4,X5,X6,X7) X1<X2<X3<X4<X5<X6<X7<MatrixStaticHandler>>>>>>>

// Naive handlers
#include "MatAddNaiveHandler.hpp"
#include "MatScalarAddNaiveHandler.hpp"
#include "MatMulNaiveHandler.hpp"
#include "MatScalarMulNaiveHandler.hpp"
#include "MatSubNaiveHandler.hpp"
#include "MatKronNaiveHandler.hpp"
#include "MatTraceNaiveHandler.hpp"
#include "MatHadamardNaiveHandler.hpp"
#include "MatDervTransposeNaiveHandler.hpp"
#include "MatTransposeNaiveHandler.hpp"
#include "MatUnaryNaiveHandler.hpp"
#include "MatConvNaiveHandler.hpp"
#include "MatDervConvNaiveHandler.hpp"

// Eigen handlers
#include "MatAddEigenHandler.hpp"
#include "MatScalarAddEigenHandler.hpp"
#include "MatMulEigenHandler.hpp"
#include "MatScalarMulEigenHandler.hpp"
#include "MatSubEigenHandler.hpp"
#include "MatKronEigenHandler.hpp"
#include "MatDetEigenHandler.hpp"
#include "MatTraceEigenHandler.hpp" 
#include "MatHadamardEigenHandler.hpp"
#include "MatTransposeEigenHandler.hpp"
#include "MatUnaryEigenHandler.hpp"
#include "MatConvEigenHandler.hpp"
#include "MatInverseEigenHandler.hpp"

// CUDA handlers
#include "MatAddCUDAHandler.hpp"
#include "MatScalarAddCUDAHandler.hpp"
#include "MatSubCUDAHandler.hpp"
#include "MatHadamardCUDAHandler.hpp"
#include "MatMulCUDAHandler.hpp"
#include "MatScalarMulCUDAHandler.hpp"
#include "MatKronCUDAHandler.hpp"
#include "MatTransposeCUDAHandler.hpp"
#include "MatUnaryCUDAHandler.hpp"

namespace CoolDiff {
    namespace TensorR2 {
        namespace MatOperators {
          // Matrix-Matrix addition - Left, Right, Result matrix pointer
          void MatrixAdd(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            static HANDLER3(MatAddEigenHandler, MatAddCUDAHandler, MatAddNaiveHandler) handler;
            handler.handle(lhs, rhs, result);        
          }

          // Matrix-Scalar addition
          void MatrixScalarAdd(Type lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            static HANDLER3(MatScalarAddEigenHandler, MatScalarAddCUDAHandler, MatScalarAddNaiveHandler) handler;
            handler.handle(lhs, rhs, result);
          }

          // Matrix-Matrix subtraction - Left, Right, Result matrix pointer
          void MatrixSub(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            static HANDLER3(MatSubEigenHandler, MatSubCUDAHandler, MatSubNaiveHandler) handler;
            handler.handle(lhs, rhs, result);
          }

          // Matrix-Matrix multiplication - Left, Right, Result matrix pointer
          void MatrixMul(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            static HANDLER3(MatMulEigenHandler, MatMulCUDAHandler, MatMulNaiveHandler) handler; 
            handler.handle(lhs, rhs, result);
          }

          // Matrix-Scalar multiplication
          void MatrixScalarMul(Type lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            static HANDLER3(MatScalarMulEigenHandler, MatScalarMulCUDAHandler, MatScalarMulNaiveHandler) handler; 
            handler.handle(lhs, rhs, result);
          }

          // Matrix-Matrix Kronocker product - Left, Right, Result matrix pointer
          void MatrixKron(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            static HANDLER3(MatKronEigenHandler, MatKronCUDAHandler, MatKronNaiveHandler) handler; 
            handler.handle(lhs, rhs, result);
          }

          // Matrix-Matrix Hadamard product - Left, Right, Result matrix pointer
          void MatrixHadamard(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            static HANDLER3(MatHadamardEigenHandler, MatHadamardCUDAHandler, MatHadamardNaiveHandler) handler;
            handler.handle(lhs, rhs, result);
          }

          // Matrix transpose
          void MatrixTranspose(const Matrix<Type>* mat, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(mat, "Matrix (mat) is a nullptr");

            static HANDLER3(MatTransposeEigenHandler, MatTransposeCUDAHandler, MatTransposeNaiveHandler) handler;
            handler.handle(mat, result);
          }

          // Matrix derivative transpose
          void MatrixDervTranspose(const size_t nrows_f, const size_t ncols_f,
                                   const size_t nrows_x, const size_t ncols_x,
                                   const Matrix<Type>* mat, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(mat, "Matrix (mat) is a nullptr");

            static HANDLER1(MatDervTransposeNaiveHandler) handler;
            handler.handle(nrows_f, ncols_f, nrows_x, ncols_x, mat, result);
          }

          // MatrixInverse
          void MatrixInverse(const Matrix<Type>* mat, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(mat, "Matrix mat is a nullptr");

            // TODO - Naive, CUDA
            static HANDLER1(MatInverseEigenHandler) handler;
            handler.handle(mat, result);
          }

          // Matrix determinant
          void MatrixDet(const Matrix<Type>* mat, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(mat, "Matrix mat is a nullptr");

            // TODO - Naive, CUDA
            static HANDLER1(MatDetEigenHandler) handler;
            handler.handle(mat, result);
          }

          // Matrix trace
          void MatrixTrace(const Matrix<Type>* mat, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(mat, "Matrix mat is a nullptr");

            // TODO - CUDA
            static HANDLER2(MatTraceEigenHandler, MatTraceNaiveHandler) handler; 
            handler.handle(mat, result);
          }

          // Matrix unary
          void MatrixUnary(const Matrix<Type>* mat, const FunctionType& func, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(mat, "Matrix mat is a nullptr");

            static HANDLER3(MatUnaryEigenHandler, MatUnaryCUDAHandler, MatUnaryNaiveHandler) handler; 
            handler.handle(mat, func, result);
          }

          // Matrix convolution
          void MatrixConv(const size_t stride_x, const size_t stride_y,
                          const size_t pad_x, const size_t pad_y, const Matrix<Type>* lhs,
                          const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Null pointer check
            NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
            NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

            // TODO - CUDA
            static HANDLER2(MatConvEigenHandler, MatConvNaiveHandler) handler; 
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
            
            // Applies only for naive handler
            static HANDLER1(MatDervConvNaiveHandler) handler;

            // Handle Matrix convolution derivative
            handler.handle(nrows_x, ncols_x, stride_x, stride_y, pad_x, pad_y, lhs, dlhs, rhs, drhs, result);
          }
        }
    }
}


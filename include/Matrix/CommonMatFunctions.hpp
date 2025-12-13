/**
 * @file include/Matrix/CommonMatFunctions.hpp
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

#pragma once

#include "GenericMatConv.hpp"
#include "GenericMatHadamard.hpp"
#include "GenericMatProduct.hpp"
#include "GenericMatSub.hpp"
#include "GenericMatSum.hpp"
#include "GenericMatTrace.hpp"
#include "GenericMatTranspose.hpp"
#include "GenericMatUnary.hpp"
#include "GenericMatInv.hpp"
#include "GenericMatDet.hpp"
#include "GenericMatVec.hpp"

#include "Matrix.hpp"
#include "MatrixBasics.hpp"

namespace CoolDiff {
  namespace Common {
    // Factorial of n 
    const size_t Factorial(const size_t);
  }

  namespace TensorR2 {
    // Matrix evaluation
    template <typename T> 
    inline Matrix<Type>& Eval(IMatrix<T>& Mexp) {
      if constexpr(true == std::is_same_v<T, Expression>) {
        // Reset graph/tree
        Mexp.resetImpl();
        // Return evaluation value
        return *(Mexp.eval());
      } else {
        // Reset graph/tree
        Mexp.reset();
        // Return evaluation value
        return *(Mexp.eval());
      }
    }

    // Matrix-Matrix forward derivative
    template <typename T>
    inline Matrix<Type>& DevalF(IMatrix<T>& Mexp, Matrix<Variable>& X) {
      if constexpr(true == std::is_same_v<T, Expression>) {
        // Reset graph/tree
        Mexp.resetImpl();
        // Return evaluation value
        return *(Mexp.devalF(X));
      } else {
        // Reset graph/tree
        Mexp.reset();
        // Return evaluation value
        return *(Mexp.devalF(X));
      }
    }

    // Precomputation of matrix expression
    template<typename T> 
    inline void PreComp(Matrix<T>& Mexp) {
      if constexpr(true == std::is_same_v<T, Expression>) {
        // Reset graph/tree
        Mexp.resetImpl();
        // Return evaluation value
        Mexp.traverse();
      } else {
        // Reset graph/tree
        Mexp.reset();
        // Return evaluation value
        Mexp.traverse();
      }
    }

    // Reverse derivative of matrix expression
    template<typename T, typename U, typename = std::enable_if_t< std::is_base_of_v<MetaMatrix, U>    || 
                                                                  std::is_base_of_v<MetaVariable, U>  ||
                                                                  std::is_base_of_v<MetaMatrix, T>>>
    inline Matrix<Type>& DevalR(T& Mexp, const U& X) {
      if constexpr(true == std::is_base_of_v<U, MetaMatrix>) {
        const size_t nrows = X.getNumRows();
        const size_t ncols = X.getNumColumns();
        if(auto it = Mexp.getCache().find(X.m_nidx); it != Mexp.getCache().end()) {
          return (*Mexp.getCache()[X.m_nidx]);
        } else {
          return *const_cast<MatType*>(CoolDiff::TensorR2::MatrixBasics::Zeros(nrows, ncols));
        }
      } else {
        const size_t nrows = Mexp.getNumRows();
        const size_t ncols = Mexp.getNumColumns();
        if(auto it = Mexp.getCache().find(X.m_nidx); it != Mexp.getCache().end()) {
          return (*Mexp.getCache()[X.m_nidx]);
        } else {
          return *const_cast<MatType*>(CoolDiff::TensorR2::MatrixBasics::Zeros(nrows, ncols));
        }
      }
    }

    // Forward mode algorithmic differentiation (Matrix)
    template <typename T, typename = CoolDiff::TensorR1::Details::IsPureMetaVariableType<T>>
    Matrix<Type>& DevalF(T& exp, const Matrix<Variable>& m, bool serial = true) {
      const size_t n = m.getNumElem();
      auto& result = Matrix<Type>::MatrixFactory::CreateMatrix(m.getNumRows(), m.getNumColumns());

      // If T is an expression
      if constexpr (true == std::is_same_v<T, Expression>) {
        if (true == exp.isRecursive() && true == serial) {
          std::transform(EXECUTION_SEQ m.getMatrixPtr(), m.getMatrixPtr() + n, result.getMatrixPtr(),
                        [&exp](const auto &v) { return CoolDiff::TensorR1::DevalF(exp, v); });
          return result;
        } else {
          // Copy expression and fill it up with exp
          Vector<Expression> exp_coll(n);
          exp_coll.reserve(n);
          std::fill(EXECUTION_PAR exp_coll.begin(), exp_coll.end(), exp);

          // Tranform and get results
          std::transform(EXECUTION_PAR exp_coll.begin(), exp_coll.end(), m.getMatrixPtr(), result.getMatrixPtr(),
                        [](auto &v1, const auto &v2) { return CoolDiff::TensorR1::DevalF(v1, v2); });
          return result;
        }
      }

      // If T is not an exprssion
      if (true == serial) {
        std::transform(EXECUTION_SEQ m.getMatrixPtr(), m.getMatrixPtr() + n, result.getMatrixPtr(),
                      [&exp](const auto &v) { return CoolDiff::TensorR1::DevalF(exp, v); });
      } else {
        // Copy expression and fill it up with exp
        Vector<Expression> exp_coll(n);
        exp_coll.reserve(n);
        std::fill(EXECUTION_PAR exp_coll.begin(), exp_coll.end(), exp);

        // Tranform and get results
        std::transform(EXECUTION_PAR exp_coll.begin(), exp_coll.end(),
                      m.getMatrixPtr(), result.getMatrixPtr(),
                      [](auto &v1, const auto &v2) { return CoolDiff::TensorR1::DevalF(v1, v2); });
      }

      return result;
    }

    // Reverse mode algorithmic differentiation (Matrix)
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>> 
    Matrix<Type>& DevalR(T& exp, const Matrix<Variable>& m) {
      const size_t n = m.getNumElem();
      auto& result = Matrix<Type>::MatrixFactory::CreateMatrix(m.getNumRows(), m.getNumColumns());

      // Precompute (By design, the operation is serial)
      CoolDiff::TensorR1::PreComp(exp);

      std::transform(EXECUTION_SEQ m.getMatrixPtr(), m.getMatrixPtr() + n, result.getMatrixPtr(),
                    [&exp](const auto &v) { return CoolDiff::TensorR1::DevalR(exp, v); });

      return result;
    }

    // Jacobian reverse mode (Scalar - Vector)
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>>
    Matrix<Type>& JacobianR(T& exp, const Vector<Variable>& vec) {
      const size_t rows = vec.size();
      auto& result = Matrix<Type>::MatrixFactory::CreateMatrix(rows, 1);

      // Precompute (By design, the operation is serial)
      CoolDiff::TensorR1::PreComp(exp);

      std::transform(EXECUTION_SEQ vec.cbegin(), vec.cend(), result.getMatrixPtr(),
                    [&exp](const auto &v) { return CoolDiff::TensorR1::DevalR(exp, v); });

      return result;
    }

    // Jacobian reverse mode (Scalar - Matrix)
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>>
    Matrix<Type>& JacobianR(T& exp, const Matrix<Variable>& X) {
      Vector<Variable> vec(X.getMatrixPtr(), X.getMatrixPtr() + X.getNumElem());
      return JacobianR(exp, vec);
    }

    // Hessian reverse mode (Scalar - Vector)
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>>
    Matrix<Type>& HessianR(T& exp, const Vector<Variable>& vec) {
      const size_t dim = vec.size();
      auto& result = Matrix<Type>::MatrixFactory::CreateMatrix(dim, dim);
      Matrix<Expression> firstSym(dim, 1);

      // Exploit Hessian symmetry
      for (size_t i{}; i < dim; ++i) {
        firstSym[i] = CoolDiff::TensorR1::SymDiff(exp, vec[i]);
        // Precompute
        CoolDiff::TensorR1::PreComp(firstSym[i]);
        for (size_t j{}; j <= i; ++j) {
          if (i < j) {
            result(i, j) = CoolDiff::TensorR1::DevalR(firstSym[i], vec[j]);
            result(j, i) = result(i, j);
          } else if (i == j) {
            result(i, j) = CoolDiff::TensorR1::DevalR(firstSym[i], vec[j]);
          }
        }
      }

      return result;
    }

    // Hessian reverse mode (Scalar - Matrix)
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>>
    Matrix<Type>& HessianR(T& exp, const Matrix<Variable>& X) {
      Vector<Variable> vec(X.getMatrixPtr(), X.getMatrixPtr() + X.getNumElem());
      return HessianR(exp, vec);
    }

    // Symbolic Jacobian of expression (Scalar - Vector)
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>>
    Matrix<Expression>& SymGradient(T& exp, const Vector<Variable>& vec) {
      const size_t rows = vec.size();
      auto& result = Matrix<Expression>::MatrixFactory::CreateMatrix(rows, 1);
      for (size_t i{}; i < rows; ++i) {
        result(i, 0) = CoolDiff::TensorR1::SymDiff(exp, vec[i]);
      }
      return result;
    }

    // Symbolic Jacobian of expression (Scalar - Matrix)
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>>
    Matrix<Expression>& SymGradient(T& exp, const Matrix<Variable>& X) {
      Vector<Variable> vec(X.getMatrixPtr(), X.getMatrixPtr() + X.getNumElem());
      return SymGradient(exp, vec);
    }

    // Symbolic Hessian of expression (Scalar - Vector)
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>>
    Matrix<Expression>& SymHessian(T& exp, const Vector<Variable>& vec) {
      const size_t dim = vec.size();
      auto &result = Matrix<Expression>::MatrixFactory::CreateMatrix(dim, dim);

      // Exploit Hessian symmetry
      for (size_t i{}; i < dim; ++i) {
        for (size_t j{}; j <= i; ++j) {
          if (i < j) {
            result(i, j) = CoolDiff::TensorR1::SymDiff(CoolDiff::TensorR1::SymDiff(exp, vec[i]), vec[j]);
            result(j, i) = result(i, j);
          } else if (i == j) {
            result(i, j) = CoolDiff::TensorR1::SymDiff(CoolDiff::TensorR1::SymDiff(exp, vec[i]), vec[j]);
          }
        }
      }

      return result;
    }

    // Symbolic Hessian of expression (Scalar - Matrix)
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>>
    Matrix<Expression>& SymHessian(T& exp, const Matrix<Variable>& X) {
      Vector<Variable> vec(X.getMatrixPtr(), X.getMatrixPtr() + X.getNumElem());
      return SymHessian(exp, vec);
    }

    // Symbolic Expression (Scalar - Matrix)
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>>
    Matrix<Expression>& SymMatDiff(T& exp, const Matrix<Variable>& m) {
      const size_t n = m.getNumElem();
      auto& result = Matrix<Expression>::MatrixFactory::CreateMatrix(m.getNumRows(), m.getNumColumns());
      for (size_t i{}; i < n; ++i) {
        result[i] = CoolDiff::TensorR1::SymDiff(exp, m[i]);
      }
      return result;
    }
  }
}


// Custom functions
// Matrix Sin function
UNARY_MATRIX_OPERATOR(SinM, [](Type a) { return std::sin(a); },
                            [](Type b) { return std::cos(b); })

// Matrix Cos function
UNARY_MATRIX_OPERATOR(CosM, [](Type a) { return std::cos(a); },
                            [](Type b) { return -std::sin(b); })

// Matrix Exp function
UNARY_MATRIX_OPERATOR(ExpM, [](Type a) { return std::exp(a); },
                            [](Type b) { return std::exp(b); })

// Matrix Log function
UNARY_MATRIX_OPERATOR(LogM, [](Type a) { return std::log(a); },
                            [](Type b) { return ((Type)(1)/b); })

// Matrix Sqrt function
UNARY_MATRIX_OPERATOR(SqrtM,  [](Type a) { return std::sqrt(a); },
                              [](Type b) { return ((Type)(1)/(2*std::sqrt(b))); })

// Matrix tanh function
UNARY_MATRIX_OPERATOR(TanhM,  [](Type a) { return std::tanh(a); },
                              [](Type b) { return ((Type)1 - std::tanh(b)*std::tanh(b)); })

// Matrix Sigmoid function
UNARY_MATRIX_OPERATOR(SigmoidM, [](Type a) { 
                                  Type res = ((Type)(1)) / (((Type)(1)) + std::exp(-a));
                                  return res;
                                  },
                                [](Type b) {
                                  Type res = ((Type)(1)) / (((Type)(1)) + std::exp(-b));
                                  return res * (((Type)(1)) - res); });
// Matrix ReLU function
#ifndef USE_COMPLEX_MATH
  // Relu activation  
  UNARY_MATRIX_OPERATOR(ReLUM,  [](Type a) { return ((a >= (Type)(0)) ? a : 0);  },
                                [](Type b) { return ((b >= (Type)(0)) ? 1 : 0); })

  // Leaky relu activation
  UNARY_MATRIX_OPERATOR(LeakyReLUM,   [](Type a) { return ((a >= (Type)(0)) ? a : 0.1*a);  },
                                      [](Type b) { return ((b >= (Type)(0)) ? 1 : 0.1); })

  // Matrix abs function
  UNARY_MATRIX_OPERATOR(AbsM, [](Type a) { return std::abs(a); },
                              [](Type b) { return (Type)(((Type)(0) < b) - (b < (Type)(0))); })
#endif


// Broadcast 
template<Axis axis = Axis::ALL, typename T>
constexpr const auto& broadcast(const IMatrix<T>& X, const size_t n) {
  const size_t xrows = X.getNumRows();
  const size_t xcols = X.getNumColumns();

  ASSERT((xcols == 1 || xrows == 1), "Cannot broadcast a matrix, use Kronocker product");
  
  if constexpr(axis == Axis::COLUMN) { 
    return (X * CoolDiff::TensorR2::MatrixBasics::OnesRef(1, n));
  } else if constexpr(axis == Axis::ROW) {
    return (CoolDiff::TensorR2::MatrixBasics::OnesRef(n, 1) * X);
  } else {
    return (CoolDiff::TensorR2::MatrixBasics::OnesRef(n, 1) * X * CoolDiff::TensorR2::MatrixBasics::OnesRef(1, n));
  }
}

// Sigma computation
template <Axis axis = Axis::ALL, typename T> 
constexpr const auto& Sigma(const IMatrix<T>& X) {
  const size_t rows = X.getNumRows();
  const size_t cols = X.getNumColumns();
  if constexpr(axis == Axis::ROW) {  
    return CoolDiff::TensorR2::MatrixBasics::OnesRef(1, rows)*X;
  } else if constexpr(axis == Axis::COLUMN) {
    return X*CoolDiff::TensorR2::MatrixBasics::OnesRef(cols, 1);
  } else {
    return CoolDiff::TensorR2::MatrixBasics::OnesRef(1, rows)*X*CoolDiff::TensorR2::MatrixBasics::OnesRef(cols, 1);
  }
}

// Frobenius norm 
template <typename T>
constexpr const auto& MatrixFrobeniusNorm(const IMatrix<T>& X) {
  return SqrtM(Sigma(X^X));
}

// Matrix vertical concatenation
template<ConcatAxis axis = ConcatAxis::VERTICAL, typename T1, typename T2>
constexpr const auto& concat(const IMatrix<T1>& X, const IMatrix<T2>& Y) {
  const size_t x_rows = X.getNumRows();
  const size_t x_cols = X.getNumColumns(); 
  const size_t y_rows = Y.getNumRows();
  const size_t y_cols = Y.getNumColumns(); 

  if constexpr(ConcatAxis::VERTICAL == axis) {
    ASSERT((x_cols == y_cols), "Column dimensions are not the same for concatenation");

    Matrix<Type>& A = Matrix<Type>::MatrixFactory::CreateMatrix((x_rows + y_rows), x_rows);
    Matrix<Type>& B = Matrix<Type>::MatrixFactory::CreateMatrix((x_rows + y_rows), y_rows);

    A.setBlockMat({0, (x_rows-1)}, {0, (x_rows-1)}, CoolDiff::TensorR2::MatrixBasics::Eye(x_rows));
    B.setBlockMat({x_rows, (x_rows+y_rows-1)}, {0, (y_rows-1)}, CoolDiff::TensorR2::MatrixBasics::Eye(y_rows));
    
    return A*X + B*Y;
  } else if constexpr(ConcatAxis::HORIZONTAL == axis) {
    ASSERT((x_rows == y_rows), "Column dimensions are not the same for concatenation");

    Matrix<Type>& A = Matrix<Type>::MatrixFactory::CreateMatrix(x_cols, (x_cols + y_cols));
    Matrix<Type>& B = Matrix<Type>::MatrixFactory::CreateMatrix(y_cols, (x_cols + y_cols));

    A.setBlockMat({0, (x_cols-1)}, {0, (x_cols-1)}, CoolDiff::TensorR2::MatrixBasics::Eye(x_cols));
    B.setBlockMat({0, (y_cols-1)}, {x_cols, (x_cols+y_cols-1)}, CoolDiff::TensorR2::MatrixBasics::Eye(y_cols));
    
    return X*A + Y*B;
  }
}

// Softmax function
template<Axis axis = Axis::ALL, typename T>
constexpr const auto& SoftMax(const IMatrix<T>& X) {
  const size_t rows = X.getNumRows();
  const size_t cols = X.getNumColumns();
  if constexpr(Axis::ROW == axis) {
    return ExpM(X - (CoolDiff::TensorR2::MatrixBasics::OnesRef(rows,1))*LogM(Sigma<Axis::ROW>(ExpM(X))));
  } else if constexpr(Axis::COLUMN == axis) {
    return ExpM(X - LogM(Sigma<Axis::COLUMN>(ExpM(X)))*(CoolDiff::TensorR2::MatrixBasics::OnesRef(1,cols)));
  } else {
    return ExpM(X - (CoolDiff::TensorR2::MatrixBasics::OnesRef(rows,1))*LogM(Sigma<Axis::ALL>(ExpM(X)))*(CoolDiff::TensorR2::MatrixBasics::OnesRef(1,cols)));
  }
}

// Matrix pow
template<typename T>
Matrix<Expression>& pow(const IMatrix<T>& X, const size_t n) {
  // Dimensions of matrix X
  const size_t rows = X.getNumRows();
  const size_t cols = X.getNumColumns();

  // Assert for square matrix
  ASSERT(rows == cols, "X matrix is not a square matrix");

  // Temporary matrix array
  Matrix<Expression>* tmp[n+1];
  for(size_t i{}; i <= n; ++i) {
    tmp[i] = Matrix<Expression>::MatrixFactory::CreateMatrixPtr();
  }

  // Run loop till truncation
  (*tmp[0]) = CoolDiff::TensorR2::MatrixBasics::EyeRef(rows);
  for(size_t i{1}; i <= n; ++i) {
      (*tmp[i]) = (*tmp[i-1])*(X);
  }

  // Return result
  return (*tmp[n]);
}

// Matrix exponential
template<size_t N = 20, typename T>
Matrix<Expression>& MatrixExponential(const IMatrix<T>& X) {
  // Dimensions of matrix X
  const size_t rows = X.getNumRows();
  const size_t cols = X.getNumColumns();

  // Assert for square matrix
  ASSERT(rows == cols, "X matrix is not a square matrix");

  // Create result matrix
  auto& result = Matrix<Expression>::MatrixFactory::CreateMatrix();

  // Temporary matrix array
  Matrix<Expression>* tmp[N+1];
  for(size_t i{}; i <= N; ++i) {
    tmp[i] = Matrix<Expression>::MatrixFactory::CreateMatrixPtr();
  }

  // Run loop till truncation
  (*tmp[0]) = CoolDiff::TensorR2::MatrixBasics::EyeRef(rows);
  result = (*tmp[0]);
  for(size_t i{1}; i <= N; ++i) {
      (*tmp[i]) = (*tmp[i-1])*X;
      result = result + ((*tmp[i])/CoolDiff::Common::Factorial(i));
  }

  // Return result
  return result;
}

// Matrix log
template<size_t N = 20, typename T>
Matrix<Expression>& MatrixLog(const IMatrix<T>& X) {
  // Dimensions of matrix X
  const size_t rows = X.getNumRows();
  const size_t cols = X.getNumColumns();

  // Assert for square matrix
  ASSERT(rows == cols, "X matrix is not a square matrix");

  // Create result matrix
  auto& result = Matrix<Expression>::MatrixFactory::CreateMatrix();
  
  // Temporary matrix array
  Matrix<Expression>* tmp[N+1];
  for(size_t i{}; i <= N; ++i) {
    tmp[i] = Matrix<Expression>::MatrixFactory::CreateMatrixPtr();
  }

  // Run loop till truncation
  (*tmp[0]) = CoolDiff::TensorR2::MatrixBasics::EyeRef(rows);
  result = CoolDiff::TensorR2::MatrixBasics::ZerosRef(rows, cols);
  for(size_t i{1}; i <= N; ++i) {
      (*tmp[i]) = (*tmp[i-1])*(X - CoolDiff::TensorR2::MatrixBasics::EyeRef(rows));
      result = result + ((Type)std::pow(((Type)(-1)), i+1)/((Type)(i)))*((*tmp[i]));
  }

  // Return result
  return result;
}
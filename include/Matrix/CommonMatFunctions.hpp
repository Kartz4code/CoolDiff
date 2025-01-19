/**
 * @file include/Matrix/CommonMatFunctions.hpp
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

#include "GenericMatConv.hpp"
#include "GenericMatHadamard.hpp"
#include "GenericMatProduct.hpp"
#include "GenericMatSigma.hpp"
#include "GenericMatSub.hpp"
#include "GenericMatSum.hpp"
#include "GenericMatTrace.hpp"
#include "GenericMatTranspose.hpp"
#include "GenericMatUnary.hpp"
#include "GenericMatInv.hpp"
#include "GenericMatDet.hpp"

#include "Matrix.hpp"
#include "MatrixBasics.hpp"

// Custom functions
UNARY_MATRIX_OPERATOR(SinM, [](Type a) { return std::sin(a); },
                            [](Type b) { return std::cos(b); })

UNARY_MATRIX_OPERATOR(CosM, [](Type a) { return std::cos(a); },
                            [](Type b) { return -std::sin(b); })

UNARY_MATRIX_OPERATOR(ExpM, [](Type a) { return std::exp(a); },
                            [](Type b) { return std::exp(b); })

UNARY_MATRIX_OPERATOR(LogM, [](Type a) { return std::log(a); },
                            [](Type b) { return ((Type)(1)/b); })

UNARY_MATRIX_OPERATOR(SqrtM, [](Type a) { return std::sqrt(a); },
                             [](Type b) { return ((Type)(1)/(2*std::sqrt(b))); })


UNARY_MATRIX_OPERATOR(SigmoidM, [](Type a) { 
                                   Type res = ((Type)(1)) / (((Type)(1)) + std::exp(-a));
                                   return res;
                                  },
                                [](Type b) {
                                   Type res = ((Type)(1)) / (((Type)(1)) + std::exp(-b));
                                   return res * (((Type)(1)) - res);
                                  });

// Frobenius norm 
template <typename T>
constexpr const auto& MatrixFrobeniusNorm(const IMatrix<T>& X) {
  return SqrtM(trace(transpose(X)*X));
}

template<Axis axis = Axis::ALL, typename T>
constexpr const auto& SoftMax(const IMatrix<T>& X) {
  const size_t rows = X.getNumRows();
  const size_t cols = X.getNumColumns();
  if constexpr(Axis::ROW == axis) {
    return ExpM(X - (OnesRef(rows,1))*LogM(sigma<Axis::ROW>(ExpM(X))));
  } else if constexpr(Axis::COLUMN == axis) {
    return ExpM(X - LogM(sigma<Axis::COLUMN>(ExpM(X)))*(OnesRef(1,cols)));
  } else {
    return ExpM(X - (OnesRef(rows,1))*LogM(sigma<Axis::ALL>(ExpM(X)))*(OnesRef(1,cols)));
  }
}

// Matrix evaluation
template <typename T> Matrix<Type> &Eval(Matrix<T> &Mexp) {
  // Reset graph/tree
  Mexp.resetImpl();
  // Return evaluation value
  return *(Mexp.eval());
}

// Matrix-Matrix derivative evaluation
template <typename T>
Matrix<Type> &DevalF(Matrix<T> &Mexp, Matrix<Variable> &X) {
  // Reset graph/tree
  Mexp.resetImpl();
  // Return evaluation value
  return *(Mexp.devalF(X));
}

///////////////////////////////////////////////////////////////////////////////////////////

// Forward mode algorithmic differentiation (Matrix)
template <typename T>
Matrix<Type> &DevalF(T &exp, const Matrix<Variable> &m, bool serial = true) {
  const size_t n = m.getNumElem();
  auto &result = Matrix<Type>::MatrixFactory::CreateMatrix(m.getNumRows(),
                                                           m.getNumColumns());

  // If T is an expression
  if constexpr (true == std::is_same_v<T, Expression>) {
    if (true == exp.isRecursive() && true == serial) {
      std::transform(EXECUTION_SEQ m.getMatrixPtr(), m.getMatrixPtr() + n,
                     result.getMatrixPtr(),
                     [&exp](const auto &v) { return DevalF(exp, v); });
      return result;
    } else {
      // Copy expression and fill it up with exp
      Vector<Expression> exp_coll(n);
      exp_coll.reserve(n);
      std::fill(EXECUTION_PAR exp_coll.begin(), exp_coll.end(), exp);

      // Tranform and get results
      std::transform(EXECUTION_PAR exp_coll.begin(), exp_coll.end(),
                     m.getMatrixPtr(), result.getMatrixPtr(),
                     [](auto &v1, const auto &v2) { return DevalF(v1, v2); });
      return result;
    }
  }

  // If T is not an exprssion
  if (true == serial) {
    std::transform(EXECUTION_SEQ m.getMatrixPtr(), m.getMatrixPtr() + n,
                   result.getMatrixPtr(),
                   [&exp](const auto &v) { return DevalF(exp, v); });
  } else {
    // Copy expression and fill it up with exp
    Vector<Expression> exp_coll(n);
    exp_coll.reserve(n);
    std::fill(EXECUTION_PAR exp_coll.begin(), exp_coll.end(), exp);

    // Tranform and get results
    std::transform(EXECUTION_PAR exp_coll.begin(), exp_coll.end(),
                   m.getMatrixPtr(), result.getMatrixPtr(),
                   [](auto &v1, const auto &v2) { return DevalF(v1, v2); });
  }

  return result;
}

// Reverse mode algorithmic differentiation (Matrix)
template <typename T> Matrix<Type> &DevalR(T &exp, const Matrix<Variable> &m) {
  const size_t n = m.getNumElem();
  auto &result = Matrix<Type>::MatrixFactory::CreateMatrix(m.getNumRows(),
                                                           m.getNumColumns());

  // Precompute (By design, the operation is serial)
  PreComp(exp);

  std::transform(EXECUTION_SEQ m.getMatrixPtr(), m.getMatrixPtr() + n,
                 result.getMatrixPtr(),
                 [&exp](const auto &v) { return DevalR(exp, v); });

  return result;
}

// Jacobian reverse mode (Scalar - Vector)
template <typename T>
Matrix<Type> &JacobianR(T &exp, const Vector<Variable> &vec) {
  const size_t rows = vec.size();
  auto &result = Matrix<Type>::MatrixFactory::CreateMatrix(rows, 1);

  // Precompute (By design, the operation is serial)
  PreComp(exp);

  std::transform(EXECUTION_SEQ vec.cbegin(), vec.cend(), result.getMatrixPtr(),
                 [&exp](const auto &v) { return DevalR(exp, v); });

  return result;
}

// Jacobian reverse mode (Scalar - Matrix)
template <typename T>
Matrix<Type> &JacobianR(T &exp, const Matrix<Variable> &X) {
  Vector<Variable> vec(X.getMatrixPtr(), X.getMatrixPtr() + X.getNumElem());
  return JacobianR(exp, vec);
}

// Hessian reverse mode (Scalar - Vector)
template <typename T>
Matrix<Type> &HessianR(T &exp, const Vector<Variable> &vec) {
  const size_t dim = vec.size();
  auto &result = Matrix<Type>::MatrixFactory::CreateMatrix(dim, dim);
  Matrix<Expression> firstSym(dim, 1);

  // Exploit Hessian symmetry
  for (size_t i{}; i < dim; ++i) {
    firstSym[i] = SymDiff(exp, vec[i]);
    // Precompute
    PreComp(firstSym[i]);
    for (size_t j{}; j <= i; ++j) {
      if (i < j) {
        result(i, j) = DevalR(firstSym[i], vec[j]);
        result(j, i) = result(i, j);
      } else if (i == j) {
        result(i, j) = DevalR(firstSym[i], vec[j]);
      }
    }
  }

  return result;
}

// Hessian reverse mode (Scalar - Matrix)
template <typename T>
Matrix<Type> &HessianR(T &exp, const Matrix<Variable> &X) {
  Vector<Variable> vec(X.getMatrixPtr(), X.getMatrixPtr() + X.getNumElem());
  return HessianR(exp, vec);
}

// Symbolic Jacobian of expression (Scalar - Vector)
template <typename T>
Matrix<Expression> &SymJacobian(T &exp, const Vector<Variable> &vec) {
  const size_t rows = vec.size();
  auto &result = Matrix<Expression>::MatrixFactory::CreateMatrix(rows, 1);
  for (size_t i{}; i < rows; ++i) {
    result(i, 0) = SymDiff(exp, vec[i]);
  }
  return result;
}

// Symbolic Jacobian of expression (Scalar - Matrix)
template <typename T>
Matrix<Expression> &SymJacobian(T &exp, const Matrix<Variable> &X) {
  Vector<Variable> vec(X.getMatrixPtr(), X.getMatrixPtr() + X.getNumElem());
  return SymJacobian(exp, vec);
}

// Symbolic Hessian of expression (Scalar - Vector)
template <typename T>
Matrix<Expression> &SymHessian(T &exp, const Vector<Variable> &vec) {
  const size_t dim = vec.size();
  auto &result = Matrix<Expression>::MatrixFactory::CreateMatrix(dim, dim);

  // Exploit Hessian symmetry
  for (size_t i{}; i < dim; ++i) {
    for (size_t j{}; j <= i; ++j) {
      if (i < j) {
        result(i, j) = SymDiff(SymDiff(exp, vec[i]), vec[j]);
        result(j, i) = result(i, j);
      } else if (i == j) {
        result(i, j) = SymDiff(SymDiff(exp, vec[i]), vec[j]);
      }
    }
  }

  return result;
}

// Symbolic Hessian of expression (Scalar - Matrix)
template <typename T>
Matrix<Expression> &SymHessian(T &exp, const Matrix<Variable> &X) {
  Vector<Variable> vec(X.getMatrixPtr(), X.getMatrixPtr() + X.getNumElem());
  return SymHessian(exp, vec);
}

// Symbolic Expression (Scalar - Matrix)
template <typename T>
Matrix<Expression> &SymMatDiff(T &exp, const Matrix<Variable> &m) {
  const size_t n = m.getNumElem();
  auto &result = Matrix<Expression>::MatrixFactory::CreateMatrix(
      m.getNumRows(), m.getNumColumns());
  for (size_t i{}; i < n; ++i) {
    result[i] = SymDiff(exp, m[i]);
  }
  return result;
}

// Matrix exponential
Matrix<Expression>& MatrixExponential(const Matrix<Expression>&, const size_t = 20);


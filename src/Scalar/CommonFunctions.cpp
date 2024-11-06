/**
 * @file src/Scalar/CommonFunctions.cpp
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

#include "CommonFunctions.hpp"
#include "Matrix.hpp"

// Evaluate function (Scalar)
Type EvalExp(Expression& exp) {
  // Reset graph/tree
  exp.resetImpl();
  // Return evaluation value
  return exp.eval();
}

// Forward mode algorithmic differentiation (Scalar)
Type DevalFExp(Expression& exp, const Variable& x) {
  // Reset visited flags in the tree
  exp.resetImpl();
  // Return forward differentiation value
  return exp.devalF(x);
}

// Reverse mode algorithmic differentiation (Scalar)
Type DevalRExp(Expression& exp, const Variable& x) {
  // Return reverse differentiation value
  return exp.devalR(x);
}

// Main precomputation of reverse mode AD computation 1st
void PreCompExp(Expression& exp) {
  // Reset visited flags in the tree
  exp.resetImpl();
  // Traverse tree
  exp.traverse();
}

// Main reverse mode AD table 1st
OMPair& PreCompCacheExp(Expression& exp) {
  // Reset flags
  exp.resetImpl();
  // Traverse evaluation graph/tree
  exp.traverse();
  // Return cache
  return exp.getCache();
}

// Symbolic Expression
Expression& SymDiff(Expression& exp, const Variable& var) {
  // Reset graph/tree
  return exp.SymDiff(var);
}

// Forward mode algorithmic differentiation (Matrix)
Matrix<Type> &DevalF(Expression &exp, const Matrix<Variable> &m,
                     bool serial_exec) {
  const size_t n = m.getNumElem();
  auto &result = Matrix<Type>::MatrixFactory::CreateMatrix(m.getNumRows(),
                                                           m.getNumColumns());

  if (true == exp.isRecursive() || true == serial_exec) {
    std::transform(EXECUTION_SEQ m.getMatrixPtr(), m.getMatrixPtr() + n,
                   result.getMatrixPtr(),
                   [&exp](const auto &v) { return DevalF(exp, v); });
  } else {
    // Copy expression and fill it up with exp
    Vector<Expression> exp_coll(n);
    std::fill(EXECUTION_PAR exp_coll.begin(), exp_coll.end(), exp);

    // Tranform and get results
    std::transform(EXECUTION_PAR exp_coll.begin(), exp_coll.end(),
                   m.getMatrixPtr(), result.getMatrixPtr(),
                   [](auto &v1, const auto &v2) { return DevalF(v1, v2); });
  }

  return result;
}
// Reverse mode algorithmic differentiation (Matrix)
Matrix<Type> &DevalR(Expression &exp, const Matrix<Variable> &m) {
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

void DevalR(Expression &exp, const Matrix<Variable> &X, Matrix<Type> *&result) {
  const size_t lrows = X.getNumRows();
  const size_t rcols = X.getNumColumns();

  if (nullptr == result) {
    result = Matrix<Type>::MatrixFactory::CreateMatrixPtr(lrows, rcols);
  } else if ((lrows != result->getNumRows()) ||
             (rcols != result->getNumColumns())) {
    result = Matrix<Type>::MatrixFactory::CreateMatrixPtr(lrows, rcols);
  }

  // Precompute (By design, the operation is serial)
  PreComp(exp);

  const size_t n = X.getNumElem();
  std::transform(EXECUTION_SEQ X.getMatrixPtr(), X.getMatrixPtr() + n,
                 result->getMatrixPtr(),
                 [&exp](const auto &v) { return DevalR(exp, v); });
}

// Derivative of expression (Matrix)
Matrix<Type> &Deval(Expression &exp, const Matrix<Variable> &m, ADMode ad) {
  if (ad == ADMode::FORWARD) {
    return DevalF(exp, m);
  } else {
    return DevalR(exp, m);
  }
}

// Jacobian forward mode
Matrix<Type> &JacobF(Expression &exp, const Vector<Variable> &vec,
                     bool serial_exec) {
  const size_t rows = vec.size();
  auto &result = Matrix<Type>::MatrixFactory::CreateMatrix(rows, 1);

  if (true == exp.isRecursive() || true == serial_exec) {
    std::transform(EXECUTION_SEQ vec.begin(), vec.end(), result.getMatrixPtr(),
                   [&exp](const auto &v) { return DevalF(exp, v); });
  } else {
    // Copy expression and fill it up with exp
    Vector<Expression> exp_coll(rows);
    std::fill(EXECUTION_PAR exp_coll.begin(), exp_coll.end(), exp);

    // Tranform and get results
    std::transform(EXECUTION_PAR exp_coll.begin(), exp_coll.end(), vec.begin(),
                   result.getMatrixPtr(),
                   [](auto &v1, const auto &v2) { return DevalF(v1, v2); });
  }

  return result;
}
// Jacobian reverse mode
Matrix<Type> &JacobR(Expression &exp, const Vector<Variable> &vec) {
  const size_t rows = vec.size();
  auto &result = Matrix<Type>::MatrixFactory::CreateMatrix(rows, 1);

  // Precompute (By design, the operation is serial)
  PreComp(exp);

  std::transform(EXECUTION_SEQ vec.cbegin(), vec.cend(), result.getMatrixPtr(),
                 [&exp](const auto &v) { return DevalR(exp, v); });

  return result;
}
// Jacobian of expression
Matrix<Type> &Jacob(Expression &exp, const Vector<Variable> &vec, ADMode ad) {
  if (ad == ADMode::FORWARD) {
    return JacobF(exp, vec);
  } else {
    return JacobR(exp, vec);
  }
}

// Symbolic Jacobian of expression (Vector)
Matrix<Expression> &JacobSym(Expression &exp, const Vector<Variable> &vec) {
  const size_t rows = vec.size();
  auto &result = Matrix<Expression>::MatrixFactory::CreateMatrix(rows, 1);
  for (size_t i{}; i < rows; ++i) {
    result(i, 1) = SymDiff(exp, vec[i]);
  }
  return result;
}

// Hessian forward mode
Matrix<Type> &HessF(Expression &exp, const Vector<Variable> &vec) {
  const size_t dim = vec.size();
  auto &result = Matrix<Type>::MatrixFactory::CreateMatrix(dim, dim);
  Matrix<Expression> firstSym(dim, 1);

  // Exploit Hessian symmetry
  for (size_t i{}; i < dim; ++i) {
    firstSym[i] = SymDiff(exp, vec[i]);
    for (size_t j{}; j < dim; ++j) {
      if (i < j) {
        result(i, j) = DevalF(firstSym[i], vec[j]);
        result(j, i) = result(i, j);
      } else if (i == j) {
        result(i, j) = DevalF(firstSym[i], vec[j]);
      }
    }
  }

  return result;
}

// Hessian reverse mode
Matrix<Type> &HessR(Expression &exp, const Vector<Variable> &vec) {
  const size_t dim = vec.size();
  auto &result = Matrix<Type>::MatrixFactory::CreateMatrix(dim, dim);
  Matrix<Expression> firstSym(dim, 1);

  // Exploit Hessian symmetry
  for (size_t i{}; i < dim; ++i) {
    firstSym[i] = SymDiff(exp, vec[i]);
    // Precompute
    PreComp(firstSym[i]);
    for (size_t j{}; j < dim; ++j) {
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

// Hessian of expression
Matrix<Type> &Hess(Expression &exp, const Vector<Variable> &vec, ADMode ad) {
  if (ad == ADMode::FORWARD) {
    return HessF(exp, vec);
  } else {
    return HessR(exp, vec);
  }
}

// Symbolic Hessian of expression
Matrix<Expression> &HessSym(Expression &exp, const Vector<Variable> &vec) {
  const size_t dim = vec.size();
  auto &result = Matrix<Expression>::MatrixFactory::CreateMatrix(dim, dim);

  // Exploit Hessian symmetry
  for (size_t i{}; i < dim; ++i) {
    for (size_t j{}; j < dim; ++j) {
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

// Symbolic Expression (Matrix)
Matrix<Expression> &SymMatDiff(Expression &exp, const Matrix<Variable> &m) {
  const size_t n = m.getNumElem();
  auto &result = Matrix<Expression>::MatrixFactory::CreateMatrix(
      m.getNumRows(), m.getNumColumns());
  for (size_t i{}; i < n; ++i) {
    result[i] = SymDiff(exp, m[i]);
  }
  return result;
}
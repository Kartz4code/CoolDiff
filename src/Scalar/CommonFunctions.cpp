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

namespace details {
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
  Expression& SymDiffExp(Expression& exp, const Variable& var) {
    // Reset graph/tree
    return exp.SymDiff(var);
  }
}
/*

// Jacobian forward mode
Matrix<Type> &JacobF(Expression &exp, const Vector<Variable> &vec, bool serial_exec) {
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


// Jacobian of expression
Matrix<Type> &Jacob(Expression &exp, const Vector<Variable> &vec, ADMode ad) {
  if (ad == ADMode::FORWARD) {
    return JacobF(exp, vec);
  } else {
    return JacobR(exp, vec);
  }
}

// Hessian of expression
Matrix<Type> &Hess(Expression &exp, const Vector<Variable> &vec, ADMode ad) {
  if (ad == ADMode::FORWARD) {
    return HessF(exp, vec);
  } else {
    //return HessR(exp, vec);
  }
}

*/
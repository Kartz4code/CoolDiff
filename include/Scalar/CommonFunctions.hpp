/**
 * @file include/CommonFunctions.hpp
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

#include "Expression.hpp"
#include "Parameter.hpp"
#include "Variable.hpp"

// Binary operators
#include "GenericDiv.hpp"
#include "GenericPow.hpp"
#include "GenericProduct.hpp"
#include "GenericSub.hpp"
#include "GenericSum.hpp"

// Unary operator
#include "GenericACos.hpp"
#include "GenericACosh.hpp"
#include "GenericASin.hpp"
#include "GenericASinh.hpp"
#include "GenericATan.hpp"
#include "GenericATanh.hpp"
#include "GenericCos.hpp"
#include "GenericCosh.hpp"
#include "GenericExp.hpp"
#include "GenericLog.hpp"
#include "GenericNeg.hpp"
#include "GenericSin.hpp"
#include "GenericSinh.hpp"
#include "GenericSqrt.hpp"
#include "GenericTan.hpp"
#include "GenericTanh.hpp"

// Main evaluate expression function
Type Eval(Expression &);
// Main precomputation of reverse mode AD computation
void PreComp(Expression &);
// Main reverse mode AD table
OMPair &PreCompCache(const Expression &);


// Symbolic Expression
Expression &SymDiff(Expression &, const Variable &);


// Main forward mode AD computation
Type DevalF(Expression &, const Variable &);
// Main reverse mode AD computation
Type DevalR(Expression &, const Variable &);
// Derivative of expression
Type Deval(Expression &, const Variable &, ADMode = ADMode::REVERSE);


// Forward mode algorithmic differentiation (Matrix)
Matrix<Type> &DevalF(Expression &, const Matrix<Variable> &, bool = false);
// Reverse mode algorithmic differentiation (Matrix)
Matrix<Type> &DevalR(Expression &, const Matrix<Variable> &);
// Derivative of expression (Matrix)
Matrix<Type> &Deval(Expression &, const Matrix<Variable> &,
                    ADMode = ADMode::REVERSE);


// Jacobian forward mode (Vector)
Matrix<Type> &JacobF(Expression &, const Vector<Variable> &, bool = false);
// Jacobian reverse mode (Vector)
Matrix<Type> &JacobR(Expression &, const Vector<Variable> &);
// Jacobian of expression (Vector)
Matrix<Type> &Jacob(Expression &, const Vector<Variable> &,
                    ADMode = ADMode::REVERSE);


// Symbolic Jacobian of expression (Vector)
Matrix<Expression> &JacobSym(Expression &, const Vector<Variable> &);

// Hessian forward mode
Matrix<Type> &HessF(Expression &, const Vector<Variable> &);

// Hessian reverse mode
Matrix<Type> &HessR(Expression &, const Vector<Variable> &);

// Hessian of expression
Matrix<Type> &Hess(Expression &, const Vector<Variable> &,
                   ADMode = ADMode::REVERSE);

// Symbolic Hessian of expression
Matrix<Expression> &HessSym(Expression &, const Vector<Variable> &);



// Symbolic Expression (Matrix)
Matrix<Expression> &SymMatDiff(Expression &, const Matrix<Variable> &);

// Create new expression
template <typename T,
          typename = std::enable_if_t<std::is_base_of_v<MetaVariable, T>>>
Expression &CreateExpr(const T &exp) {
  auto tmp = Allocate<Expression>(exp);
  return *tmp;
}

// Free function for creating new expression
Expression &CreateExpr(const Type & = 0);

// Create new variable
Variable &CreateVar(const Type &);

// Create new parameter
Parameter &CreateParam(const Type &);

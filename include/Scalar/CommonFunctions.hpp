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

// Main forward mode AD computation
Type DevalF(Expression &, const Variable &);

// Main precomputation of reverse mode AD computation
void PreComp(Expression &);
// Main reverse mode AD computation
Type DevalR(Expression &, const Variable &);

// Derivative of expression
Type Deval(Expression &, const Variable &, ADMode = ADMode::REVERSE);

// Jacobian forward mode
Matrix<Type> &JacobF(Expression &, const Vector<Variable> &);

// Jacobian reverse mode
Matrix<Type> &JacobR(Expression &, const Vector<Variable> &);

// Jacobian of expression
Matrix<Type> &Jacobian(Expression &,
                       const Vector<Variable> &,
                       ADMode = ADMode::REVERSE);

// Symbolic Jacobian of expression
Matrix<Expression> &JacobianSym(Expression &, const Vector<Variable> &);

// Hessian forward mode
Matrix<Type> &HessF(Expression &, const Vector<Variable> &);

// Hessian reverse mode
Matrix<Type> &HessR(Expression &, const Vector<Variable> &);

// Hessian of expression
Matrix<Type> &Hessian(Expression &,
                      const Vector<Variable> &,
                      ADMode = ADMode::REVERSE);

// Symbolic Hessian of expression
Matrix<Expression> &HessianSym(Expression &, const Vector<Variable> &);

// Main reverse mode AD table
OMPair &PreCompCache(const Expression &);

// Symbolic Expression
Expression &SymDiff(Expression &, const Variable &);

// Create new expression
template <typename T,
          typename = std::enable_if_t<std::is_base_of_v<MetaVariable, T>>>
Expression &CreateExpr(const T &exp)
{
    auto tmp = Allocate<Expression>(exp);
    return *tmp;
}

// Free function for creating new expression
Expression &CreateExpr(const Type & = 0);

// Create new variable
Variable &CreateVar(const Type &);

// Create new parameter
Parameter &CreateParam(const Type &);

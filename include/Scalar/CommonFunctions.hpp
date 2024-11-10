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
#include "GenericBinaryC0Function.hpp"
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
#include "GenericUnaryC0Function.hpp"

// Main evaluate expression function
Type EvalExp(Expression&);

// Main forward mode AD computation
Type DevalFExp(Expression&, const Variable&);

// Main reverse mode AD computation
Type DevalRExp(Expression&, const Variable&);

// Main precomputation of reverse mode AD computation
void PreCompExp(Expression&);

// Main reverse mode AD table
OMPair& PreCompCacheExp(const Expression&);

// Symbolic Expression
Expression& SymDiffExp(Expression&, const Variable&);

template <typename T> 
inline constexpr Type Eval(T& v) {
  // If T is Expression
  if constexpr (true == std::is_same_v<Expression, T>) {
    return EvalExp(v);
  }
  // If T is Parameter
  else if constexpr (true == std::is_same_v<T, Parameter>) {
    return v.eval();
  }
  // If T is Variable
  else if constexpr (true == std::is_same_v<T, Variable>) {
    return v.getValue();
  }
  // If T is an expression template
  else if constexpr (true == std::is_base_of_v<MetaVariable, T>) {
    return const_cast<std::decay_t<T>&>(v).eval();
  }
  // If T is Numeric type
  else if constexpr (true == is_numeric_v<T>) {
    return (Type)(v);
  }
  // If unknown type, throw error
  else {
    ASSERT(false, "Unknown type");
  }
}

template <typename T> 
inline constexpr Type DevalF(T& v, const Variable& var) {
  // If T is Expression
  if constexpr (true == std::is_same_v<Expression, T>) {
    return DevalFExp(v, var);
  }
  // If T is Variable
  else if constexpr (true == std::is_same_v<T, Variable>) {
    return ((v.m_nidx == var.m_nidx) ? (Type)(1) : (Type)(0));
  }
  // If T is Parameter or Numeric type
  else if constexpr (true == std::is_same_v<T, Parameter> || true == is_numeric_v<T>) {
    return (Type)(0);
  }
  // If T is an expression template
  else if constexpr (true == std::is_base_of_v<MetaVariable, T>) {
    return const_cast<std::decay_t<T>&>(v).devalF(var);
  }
  // If unknown type, throw error
  else {
    ASSERT(false, "Unknown type");
  }
}

template <typename T> 
inline constexpr void PreComp(T& v) {
  // If T is Expression
  if constexpr (true == std::is_same_v<T, Expression>) {
    PreCompExp(v);
  }
  // If T is an expression template
  else if constexpr (true == std::is_base_of_v<MetaVariable, T>) {
    const_cast<std::decay_t<T> &>(v).reset();
    const_cast<std::decay_t<T> &>(v).traverse();
  }
  // If unknown type, throw error
  else {
    ASSERT(false, "Unknown type");
  }
}

template <typename T> 
inline constexpr Type DevalR(T& v, const Variable& var) {
  // If T is Expression
  if constexpr (true == std::is_same_v<Expression, T>) {
    return DevalRExp(v, var);
  }
  // If T is Variable
  else if constexpr (true == std::is_same_v<T, Variable>) {
    return ((v.m_nidx == var.m_nidx) ? (Type)(1) : (Type)(0));
  }
  // If T is Parameter or Numeric type
  else if constexpr (true == std::is_same_v<T, Parameter> || true == is_numeric_v<T>) {
    return (Type)(0);
  }
  // If T is an expression template
  else if constexpr (true == std::is_base_of_v<MetaVariable, T>) {
    PreComp(v);
    return const_cast<std::decay_t<T>&>(v).getCache()[var.m_nidx];
  }
  // If unknown type, throw error
  else {
    ASSERT(false, "Unknown type");
  }
}

template <typename T> 
inline constexpr OMPair& PreCompCache(const T& v) {
  if constexpr (true == std::is_same_v<Expression, T>) {
    return PreCompCacheExp(v);
  }
  // If T is an expression template
  else if constexpr (true == std::is_base_of_v<MetaVariable, T>) {
    const_cast<std::decay_t<T>&>(v).reset();
    const_cast<std::decay_t<T>&>(v).traverse();
    return const_cast<std::decay_t<T>&>(v).getCache();
  }
  // If unknown type, throw error
  else {
    ASSERT(false, "Unknown type");
  }
}

template<typename T>
inline constexpr Expression& SymDiff(T& v, const Variable& var) {
  // If T is Expression
  if constexpr (true == std::is_same_v<Expression, T>) {
    return SymDiffExp(v, var);
  } 
  // If T is Variable
  else if constexpr (true == std::is_same_v<T, Variable>) {
    return ((v.m_nidx == var.m_nidx) ? Expression::t1 : Expression::t0);
  }
  // If T is Parameter or Numeric type
  else if constexpr (true == std::is_same_v<T, Parameter> || true == is_numeric_v<T>) {
    return Expression::t0;
  }
  // If T is an expression template
  else if constexpr (true == std::is_base_of_v<MetaVariable, T>) {
    return const_cast<std::decay_t<T>&>(v).symDeval(var);
  }
  // If unknown type, throw error
  else {
    ASSERT(false, "Unknown type");
  }
}


// Forward mode algorithmic differentiation (Matrix)
Matrix<Type> &DevalF(Expression &, const Matrix<Variable> &, bool = false);
// Reverse mode algorithmic differentiation (Matrix)
Matrix<Type> &DevalR(Expression &, const Matrix<Variable> &);
void DevalR(Expression &, const Matrix<Variable> &, Matrix<Type> *&);

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

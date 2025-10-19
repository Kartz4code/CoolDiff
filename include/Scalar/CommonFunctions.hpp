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
#include "GenericScalarBinary.hpp"
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
#include "GenericScalarUnary.hpp"

/* Custom functions (Unary operations) */
// Sin scalar 
UNARY_SCALAR_OPERATOR(SinS, [](Type a) { return std::sin(a); },
                            [](Type b) { return std::cos(b); })
// Cos scalar
UNARY_SCALAR_OPERATOR(CosS, [](Type a) { return std::cos(a); },
                            [](Type b) { return -std::sin(b); })

/* Custom functions (Binary operations) */
// Product scalar
BINARY_SCALAR_OPERATOR(ProductS,  [](const Type u, const Type v) { return u * v; },
                                  [](const Type u, const Type v) { return v; },
                                  [](const Type u, const Type v) { return u; })

// Namespace CoolDiff
namespace CoolDiff {
  // Namespace Scalar
  namespace TensorR1 {
    // Namespace Details
    namespace Details {
      // Main evaluate expression function
      Type EvalExp(Expression&);

      // Main forward mode AD computation
      Type DevalFExp(Expression&, const Variable&);

      // Main reverse mode AD computation
      Type DevalRExp(Expression&, const Variable&);

      // Main precomputation of reverse mode AD computation
      void PreCompExp(Expression&);

      // Main reverse mode AD table
      OMPair& PreCompCacheExp(Expression&);

      // Symbolic Expression
      Expression& SymDiffExp(Expression&, const Variable&);
    }

    // TODO - Modify code to accomodate changes made to the clone function 
    template <typename T, typename = CoolDiff::TensorR1::Details::IsValidScalarType<T>> 
    inline Type Eval(T& v) {
      // If T is Expression
      if constexpr (true == std::is_same_v<Expression, T>) {
        return CoolDiff::TensorR1::Details::EvalExp(v);
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
        Expression exp{v};
        return CoolDiff::TensorR1::Details::EvalExp(exp);
      }
      // If T is Numeric type
      else if constexpr (true == CoolDiff::TensorR1::Details::is_numeric_v<T>) {
        return (Type)(v);
      }
      // If unknown type, throw error
      else {
        ASSERT(false, "Unknown type");
        return (Type)(0);
      }
    }

    template <typename T, typename = CoolDiff::TensorR1::Details::IsValidScalarType<T>> 
    inline Type DevalF(T& v, const Variable& var) {
      // If T is Expression
      if constexpr (true == std::is_same_v<Expression, T>) {
        return CoolDiff::TensorR1::Details::DevalFExp(v, var);
      }
      // If T is Variable
      else if constexpr (true == std::is_same_v<T, Variable>) {
        return ((v.m_nidx == var.m_nidx) ? (Type)(1) : (Type)(0));
      }
      // If T is Parameter or Numeric type
      else if constexpr (true == std::is_same_v<T, Parameter> || true == CoolDiff::TensorR1::Details::is_numeric_v<T>) {
        return (Type)(0);
      }
      // If T is an expression template
      else if constexpr (true == std::is_base_of_v<MetaVariable, T>) {
        Expression exp{v};
        return CoolDiff::TensorR1::Details::DevalFExp(exp, var);
      }
      // If unknown type, throw error
      else {
        ASSERT(false, "Unknown type");
      }
    }

    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>> 
    inline void PreComp(T& v) {
      CoolDiff::TensorR1::Details::PreCompExp(v);
    }

    template <typename T, typename = CoolDiff::TensorR1::Details::IsValidScalarType<T>> 
    inline Type DevalR(T& v, const Variable& var) {
      // If T is Expression
      if constexpr (true == std::is_same_v<Expression, T>) {
        return CoolDiff::TensorR1::Details::DevalRExp(v, var);
      }
      // If T is Variable
      else if constexpr (true == std::is_same_v<T, Variable>) {
        return ((v.m_nidx == var.m_nidx) ? (Type)(1) : (Type)(0));
      }
      // If T is Parameter or Numeric type
      else if constexpr (true == std::is_same_v<T, Parameter> || true == CoolDiff::TensorR1::Details::is_numeric_v<T>) {
        return (Type)(0);
      }
      // If unknown type, throw error
      else {
        ASSERT(false, "Unknown type. Not an expression type");
      }
    }

    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Expression>>> 
    inline OMPair& PreCompCache(T& v) {
      return CoolDiff::TensorR1::Details::PreCompCacheExp(v);
    }

    template <typename T, typename = CoolDiff::TensorR1::Details::IsValidScalarType<T>> 
    inline Expression& SymDiff(T& v, const Variable& var) {
      // If T is Expression
      if constexpr (true == std::is_same_v<Expression, T>) {
        return CoolDiff::TensorR1::Details::SymDiffExp(v, var);
      }
      // If T is Variable
      else if constexpr (true == std::is_same_v<T, Variable>) {
        return ((v.m_nidx == var.m_nidx) ? Expression::t1 : Expression::t0);
      }
      // If T is Parameter or Numeric type
      else if constexpr (true == std::is_same_v<T, Parameter> || true == CoolDiff::TensorR1::Details::is_numeric_v<T>) {
        return Expression::t0;
      }
      // If T is an expression template
      else if constexpr (true == std::is_base_of_v<MetaVariable, T>) {
        auto exp = Allocate<Expression>(const_cast<std::decay_t<T>&>(v));
        return CoolDiff::TensorR1::Details::SymDiffExp(*exp, var);
      }
      // If unknown type, throw error
      else {
        ASSERT(false, "Unknown type");
      }
    }
  }
}


/*

// Jacobian forward mode (Vector)
Matrix<Type> &JacobF(Expression &, const Vector<Variable> &, bool = false);

// Hessian forward mode
Matrix<Type> &HessF(Expression &, const Vector<Variable> &);


// Jacobian of expression (Vector)
Matrix<Type> &Jacob(Expression &, const Vector<Variable> &,
                    ADMode = ADMode::REVERSE);

// Hessian of expression
Matrix<Type> &Hess(Expression &, const Vector<Variable> &,
                   ADMode = ADMode::REVERSE);
*/
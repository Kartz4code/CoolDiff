/**
 * @file include/Expression.hpp
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

#include "GenericProduct.hpp"
#include "Parameter.hpp"
#include "Variable.hpp"

class Expression : public Variable {
private:
  // Is expression recursive?
  bool m_recursive_exp{false};

public:
  // Default constructor
  Expression();

  template <typename T> 
  constexpr Expression(const IVariable<T>& expr) {
    Variable::m_nidx = this->m_idx_count++;
    // Reserve a buffer of expressions
    Variable::m_gh_vec.reserve(g_vec_init);
    // Emplace the expression in a generic holder
    if constexpr (true == std::is_same_v<T, Expression>) {
      Variable::m_gh_vec.push_back((Expression*)(&expr));
    } else {
      Variable::m_gh_vec.push_back((Expression*)(&(expr * (Type)(1))));
    }
  }

  // Copy assignment for expression evaluation - e.g.Variable x = x1 + x2 + x3;
  template <typename T> 
  Expression& operator=(const IVariable<T>& expr) {
    if (auto rec = static_cast<const T&>(expr).findMe(this); rec == false) {
      m_gh_vec.clear();
    } else {
      m_recursive_exp = rec;
    }

    // Emplace the expression in a generic holder
    if constexpr (true == std::is_same_v<T, Expression>) {
      Variable::m_gh_vec.push_back((Expression*)(&expr));
    } else {
      Variable::m_gh_vec.push_back((Expression*)(&(expr * (Type)(1))));
    }
    return *this;
  }

  // Is recursive expression
  bool isRecursive() const;

  // Symbolic differentiation of expression
  Expression& SymDiff(const Variable&);

  // Expression factory
  class ExpressionFactory {
  public:
    template <typename T, typename = std::enable_if_t<is_valid_v<T>>>
    constexpr inline static Expression& CreateExpression(const T& exp) {
      if constexpr (true == is_numeric_v<T>) {
        auto tmp = Allocate<Expression>(*Allocate<Parameter>(exp)).get();
        return *tmp;
      } else {
        auto tmp = Allocate<Expression>(exp);
        return *tmp;
      }
    }
  };

  // Get type
  V_OVERRIDE(std::string_view getType() const);

  V_DTR(~Expression());
};

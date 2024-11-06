/**
 * @file include/IVariable.hpp
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

#include "MemoryManager.hpp"
#include "MetaVariable.hpp"

// IVariable class to enforce expression templates for lazy evaluation
template <typename T> 
class IVariable : public MetaVariable {
protected:
  // Protected constructor
  IVariable() = default;
  // Protected destructor
  V_DTR(~IVariable()) = default;
};

// Binary left reset
#define BINARY_LEFT_RESET()                                                    \
  this->m_visited = false;                                                     \
  if (false == m_cache.empty()) {                                              \
    m_cache.clear();                                                           \
  }                                                                            \
  MetaVariable::resetTemp();                                                   \
  mp_left->reset();

// Binary right reset
#define BINARY_RIGHT_RESET()                                                   \
  this->m_visited = false;                                                     \
  if (false == m_cache.empty()) {                                              \
    m_cache.clear();                                                           \
  }                                                                            \
  MetaVariable::resetTemp();                                                   \
  mp_right->reset();

// Binary reset
#define BINARY_RESET()                                                         \
  this->m_visited = false;                                                     \
  if (false == m_cache.empty()) {                                              \
    m_cache.clear();                                                           \
  }                                                                            \
  MetaVariable::resetTemp();                                                   \
  mp_left->reset();                                                            \
  mp_right->reset();

// Unary reset
#define UNARY_RESET()                                                          \
  this->m_visited = false;                                                     \
  if (false == m_cache.empty()) {                                              \
    m_cache.clear();                                                           \
  }                                                                            \
  MetaVariable::resetTemp();                                                   \
  if (nullptr != mp_left) {                                                    \
    mp_left->reset();                                                          \
  }

// Operations enum [Order matters!]
enum Op : size_t {
  ADD = 0, MUL, SUB, DIV, POW,
  SIN, COS, TAN, SINH, COSH,
  TANH, ASIN, ACOS, ATAN, ASINH,
  ACOSH, ATANH, SQRT, EXP, LOG,
  COUNT
};

// Policy based design (Scalar doesn't need one, yet!)
struct __XOXO__ {};
#define OpType __XOXO__
#define OpObj OpType()

// Numeric here indicates both arithmetic and Type
template <typename T>
constexpr inline static const bool is_numeric_v = (std::is_arithmetic_v<T> || std::is_same_v<T, Type>);

// Is a valid type for operator()
template <typename T>
constexpr inline static const bool is_valid_v = (std::is_base_of_v<MetaVariable, T> || is_numeric_v<T>);
/**
 * @file include/Variable.hpp
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

#include "IVariable.hpp"
#include "VarWrap.hpp"

// Variable Expression is a wrapper around the variable class
class Variable : public IVariable<Variable> {
protected:
  // Underlying symbolic variable
  VarWrap m_var{};

  // Static variable for one seed
  static Variable t1;
  // Static variable for zero seed
  static Variable t0;

  // Friend class Parameter
  friend class Parameter;

  // Type real-time value
  SharedPtr<Type> m_value_var{std::make_shared<Type>()};

  // Collection of meta variable expressions
  Vector<MetaVariable *> m_gh_vec{};

  // Exposed to user to compute symbolic differentiation
  Expression SymDiff(const Variable &);

public:
  // Block index
  size_t m_nidx{};
  // Cache for reverse 1st AD
  OMPair m_cache{};

  // Constructors
  Variable();
  // Copy constructor
  Variable(const Variable &);
  // Move constructor
  Variable(Variable &&) noexcept;
  // Copy assignment from one variable to another variable
  Variable &operator=(const Variable &);
  // Move assignment from one variable to another variable
  Variable &operator=(Variable &&) noexcept;

  // Constructors for Type values
  Variable(const Type &);
  // Assignment to Type
  Variable &operator=(const Type &);

  // Reset impl
  void resetImpl();

  // Deval in run-time for reverse derivative
  Type devalR(const Variable &);

  // Getters and setters
  // Get/Set value
  void setValue(const Type &);
  Type getValue() const;

  // Get/Set dvalue
  void setdValue(const Type &);
  Type getdValue() const;

  // Evaluate value and derivative value in run-time
  V_OVERRIDE(Type eval());
  // Evaluate 1st derivative in forward mode
  V_OVERRIDE(Type devalF(const Variable &));

  // Traverse tree
  V_OVERRIDE(void traverse(OMPair * = nullptr));
  // Get the map of derivatives
  V_OVERRIDE(OMPair &getCache());

  // Evaluate variable and its derivative value in run-time
  V_OVERRIDE(Variable *symEval());
  V_OVERRIDE(Variable *symDeval(const Variable &));

  // Reset all visited flags
  V_OVERRIDE(void reset());

  // Get type
  V_OVERRIDE(std::string_view getType() const);

  // Find me
  V_OVERRIDE(bool findMe(void *) const);

  // Variable factory
  class VariableFactory {
  public:
    // Create new variable
    static Variable &CreateVariable(const Type & = (Type)(0));
  };

  // Destructor
  V_DTR(~Variable());
};

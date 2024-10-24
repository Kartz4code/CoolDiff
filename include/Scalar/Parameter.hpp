/**
 * @file include/Parameter.hpp
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

// Parameter class for l-value data variables
class Parameter : public IVariable<Parameter> {
public:
  // Block index
  size_t m_nidx{};
  // Cache for reverse AD
  OMPair m_cache{};

  // Constructors
  Parameter();

  // Constructors for Type values
  Parameter(const Type &);

  // Copy constructor
  Parameter(const Parameter &);

  // Move constructor
  Parameter(Parameter &&) noexcept;

  // Copy assignment
  Parameter &operator=(const Parameter &);

  // Move assignment
  Parameter &operator=(Parameter &&) noexcept;

  // Assignment to Type
  Parameter &operator=(const Type &);

  // Deval in run-time for reverse derivative
  Type devalR(const Variable &);

  // Evaluate variable and its derivative value in run-time
  V_OVERRIDE(Variable *symEval());
  V_OVERRIDE(Variable *symDeval(const Variable &));

  // Evaluate value and derivative value in run-time
  V_OVERRIDE(Type eval());

  // Reset all visited flag
  V_OVERRIDE(void reset());

  // Evaluate derivative in forward mode
  V_OVERRIDE(Type devalF(const Variable &));

  // Traverse tree
  V_OVERRIDE(void traverse(OMPair * = nullptr));

  // Get the map of derivatives
  V_OVERRIDE(OMPair &getCache());

  // Get type
  V_OVERRIDE(std::string_view getType() const);

  // Find me
  V_OVERRIDE(bool findMe(void *) const);

  // Parameter factory
  class ParameterFactory {
  public:
    // Create new parameter
    static Parameter &CreateParameter(const Type & = (Type)(0));
  };

  // Destructor
  V_DTR(~Parameter() = default);
};

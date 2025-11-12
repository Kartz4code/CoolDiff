/**
 * @file src/Scalar/Parameter.cpp
 *
 * @copyright 2023-2025 Karthik Murali Madhavan Rathai
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

#include "Parameter.hpp"
#include "Variable.hpp"

// Swap for assignment operator
void Parameter::swap(Parameter& other) noexcept {
  std::swap(m_nidx, other.m_nidx);
  std::swap(m_cache, other.m_cache);
}

// Constructors
Parameter::Parameter() : m_nidx{this->m_idx_count++} {
  auto tmp = Allocate<Variable>((Type)0);
  this->mp_tmp = tmp.get();
}

// Constructors for Type values
Parameter::Parameter(const Type& value) : m_nidx{this->m_idx_count++} {
  auto tmp = Allocate<Variable>(value);
  this->mp_tmp = tmp.get();
}

// Copy constructor
Parameter::Parameter(const Parameter& s) :  m_nidx{s.m_nidx}, 
                                            m_cache{s.m_cache} {
  this->mp_tmp = s.mp_tmp;
}

// Move constructor
Parameter::Parameter(Parameter&& s) noexcept :  m_nidx{std::exchange(s.m_nidx, -1)}, 
                                                m_cache{std::move(s.m_cache)} {
  this->mp_tmp = std::exchange(s.mp_tmp, nullptr);
}

// Copy assignment
Parameter& Parameter::operator=(const Parameter& s) {
  Parameter{s}.swap(*this);
  return *this;
}

// Move assignment
Parameter& Parameter::operator=(Parameter&& s) noexcept {
  Parameter{std::move(s)}.swap(*this);
  return *this;
}

// Assignment to Type
Parameter& Parameter::operator=(const Type& value) {
  *this->mp_tmp = value;
  return *this;
}

// Evaluate value and derivative value
Type Parameter::eval() { 
  return this->mp_tmp->getValue(); 
}

void Parameter::reset() {
  // Reset temporaries
  MetaVariable::resetTemp();
  return;
}

// Evaluate paramter
Variable* Parameter::symEval() { 
  return this->mp_tmp; 
}

// Forward derivative of paramter in forward mode
Variable* Parameter::symDeval(const Variable& var) {
  // Static variable for zero seed
  return &Variable::t0;
}

// Evaluate derivative in forward mode
Type Parameter::devalF(const Variable&) { 
  return (Type)(0); 
}

// Deval in run-time for reverse derivative
Type Parameter::devalR(const Variable&) { 
  return (Type)(0); 
}

// Traverse tree
void Parameter::traverse(OMPair*) { 
  return; 
}

// Get the map of derivatives
OMPair& Parameter::getCache() { 
  return m_cache; 
}

// Get type
std::string_view Parameter::getType() const { 
  return "Parameter"; 
}

// Create new parameter
Parameter& Parameter::ParameterFactory::CreateParameter(const Type& val) {
  auto tmp = Allocate<Parameter>(val);
  return *tmp;
}

// Find me
bool Parameter::findMe(void* v) const {
  if (static_cast<const void*>(this) == v) {
    return true;
  } else {
    return false;
  }
}

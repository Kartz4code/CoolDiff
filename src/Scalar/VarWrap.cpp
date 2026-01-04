/**
 * @file src/Scalar/VarWrap.cpp
 *
 * @copyright 2023-2026 Karthik Murali Madhavan Rathai
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

#include "VarWrap.hpp"

// Copy-swap idiom 
void VarWrap::swap(VarWrap& vw) noexcept {
  std::swap(m_value, vw.m_value);
  std::swap(m_dvalue, vw.m_dvalue);
}

VarWrap::VarWrap(Type val) :  m_value{val}, 
                              m_dvalue{(Type)(0)} 
{}

VarWrap::VarWrap() :  m_value{(Type)(0)}, 
                      m_dvalue{(Type)(0)} 
{}

// Copy constructor
VarWrap::VarWrap(const VarWrap& vw) : m_value{vw.m_value}, 
                                      m_dvalue{vw.m_dvalue} 
{}

// Move constructor
VarWrap::VarWrap(VarWrap&& vw) noexcept : m_value{std::exchange(vw.m_value, {})}, 
                                          m_dvalue{std::exchange(vw.m_dvalue, {})} 
{}

// Copy assignment
VarWrap& VarWrap::operator=(const VarWrap& vw) {
  VarWrap{vw}.swap(*this);
  return *this;
}

// Move assignment
VarWrap& VarWrap::operator=(VarWrap&& vw) noexcept {
  VarWrap{std::move(vw)}.swap(*this);
  return *this;
}

// Set constructor
void VarWrap::setConstructor(const Type& val) {
  m_value = {val};
  m_dvalue = {(Type)(0)};
}

const Type VarWrap::getValue() const { 
  return m_value; 
}

void VarWrap::setValue(const Type& val) { 
  m_value = val; 
}

const Type VarWrap::getdValue() const { 
  return m_dvalue; 
}

void VarWrap::setdValue(const Type& val) { 
  m_dvalue = val; 
}

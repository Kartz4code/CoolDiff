/**
 * @file src/Parameter.cpp
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

#include "Parameter.hpp"
#include "Variable.hpp"

// Constructors
Parameter::Parameter() : m_nidx{this->m_idx_count++}, m_value{(Type)0}
{
    auto tmp = Allocate<Variable>((Type)0);
    this->mp_tmp = tmp.get();
}

// Constructors for Type values
Parameter::Parameter(const Type &value)
    : m_nidx{this->m_idx_count++}, m_value{value}
{
    auto tmp = Allocate<Variable>(m_value);
    this->mp_tmp = tmp.get();
}

// Copy constructor
Parameter::Parameter(const Parameter &s)
    : m_nidx{this->m_idx_count++}, m_value{s.m_value}
{
    this->mp_tmp = s.mp_tmp;
}

// Copy assignment
Parameter &Parameter::operator=(const Parameter &s)
{
    m_value = s.m_value;
    this->mp_tmp = s.mp_tmp;
    return *this;
}

// Assignment to Type
Parameter &Parameter::operator=(const Type &value)
{
    m_value = value;
    *this->mp_tmp = value;
    return *this;
}

// Evaluate value and derivative value
Type Parameter::eval()
{
    return m_value;
}

void Parameter::reset()
{
    // Reset temporaries
    MetaVariable::resetTemp();
    return;
}

// Evaluate paramter
Variable *Parameter::symEval()
{
    return this->mp_tmp;
}

// Forward derivative of paramter in forward mode
Variable *Parameter::symDeval(const Variable &var)
{
    // Static variable for zero seed
    return &Variable::t0;
}

// Evaluate derivative in forward mode
Type Parameter::devalF(const Variable &)
{
    return (Type)0;
}

// Deval in run-time for reverse derivative
Type Parameter::devalR(const Variable &)
{
    return (Type)0;
}

// Traverse tree
void Parameter::traverse(OMPair *)
{
    return;
}

// Get the map of derivatives
OMPair &Parameter::getCache()
{
    return m_cache;
}

// Get type
std::string_view Parameter::getType() const
{
    return "Parameter";
}

// Find me
bool Parameter::findMe(void *v) const
{
    if (static_cast<const void *>(this) == v)
    {
        return true;
    }
    else
    {
        return false;
    }
}

Parameter::~Parameter() = default;

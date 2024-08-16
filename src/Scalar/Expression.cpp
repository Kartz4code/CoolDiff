/**
 * @file src/Scalar/Expression.cpp
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

#include "Expression.hpp"

Expression::Expression()
{
    Variable::m_nidx = this->m_idx_count++;
    // Reserve a buffer of expressions
    Variable::m_gh_vec.reserve(g_vec_init);
    // Emplace the expression in a generic holder
    Variable::m_gh_vec.push_back(&Variable::t0);
}

Expression::Expression(const Expression &exp)
{
    Variable::m_gh_vec.emplace_back(&exp);
}

Expression &Expression::SymDiff(const Variable &var)
{
    auto tmp = Allocate<Expression>();
    *tmp = Variable::SymDiff(var);
    return *tmp;
}

// Get type
std::string_view Expression::getType() const
{
    return "Expression";
}

Expression::~Expression() = default;

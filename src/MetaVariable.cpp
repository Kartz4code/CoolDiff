/**
 * @file src/MetaVariable.cpp
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

#include "MetaVariable.hpp"
#include "Variable.hpp"

// Reset temporaries
void MetaVariable::resetTemp()
{
    if (this->mp_tmp != nullptr)
    {
        this->mp_tmp->reset();
    }
    for (auto &[k, v] : this->mp_dtmp)
    {
        if (v != nullptr)
        {
            v->reset();
        }
    }
    this->m_visited = false;
}
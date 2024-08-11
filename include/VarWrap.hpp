/**
 * @file include/VarWrap.hpp
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

#include "CommonHeader.hpp"

class VarWrap {
private:
    // Count the number of VarWrap
    inline static size_t m_count{};

    // Resources
    std::string m_expression{};
    Type m_value{}, m_dvalue{};
    std::string m_var_name{};

public:
    // Constructors
    VarWrap();
    VarWrap(Type);
    VarWrap(const VarWrap&);
    VarWrap(VarWrap&&) noexcept;

    // Assignments
    VarWrap& operator=(const VarWrap&);
    VarWrap& operator=(VarWrap&&) noexcept;

    // Set constructor
    void setConstructor(Type);

    // Getters and setters methods
    // Get/Set string
    const std::string& getVariableName() const;
    void setString(const std::string&);

    // Get/Set expression
    void setExpression(const std::string&);
    const std::string& getExpression() const;

    // Get/Set value
    const Type getValue() const;
    void setValue(Type);

    // Get/Set dvalue
    const Type getdValue() const;
    void setdValue(Type);

    // Destructor
    ~VarWrap() = default;
};

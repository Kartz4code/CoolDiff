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
  // Resources
  Type m_value{}, m_dvalue{};

public:
  // Constructors
  VarWrap();
  VarWrap(Type);
  VarWrap(const VarWrap &);
  VarWrap(VarWrap &&) noexcept;

  // Assignments
  VarWrap &operator=(const VarWrap &);
  VarWrap &operator=(VarWrap &&) noexcept;

  // Set constructor
  void setConstructor(const Type&);

  // Getters and setters methods
  // Get/Set value
  const Type getValue() const;
  void setValue(const Type&);

  // Get/Set dvalue
  const Type getdValue() const;
  void setdValue(const Type&);

  // Destructor
  ~VarWrap() = default;
};

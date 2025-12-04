/**
 * @file example/GaussNewton/include/Oracle.hpp
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

#pragma once

#include "CoolDiff.hpp"

// Oracle base class
class Oracle {
private:
  inline static Vector<SharedPtr<Oracle>> m_oracle_factory;

public:
  Oracle() = default;

  class OracleFactory {
  public:
    static Oracle* CreateOracle(Expression&, Matrix<Variable>&);
    static Oracle* CreateOracle(Matrix<Expression>&, Matrix<Variable>&);

    ~OracleFactory() = default;
  };

  // Oracle functions
  V_PURE(Type eval());
  V_PURE(Matrix<Type>* evalMat());
  V_PURE(Matrix<Type>* jacobian());
  V_PURE(const size_t getVariableSize() const);
  V_PURE(std::string_view getOracleType() const);

  V_DTR(~Oracle());
};
/**
 * @file example/GaussNewton/src/Oracle.cpp
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

#include "OracleMatrix.hpp"
#include "OracleScalar.hpp"

Oracle *Oracle::OracleFactory::CreateOracle(Expression &exp,
                                            Matrix<Variable> &X) {
  auto it = SharedPtr<Oracle>(new OracleScalar(exp, X));
  m_oracle_factory.push_back(it);
  return it.get();
}

Oracle *Oracle::OracleFactory::CreateOracle(Matrix<Expression> &exp,
                                            Matrix<Variable> &X) {
  auto it = SharedPtr<Oracle>(new OracleMatrix(exp, X));
  m_oracle_factory.push_back(it);
  return it.get();
}

Oracle::~Oracle() = default;
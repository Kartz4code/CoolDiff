/**
 * @file example/GaussNewton/include/OracleScalar.hpp
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

#include "Oracle.hpp"

// Oracle scalar class
class OracleScalar : public Oracle {
private:
  friend class Oracle;
  friend class Oracle::OracleFactory;

  // Dimension of variable vector
  size_t m_dim{};
  // Vector of variables
  Vector<Variable> m_vec{};
  // Hessian pointer
  Matrix<Type>* m_hessian{nullptr};
  // Jacobian pointer
  Matrix<Type>* m_jacobian{nullptr};
  
  // Expression
  Expression& m_exp;
  // Vector of Jacobian expressions (For Hessian computation)
  Matrix<Expression> m_jacobian_sym;

  OracleScalar(Expression&, const Matrix<Variable>&);
  OracleScalar(Expression&, const Vector<Variable>&);

  // Symbolic Jacobian for Hessian computation 
  template<typename T>
  void symJacob(T& exp) {
    for (size_t i{}; i < m_dim; ++i) {
      m_jacobian_sym[i] = SymDiff(exp, m_vec[i]);                                              
    } 
  }

public:
  // Oracle functions
  V_OVERRIDE( Type eval() );
  V_OVERRIDE( Matrix<Type>* evalMat() );
  V_OVERRIDE( Matrix<Type>* jacobian() );
  V_OVERRIDE( const size_t getVariableSize() const );
  V_OVERRIDE( std::string_view getOracleType() const );

  // Get Hessian
  Matrix<Type>* hessian();
  // Get variables
  const Vector<Variable>& getVariables() const;
  Vector<Variable>& getVariables();

  virtual ~OracleScalar() = default;
};

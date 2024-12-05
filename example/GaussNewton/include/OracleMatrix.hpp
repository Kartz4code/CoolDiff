/**
 * @file example/GaussNewton/include/OracleMatrix.hpp
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

// OracleMatrix class
class OracleMatrix : public Oracle {
  private:
    friend class Oracle;
    friend class Oracle::OracleFactory;

    OracleMatrix(Matrix<Expression>&, const Matrix<Variable>&);
    
    // Dimension of variable vector
    size_t m_dim{};
    // Matrix expression
    Matrix<Expression>& m_exp;
    // Matrix variable
    Matrix<Variable> m_X;

  public:
    
    // Oracle functions
    V_OVERRIDE( Type eval() );
    V_OVERRIDE( Matrix<Type>* evalMat() );
    V_OVERRIDE( Matrix<Type>* jacobian() );
    V_OVERRIDE( const size_t getVariableSize() const );
    V_OVERRIDE( std::string_view getOracleType() const );

    // Get variables
    const Matrix<Variable>& getVariables() const;

    virtual ~OracleMatrix() = default;
};
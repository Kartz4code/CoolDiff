/**
 * @file example/GaussNewton/src/OracleMatrix.cpp
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

OracleMatrix::OracleMatrix(Matrix<Expression>& exp, const Matrix<Variable>& X) : m_dim{ X.getNumElem() },
                                                                                 m_exp{ exp },
                                                                                 m_X{ X } {
    ASSERT(m_exp.getFinalNumColumns() == 1, "Error function is not a column vector");
    ASSERT(m_X.getNumRows() == 1, "Variable matrix is not a row vector");
}

// Oracle functions
Type OracleMatrix::eval() {
    ASSERT(false, "OracleMatrix cannot evaluate to scalar");
    return (Type)(0);
}

Matrix<Type>* OracleMatrix::evalMat() {
    return &Eval(m_exp);
}

Matrix<Type>* OracleMatrix::jacobian() {
    return &DevalF(m_exp, m_X);
}

std::string_view OracleMatrix::getOracleType() const {
    return "OracleMatrix";
}

// Get variables
const Matrix<Variable>& OracleMatrix::getVariables() const {
    return m_X;
}

 Matrix<Variable>& OracleMatrix::getVariables() {
    return m_X;
 }

// Get variable size
const size_t OracleMatrix::getVariableSize() const {
    return m_dim;
}
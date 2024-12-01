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
    ASSERT(m_exp.getFinalNumElem() == 1, "The loss function is non-scalar");
    ASSERT(m_X.getNumColumns() == 1, "The variable matrix is not in a column vector form");
}

// Oracle functions
Type OracleMatrix::eval() {
    return Eval(m_exp)[0];
}

Matrix<Type>& OracleMatrix::jacobian() {
    return DevalF(m_exp, m_X);
}

// Get variables
const Matrix<Variable>& OracleMatrix::getVariables() const {
    return m_X;
}

// Get variable size
const size_t OracleMatrix::getVariableSize() const {
    return m_dim;
}
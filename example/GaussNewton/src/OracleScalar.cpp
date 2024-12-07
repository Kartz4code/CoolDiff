/**
 * @file example/GaussNewton/src/OracleScalar.cpp
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

#include "OracleScalar.hpp"


OracleScalar::OracleScalar(Expression& exp, const Matrix<Variable>& X) : m_dim{ X.getNumElem() },
                                                                         m_exp{ exp },
                                                                         m_vec{ X.getMatrixPtr(), X.getMatrixPtr() + X.getNumElem() } 
{}

OracleScalar::OracleScalar(Expression& exp, const Vector<Variable>& vec) : m_dim{ vec.size() },
                                                                           m_exp{ exp },
                                                                           m_vec{ vec } 
{}

Type OracleScalar::eval() {
    return Eval(m_exp);
}

Matrix<Type>* OracleScalar::evalMat() {
    ASSERT(false, "OracleScalar cannot evaluate to Matrix");
    return nullptr;
}

Matrix<Type>* OracleScalar::hessian() {
    if(nullptr == m_hessian) {
        m_hessian = Matrix<Type>::MatrixFactory::CreateMatrixPtr(m_dim, m_dim);
        symJacob(m_exp);
    }
    // Exploit Hessian symmetry
    for (size_t i{}; i < m_dim; ++i) {
        // Precompute
        PreComp(m_jacobian_sym[i]);
        for (size_t j{}; j <= i; ++j) {
            if (j < i) {
                (*m_hessian)(i, j) = DevalR(m_jacobian_sym[i], m_vec[j]);
                (*m_hessian)(j, i) = (*m_hessian)(i, j);
            } else if (i == j) {
                (*m_hessian)(i, j) = DevalR(m_jacobian_sym[i], m_vec[j]);
            }
        }
    }
    return m_hessian;
}

Matrix<Type>* OracleScalar::jacobian() {
    if(nullptr == m_jacobian) {
        m_jacobian = Matrix<Type>::MatrixFactory::CreateMatrixPtr(m_dim, 1);
    }
    // Precompute (By design, the operation is serial)
    PreComp(m_exp);
    std::transform(EXECUTION_SEQ m_vec.cbegin(), m_vec.cend(), 
                                 m_jacobian->getMatrixPtr(),
                                 [this](const auto &v) { return DevalR(m_exp, v); });

    return m_jacobian;
}

std::string_view OracleScalar::getOracleType() const {
    return "OracleScalar";
}


// Get variables
const Vector<Variable>& OracleScalar::getVariables() const {
    return m_vec;
}

Vector<Variable>& OracleScalar::getVariables() {
    return m_vec;
}

// Get variable size
const size_t OracleScalar::getVariableSize() const {
    return m_dim;
}

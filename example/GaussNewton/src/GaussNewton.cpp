/**
 * @file example/GaussNewton/src/GaussNewton.cpp
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

#include "GaussNewton.hpp"

// Set data
void GaussNewton::setData(const size_t i) {
    for(size_t j{}; j < m_X->getNumColumns(); ++j) {
        (*m_PX)[j] = (*m_X)(i,j); 
    }
    for(size_t j{}; j < m_Y->getNumColumns(); ++j) {
        (*m_PY)[j] = (*m_Y)(i,j);
    }
}

// Get A,B for matrix solve
void GaussNewton::computeABScalar() {
    const size_t var_size = static_cast<OracleScalar*>(m_oracle)->getVariableSize();
    // If m_A is a nullptr
    if(nullptr == m_A) {
        m_A = Matrix<Type>::MatrixFactory::CreateMatrixPtr(var_size, var_size);
    }

    // If m_B is a nullptr
    if(nullptr == m_B) {
        m_B = Matrix<Type>::MatrixFactory::CreateMatrixPtr(var_size, 1);
    }

    // Reset m_A and m_B
    ResetZero(m_A);
    ResetZero(m_B);

    for(size_t i{}; i < m_size; ++i) {
        // Set data
        setData(i);

        // Eval and jacobian
        const Type eval = static_cast<OracleScalar*>(m_oracle)->eval();
        const Matrix<Type>* jacobian = static_cast<OracleScalar*>(m_oracle)->jacobian();

        // Compute A matrix
        MatrixTranspose(jacobian, m_tempA1);
        MatrixMul(jacobian, m_tempA1, m_tempA2); 
        MatrixAdd(m_A, m_tempA2, m_A);

        // Compute B matrix
        MatrixScalarMul(eval, jacobian, m_tempB);
        MatrixAdd(m_B, m_tempB, m_B); 
    }
}

// Get A,B for matrix solve
void  GaussNewton::computeABMatrix() {
    const size_t var_size = static_cast<OracleMatrix*>(m_oracle)->getVariableSize();
    // If m_A is a nullptr
    if(nullptr == m_A) {
        m_A = Matrix<Type>::MatrixFactory::CreateMatrixPtr(var_size, var_size);
    }

    // If m_B is a nullptr
    if(nullptr == m_B) {
        m_B = Matrix<Type>::MatrixFactory::CreateMatrixPtr(var_size, 1);
    }

    // Reset m_A and m_B
    ResetZero(m_A);
    ResetZero(m_B);

    for(size_t i{}; i < m_size; ++i) {
        // Set data
        setData(i);
        // Eval and jacobian
        const Matrix<Type>* eval = static_cast<OracleMatrix*>(m_oracle)->evalMat();
        const Matrix<Type>* jacobian = static_cast<OracleMatrix*>(m_oracle)->jacobian();

        // Compute A matrix
        MatrixTranspose(jacobian, m_tempA1);
        MatrixMul(m_tempA1, jacobian, m_tempA2); 
        MatrixAdd(m_A, m_tempA2, m_A);

        // Compute B matrix
        MatrixMul(m_tempA1, eval, m_tempB);
        MatrixAdd(m_B, m_tempB, m_B); 
    }
}

// Set data (X,Y,size)
GaussNewton& GaussNewton::setData(Matrix<Type>* X, Matrix<Type>* Y, const size_t size) {
    m_size = size;
    m_X = X;
    m_Y = Y;
    return *this;
}

// Set data parameters
GaussNewton& GaussNewton::setParameters(Matrix<Parameter>* PX, Matrix<Parameter>* PY) {
    m_PX = PX;
    m_PY = PY;
    return *this;
}

// Set oracle
GaussNewton& GaussNewton::setOracle(Oracle* oracle) {
    m_oracle = oracle;
    m_oracle_type = oracle->getOracleType();
    return *this;
} 

// Get jacobian
Pair<Matrix<Type>*,Matrix<Type>*> GaussNewton::getAB() {
    if("OracleScalar" == m_oracle_type) {
        computeABScalar();
        return {m_A,m_B};
    } else if("OracleMatrix" == m_oracle_type) {
        computeABMatrix();
        return {m_A,m_B};
    } 
    else {
        ASSERT(false, "[OracleMatrix] getA method not implemented yet");
        return {nullptr,nullptr};
    }
}

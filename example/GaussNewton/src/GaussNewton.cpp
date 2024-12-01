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

// Compute jacobians for individual data and store it in m_jt matrix
void GaussNewton::computeJt() {
    const size_t var_size = m_oracle->getVariableSize();
    if(nullptr == m_jt) {
        m_jt = Matrix<Type>::MatrixFactory::CreateMatrixPtr(var_size, m_size);
    }
    for(size_t i{}; i < m_size; ++i) {
        const Pair<size_t,size_t> row_select{0,var_size-1};
        const Pair<size_t,size_t> col_select{i,i};
        setData(i);
        m_jt->setBlockMat(row_select, col_select, &m_oracle->jacobian());  
    }
}

// Compute residual for individual data and store it in m_res matrix
void GaussNewton::computeRes() {
    if(nullptr == m_rd) {
        m_rd = Matrix<Type>::MatrixFactory::CreateMatrixPtr(m_size, 1);
    }
    for(size_t i{}; i < m_size; ++i) {
        setData(i);
        (*m_rd)(i,0) = m_oracle->eval();
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
    return *this;
} 

// Get jacobian
Matrix<Type>* GaussNewton::getJt() {
    computeJt();
    return m_jt;
}

// Get residual
Matrix<Type>* GaussNewton::getRes() {
    computeRes();
    return m_rd;
}

// Get objective
Type GaussNewton::computeObj() {
    // Compute residual and return accumulated value
    computeRes(); 
    return std::reduce(EXECUTION_PAR m_rd->getMatrixPtr(),  m_rd->getMatrixPtr() + m_rd->getNumElem());    
}

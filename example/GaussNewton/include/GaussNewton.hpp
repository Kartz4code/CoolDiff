/**
 * @file example/GaussNewton/include/GaussNewton.hpp
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

#include "OracleMatrix.hpp"
#include "OracleScalar.hpp"

class GaussNewton {
  private:
    // Parameter matrices for input and output data
    Matrix<Parameter>* m_PX{nullptr};
    Matrix<Parameter>* m_PY{nullptr};

    // Data size, X, Y dataset
    size_t m_size{(size_t)(-1)}; 
    Matrix<Type>* m_X{nullptr};
    Matrix<Type>* m_Y{nullptr}; 
    
    // Oracle
    Oracle* m_oracle{nullptr};

    // Jacobian transpose, Residual
    Matrix<Type>* m_jt{nullptr};
    Matrix<Type>* m_rd{nullptr};

    // Set data
    void setData(const size_t);
    // Compute jacobians for individual data and store it in m_jt matrix
    void computeJt();
    // Compute residual for individual data and store it in m_rd matrix
    void computeRes();

  public:
    GaussNewton() = default; 

    // Set data (X,Y,size)
    GaussNewton& setData(Matrix<Type>*, Matrix<Type>*, const size_t);
    // Set data parameters
    GaussNewton& setParameters(Matrix<Parameter>*, Matrix<Parameter>*);
    // Set oracle
    GaussNewton& setOracle(Oracle*);
    // Get jacobian
    Matrix<Type>* getJt();
    // Get residual
    Matrix<Type>* getRes();

    // Get objective
    Type computeObj();

    ~GaussNewton() = default;
};

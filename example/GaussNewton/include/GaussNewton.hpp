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
    std::string_view m_oracle_type{""};

    // A matrix results and temps
    Matrix<Type>* m_A{nullptr};
    Matrix<Type>* m_tempA1{nullptr};
    Matrix<Type>* m_tempA2{nullptr};

    // B matrix results and temps
    Matrix<Type>* m_B{nullptr};
    Matrix<Type>* m_tempB{nullptr};

    // X matrix result
    Matrix<Type>* m_delX{nullptr};

    // Maximum number of iterations
    size_t m_max_iter{20};

    // Set data
    void setData(const size_t);

    // Get A,B for matrix solve
    void computeABScalar(const size_t);
    // Get A,B for matrix solve
    void computeABMatrix(const size_t);
    // Get A,B
    void computeAB(const size_t);


    // Update values for scalar solve
    void updateScalar(const size_t);
    // Update values for matrix solve
    void updateMatrix(const size_t);
    // Update 
    void update(const size_t);


  public:
    GaussNewton() = default; 

    // Set data (X,Y,size)
    GaussNewton& setData(Matrix<Type>*, Matrix<Type>*);
    // Set data parameters
    GaussNewton& setParameters(Matrix<Parameter>*, Matrix<Parameter>*);
    // Set oracle
    GaussNewton& setOracle(Oracle*);
    // Set maximum number of iterations
    GaussNewton& setMaxIterations(const size_t);

    // Solve Gauss Newton problem
    void solve();

    ~GaussNewton() = default;
};

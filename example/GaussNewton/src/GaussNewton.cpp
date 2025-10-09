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

// Set data unit
void GaussNewton::setDataUnit(const size_t i) {
  // If parameter vector or data is empty, don't enter loop
  if ((nullptr != m_DX) && (nullptr != m_X)) {
    for (size_t j{}; j < m_X->getNumColumns(); ++j) {
      (*m_DX)[j] = (*m_X)(i, j);
    }
  }

  // If parameter vector or data is empty, don't enter loop
  if ((nullptr != m_DY) && (nullptr != m_Y)) {
    for (size_t j{}; j < m_Y->getNumColumns(); ++j) {
      (*m_DY)[j] = (*m_Y)(i, j);
    }
  }
}

// Get A,B for matrix solve
void GaussNewton::computeABScalar(const size_t var_size) {
  // Reset m_A and m_B
  ResetZero(m_A);
  ResetZero(m_B);

  for (size_t i{}; i < m_size; ++i) {
    // Set data unit
    setDataUnit(i);

    // Eval and jacobian
    const Type eval = static_cast<OracleScalar*>(m_oracle)->eval();
    const Matrix<Type>* jacobian = static_cast<OracleScalar*>(m_oracle)->jacobian();

    // Compute A matrix
    MatrixTranspose(jacobian, m_tempA1);
    MatrixMul(m_tempA1, jacobian, m_tempA2);
    MatrixAdd(m_A, m_tempA2, m_A);

    // Compute B matrix
    MatrixScalarMul(eval, m_tempA1, m_tempB);
    MatrixAdd(m_B, m_tempB, m_B);
  }
}

// Get A,B for matrix solve
void GaussNewton::computeABMatrix(const size_t var_size) {
  // Reset m_A and m_B
  ResetZero(m_A);
  ResetZero(m_B);

  for (size_t i{}; i < m_size; ++i) {
    // Set data unit
    setDataUnit(i);

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

// Allocate resources
void GaussNewton::allocateMem(const size_t var_size) {
  // If m_A is a nullptr
  if (nullptr == m_A) {
    m_A = Matrix<Type>::MatrixFactory::CreateMatrixPtr(var_size, var_size);
  }
  // If m_B is a nullptr
  if (nullptr == m_B) {
    m_B = Matrix<Type>::MatrixFactory::CreateMatrixPtr(var_size, 1);
  }
}

// Compute AB matrices
void GaussNewton::computeAB(const size_t var_size) {
  if ("OracleScalar" == m_oracle_type) {
    computeABScalar(var_size);
  } else if ("OracleMatrix" == m_oracle_type) {
    computeABMatrix(var_size);
  } else {
    ASSERT(false, "Unknown oracle");
  }
}

// Update values for scalar solve
void GaussNewton::updateScalar(const size_t var_size) {
  // Get variables
  auto& X = static_cast<OracleScalar *>(m_oracle)->getVariables();
  // Update all variable values
  const auto idx = Range<size_t>(0, var_size);
  std::for_each(
      EXECUTION_PAR idx.begin(), idx.end(),
      [this, &X](const size_t i) { X[i] = CoolDiff::Scalar::Eval(X[i]) - (*m_delX)[i]; });
}

// Update values for matrix solve
void GaussNewton::updateMatrix(const size_t var_size) {
  // Get variables
  auto& X = static_cast<OracleMatrix*>(m_oracle)->getVariables();
  // Update all variable values
  const auto idx = Range<size_t>(0, var_size);
  std::for_each(EXECUTION_PAR idx.begin(), idx.end(),
                [this, &X](const size_t i) { X[i] = CoolDiff::Scalar::Eval(X[i]) - (*m_delX)[i]; });
}

// Update
void GaussNewton::update(const size_t var_size) {
  // If oracle type is scalar
  if ("OracleScalar" == m_oracle_type) {
    updateScalar(var_size);
  }
  // If oracle type is matrix
  else if ("OracleMatrix" == m_oracle_type) {
    updateMatrix(var_size);
  } else {
    ASSERT(false, "Unknown oracle");
  }
}

// Set data (X,Y,size)
GaussNewton& GaussNewton::setData(Matrix<Type> *X, Matrix<Type> *Y) {
  ASSERT((X->getNumRows() == Y->getNumRows()), "Number of input/output rows are not equal");
  ASSERT((X->getNumColumns() == Y->getNumColumns()), "Number of input/output rows are not equal");
  m_size = X->getNumRows(); m_X = X; m_Y = Y;
  return *this;
}

// Set data parameters
GaussNewton& GaussNewton::setDataParameters(Matrix<Parameter> *PX, Matrix<Parameter> *PY) {
  m_DX = PX; m_DY = PY;
  return *this;
}

// Set oracle
GaussNewton& GaussNewton::setOracle(Oracle *oracle) {
  m_oracle = oracle;
  m_oracle_type = oracle->getOracleType();
  return *this;
}

// Set maximum number of iterations
GaussNewton& GaussNewton::setMaxIterations(const size_t max_iter) {
  m_max_iter = max_iter;
  return *this;
}

// Solve Gauss Newton problem
void GaussNewton::solve() {
  // Null check of oracle
  NULL_CHECK(m_oracle, "Oracle is a null pointer");
  const size_t var_size = m_oracle->getVariableSize();

  // If m_A is a nullptr
  if (nullptr == m_delX) {
    m_delX = Matrix<Type>::MatrixFactory::CreateMatrixPtr(var_size, 1);
  }

  // Allocate memory
  allocateMem(var_size);

  // Raw pointer for A and B matrices
  Type* A = m_A->getMatrixPtr();
  Type* B = m_B->getMatrixPtr();

  NULL_CHECK(A, "A matrix is a null pointer");
  NULL_CHECK(B, "B matrix is a null pointer");

  /* Linear algebra solve A/B */
  // Convert A and B to Eigen matrix
  const Eigen::Map<EigenMatrix> eigA(A, var_size, var_size);
  const Eigen::Map<EigenMatrix> eigB(B, var_size, 1);

  // Gauss Newton iterations
  for (size_t iter{}; iter < m_max_iter; ++iter) {
    /* Compute A and B matrices */
    computeAB(var_size);
    
    const Eigen::LLT<EigenMatrix> llt(eigA);
    // Solve and store results
    const auto delX = llt.solve(eigB);
    Eigen::Map<EigenMatrix>(m_delX->getMatrixPtr(), delX.rows(), delX.cols()) = delX;

    /* Update variable values */
    update(var_size);
  }
}
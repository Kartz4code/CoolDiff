/**
 * @file src/Matrix/MatrixSplOps/MatrixEyeOps.cpp
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

#include "MatrixEyeOps.hpp"
#include "CommonMatFunctions.hpp"

// Is the matrix square
bool IsSquareMatrix(const Matrix<Type> *m) {
  // Check for square matrix
  return (m->getNumColumns() == m->getNumRows());
}

// Is the matrix identity
bool IsEyeMatrix(const Matrix<Type> *m) {
  // If matrix not square, return false
  if (false == IsSquareMatrix(m)) {
    return false;
  }

  // Eye special matrix check
  if (m->getMatType() == MatrixSpl::EYE) {
    return true;
  }
  // else if m is some special matrix
  else if (m->getMatType() != -1) {
    return false;
  }

  // Rows and columns
  const size_t rows = m->getNumRows();
  const size_t cols = m->getNumColumns();

  // Diagonal elements check (1's)
  const auto diag_idx = Range<size_t>(0, rows);
  if (auto it = std::find_if(
          EXECUTION_PAR diag_idx.begin(), diag_idx.end(),
          [&m](const size_t i) { return ((*m)(i, i) != (Type)(1)); });
      it != diag_idx.end()) {
    return false;
  }

  // Non-diagonal elements check (0's)
  const auto non_diag_idx = Range<size_t>(0, rows * cols);
  if (auto it =
          std::find_if(EXECUTION_PAR non_diag_idx.begin(), non_diag_idx.end(),
                       [&m, rows, cols](const size_t n) {
                         const size_t j = n % cols;
                         const size_t i = (n - j) / cols;
                         return ((i != j) && ((*m)(i, j) != (Type)(0)));
                       });
      it != non_diag_idx.end()) {
    return false;
  }

  // If none of the above conditions are satisfied, return true
  return true;
}

// Eye matrix addition checks
const Matrix<Type> *EyeMatAdd(const Matrix<Type> *lhs,
                              const Matrix<Type> *rhs) {
  // Left matrix rows and column numbers
  const size_t lrows = lhs->getNumRows();
  const size_t rcols = rhs->getNumColumns();

  // If both lhs and rhs matrices are zero matrices
  if (lhs->getMatType() == MatrixSpl::EYE &&
      rhs->getMatType() == MatrixSpl::EYE) {
    return MemoryManager::MatrixSplPool(lrows, rcols, MatrixSpl::EYE);
  } else if (lhs->getMatType() == MatrixSpl::EYE) {
    return rhs;
  } else if (rhs->getMatType() == MatrixSpl::EYE) {
    return lhs;
  } else {
    return nullptr;
  }
}

// Eye matrix scalar addition
const Matrix<Type> *EyeMatScalarAdd(Type lhs, const Matrix<Type> *rhs) {
  if (rhs->getMatType() == MatrixSpl::EYE) {
    return rhs;
  } else {
    return nullptr;
  }
}

// Eye matrix subtraction
const Matrix<Type> *EyeMatSub(const Matrix<Type> *lhs,
                              const Matrix<Type> *rhs) {
  // Left matrix rows and column numbers
  const size_t lrows = lhs->getNumRows();
  const size_t rcols = rhs->getNumColumns();

  // If both lhs and rhs matrices are zero matrices
  if (lhs->getMatType() == MatrixSpl::EYE &&
      rhs->getMatType() == MatrixSpl::EYE) {
    return MemoryManager::MatrixSplPool(lrows, rcols, MatrixSpl::ZEROS);
  } else if (lhs->getMatType() == MatrixSpl::EYE) {
    return rhs;
  } else if (rhs->getMatType() == MatrixSpl::EYE) {
    return lhs;
  } else {
    return nullptr;
  }
}

// Eye matrix multiplication checks
const Matrix<Type> *EyeMatMul(const Matrix<Type> *lhs,
                              const Matrix<Type> *rhs) {
  // Left matrix rows and column numbers
  const size_t lrows = lhs->getNumRows();
  const size_t rcols = rhs->getNumColumns();

  // If both lhs and rhs matrices are zero matrices
  if (lhs->getMatType() == MatrixSpl::EYE &&
      rhs->getMatType() == MatrixSpl::EYE) {
    return MemoryManager::MatrixSplPool(lrows, rcols, MatrixSpl::EYE);
  }
  // If lhs is a zero matrix
  else if (lhs->getMatType() == MatrixSpl::EYE) {
    return rhs;
    // If rhs is a zero matrix
  } else if (rhs->getMatType() == MatrixSpl::EYE) {
    return lhs;
  }
  // If neither, then return nullptr
  else {
    return nullptr;
  }
}

// Eye matrix scalar addition
const Matrix<Type> *EyeMatScalarMul(Type lhs, const Matrix<Type> *rhs) {
  if (rhs->getMatType() == MatrixSpl::EYE) {
    return rhs;
  } else {
    return nullptr;
  }
}

// Eye matrix Kronocker
const Matrix<Type> *EyeMatKron(const Matrix<Type> *lhs,
                               const Matrix<Type> *rhs) {
  // Left matrix rows and right matrix columns
  const size_t lr = lhs->getNumRows();
  const size_t lc = lhs->getNumColumns();
  const size_t rr = rhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // If both lhs and rhs matrices are eye matrices
  if ((lhs->getMatType() == MatrixSpl::EYE) &&
      (rhs->getMatType() == MatrixSpl::EYE)) {
    return MemoryManager::MatrixSplPool(lr * rr, lc * rc, MatrixSpl::EYE);
  } else if (lhs->getMatType() == MatrixSpl::EYE) {
    return rhs;
  } else if (rhs->getMatType() == MatrixSpl::EYE) {
    return lhs;
  }
  // If neither, then return nullptr
  else {
    return nullptr;
  }
}

// Eye matrix Hadamard product
const Matrix<Type> *EyeMatHadamard(const Matrix<Type> *lhs,
                                   const Matrix<Type> *rhs) {
  // Left matrix rows and column numbers
  const size_t lrows = lhs->getNumRows();
  const size_t rcols = rhs->getNumColumns();

  // If both lhs and rhs matrices are zero matrices
  if (lhs->getMatType() == MatrixSpl::EYE &&
      rhs->getMatType() == MatrixSpl::EYE) {
    return MemoryManager::MatrixSplPool(lrows, rcols, MatrixSpl::EYE);
  }
  // If lhs is a zero matrix
  else if (lhs->getMatType() == MatrixSpl::EYE) {
    return rhs;
    // If rhs is a zero matrix
  } else if (rhs->getMatType() == MatrixSpl::EYE) {
    return lhs;
  }
  // If neither, then return nullptr
  else {
    return nullptr;
  }
}

// Eye matrix addition numerical checks
const Matrix<Type> *EyeMatAddNum(const Matrix<Type> *lhs,
                                 const Matrix<Type> *rhs) {
  // Left matrix rows and column numbers
  const size_t lrows = lhs->getNumRows();
  const size_t rcols = rhs->getNumColumns();

  // Boolean check
  const bool lhs_bool = IsEyeMatrix(lhs);
  const bool rhs_bool = IsEyeMatrix(rhs);

  if (true == lhs_bool && true == rhs_bool) {
    return MemoryManager::MatrixSplPool(lrows, rcols, MatrixSpl::EYE);
  } else if (lhs_bool == true) {
    return rhs;
  } else if (rhs_bool == true) {
    return lhs;
  } else {
    return nullptr;
  }
}

// Eye matrix scalar addition
const Matrix<Type> *EyeMatScalarAddNum(Type lhs, const Matrix<Type> *rhs) {
  const bool rhs_bool = IsEyeMatrix(rhs);
  if (rhs_bool == true) {
    return rhs;
  } else {
    return nullptr;
  }
}

// Eye matrix subtraction numerics
const Matrix<Type> *EyeMatSubNum(const Matrix<Type> *lhs,
                                 const Matrix<Type> *rhs) {
  // Left matrix rows and column numbers
  const size_t lrows = lhs->getNumRows();
  const size_t rcols = rhs->getNumColumns();

  // Boolean check
  const bool lhs_bool = IsEyeMatrix(lhs);
  const bool rhs_bool = IsEyeMatrix(rhs);

  if (true == lhs_bool && true == rhs_bool) {
    return MemoryManager::MatrixSplPool(lrows, rcols, MatrixSpl::ZEROS);
  } else if (lhs_bool == true) {
    return rhs;
  } else if (rhs_bool == true) {
    return lhs;
  } else {
    return nullptr;
  }
}

// Eye matrix multiplication numerical check
const Matrix<Type> *EyeMatMulNum(const Matrix<Type> *lhs,
                                 const Matrix<Type> *rhs) {
  // Get rows and columns
  const size_t lrows = lhs->getNumRows();
  const size_t rcols = rhs->getNumColumns();

  // Boolean check
  const bool lhs_bool = IsEyeMatrix(lhs);
  const bool rhs_bool = IsEyeMatrix(rhs);
  if (lhs_bool == true && rhs_bool == true) {
    return MemoryManager::MatrixSplPool(lrows, rcols, MatrixSpl::EYE);
  } else if (lhs_bool == true) {
    return rhs;
  } else if (rhs_bool == true) {
    return lhs;
  } else {
    return nullptr;
  }
}

// Zero matrix scalar multiplication numerics
const Matrix<Type> *EyeMatScalarMulNum(Type type, const Matrix<Type> *rhs) {
  const bool rhs_bool = IsEyeMatrix(rhs);
  if (rhs_bool == true) {
    return rhs;
  } else {
    return nullptr;
  }
}

// Eye matrix Kronocker product numerics
const Matrix<Type> *EyeMatKronNum(const Matrix<Type> *lhs,
                                  const Matrix<Type> *rhs) {
  // Left matrix rows and right matrix columns
  const size_t lr = lhs->getNumRows();
  const size_t lc = lhs->getNumColumns();
  const size_t rr = rhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // Boolean check
  const bool lhs_bool = IsEyeMatrix(lhs);
  const bool rhs_bool = IsEyeMatrix(rhs);

  if (lhs_bool == true && rhs_bool == true) {
    return MemoryManager::MatrixSplPool(lr * rr, lc * rc, MatrixSpl::EYE);
  } else if (lhs_bool == true) {
    return rhs;
  } else if (rhs_bool == true) {
    return lhs;
  }
  // If neither, then return nullptr
  else {
    return nullptr;
  }
}

// Eye matrix Hadamard product numerics
const Matrix<Type> *EyeMatHadamardNum(const Matrix<Type> *lhs,
                                      const Matrix<Type> *rhs) {
  // Get rows and columns
  const size_t lrows = lhs->getNumRows();
  const size_t rcols = rhs->getNumColumns();

  // Boolean check
  const bool lhs_bool = IsEyeMatrix(lhs);
  const bool rhs_bool = IsEyeMatrix(rhs);
  if (lhs_bool == true && rhs_bool == true) {
    return MemoryManager::MatrixSplPool(lrows, rcols, MatrixSpl::EYE);
  } else if (lhs_bool == true) {
    return rhs;
  } else if (rhs_bool == true) {
    return lhs;
  } else {
    return nullptr;
  }
}

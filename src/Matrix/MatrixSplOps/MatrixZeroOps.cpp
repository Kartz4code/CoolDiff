/**
 * @file src/Matrix/MatrixSplOps/MatrixZeroOps.cpp
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

#include "MatrixZeroOps.hpp"
#include "CommonMatFunctions.hpp"

#define ABS(X) (X > 0 ? (X) : (-X))

// Is the matrix zero
bool IsZeroMatrix(const Matrix<Type> *m) {
  // Zero special matrix check
  if (m->getMatType() == MatrixSpl::ZEROS) {
    return true;
  }
  // else if m is some special matrix
  else if (m->getMatType() != -1) {
    return false;
  }

  // Get Matrix pointer and check for null pointer
  auto *it = m->getMatrixPtr();

  // Check all elements for zero
  return std::all_of(EXECUTION_PAR it, it + m->getNumElem(),
                     [](Type i) { return (i == (Type)(0)); });
}

// Zero matrix addition checks
const Matrix<Type> *ZeroMatAdd(const Matrix<Type> *lhs,
                               const Matrix<Type> *rhs) {
  // Rows and columns of result matrix
  const size_t nrows{lhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};

  // If both lhs and rhs matrices are zero matrices
  if (lhs->getMatType() == MatrixSpl::ZEROS &&
      rhs->getMatType() == MatrixSpl::ZEROS) {
    return MemoryManager::MatrixSplPool(nrows, ncols, MatrixSpl::ZEROS);
  }
  // If lhs is a zero matrix
  else if (lhs->getMatType() == MatrixSpl::ZEROS) {
    return rhs;
    // If rhs is a zero matrix
  } else if (rhs->getMatType() == MatrixSpl::ZEROS) {
    return lhs;
  } else {
    return nullptr;
  }
}

// Zero matrix scalar addition
const Matrix<Type> *ZeroMatScalarAdd(Type lhs, const Matrix<Type> *rhs) {
  // Rows and columns of result matrix
  const size_t nrows{rhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};

  // Result matrix
  Matrix<Type> *result{nullptr};

  // If both lhs and rhs matrices are zero matrices
  if ((lhs == (Type)(0)) && (rhs->getMatType() == MatrixSpl::ZEROS)) {
    return MemoryManager::MatrixSplPool(nrows, ncols, MatrixSpl::ZEROS);
  } else if (lhs == (Type)(0)) {
    return rhs;
  } else if (rhs->getMatType() == MatrixSpl::ZEROS) {
    // Pool matrix
    MemoryManager::MatrixPool(nrows, ncols, result, lhs);
    return result;
  } else {
    return nullptr;
  }
}

// Zero matrix subtraction
const Matrix<Type> *ZeroMatSub(const Matrix<Type> *lhs,
                               const Matrix<Type> *rhs) {
  // Rows and columns of result matrix
  const size_t nrows{lhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};

  // If both lhs and rhs matrices are zero matrices
  if (lhs->getMatType() == MatrixSpl::ZEROS &&
      rhs->getMatType() == MatrixSpl::ZEROS) {
    return MemoryManager::MatrixSplPool(nrows, ncols, MatrixSpl::ZEROS);
  }
  // If lhs is a zero matrix
  else if (lhs->getMatType() == MatrixSpl::ZEROS) {
    return rhs;
    // If rhs is a zero matrix
  } else if (rhs->getMatType() == MatrixSpl::ZEROS) {
    return lhs;
  } else {
    return nullptr;
  }
}

// Zero matrix multiplication checks
const Matrix<Type> *ZeroMatMul(const Matrix<Type> *lhs,
                               const Matrix<Type> *rhs) {
  // Left matrix rows and right matrix columns
  const size_t lr = lhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // If both lhs and rhs matrices are zero matrices
  if (lhs->getMatType() == MatrixSpl::ZEROS ||
      rhs->getMatType() == MatrixSpl::ZEROS) {
    return MemoryManager::MatrixSplPool(lr, rc, MatrixSpl::ZEROS);
  }
  // If neither, then return nullptr
  else {
    return nullptr;
  }
}

// Zero matrix scalar multiplication
const Matrix<Type> *ZeroMatScalarMul(Type lhs, const Matrix<Type> *rhs) {
  // Rows and columns of result matrix
  const size_t nrows{rhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};

  // If both lhs and rhs matrices are zero matrices
  if ((lhs == (Type)(0)) || (rhs->getMatType() == MatrixSpl::ZEROS)) {
    return MemoryManager::MatrixSplPool(nrows, ncols, MatrixSpl::ZEROS);
  } else {
    return nullptr;
  }
}

// Zero matrix kronocker product
const Matrix<Type> *ZeroMatKron(const Matrix<Type> *lhs,
                                const Matrix<Type> *rhs) {
  // Left matrix rows and right matrix columns
  const size_t lr = lhs->getNumRows();
  const size_t lc = lhs->getNumColumns();
  const size_t rr = rhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // If both lhs and rhs matrices are zero matrices
  if (lhs->getMatType() == MatrixSpl::ZEROS ||
      rhs->getMatType() == MatrixSpl::ZEROS) {
    return MemoryManager::MatrixSplPool(lr * rr, lc * rc, MatrixSpl::ZEROS);
  }
  // If neither, then return nullptr
  else {
    return nullptr;
  }
}

// Zero matrix Hadamard product
const Matrix<Type> *ZeroMatHadamard(const Matrix<Type> *lhs,
                                    const Matrix<Type> *rhs) {
  // Left matrix rows and right matrix columns
  const size_t lr = lhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // If both lhs and rhs matrices are zero matrices
  if (lhs->getMatType() == MatrixSpl::ZEROS ||
      rhs->getMatType() == MatrixSpl::ZEROS) {
    return MemoryManager::MatrixSplPool(lr, rc, MatrixSpl::ZEROS);
  }
  // If neither, then return nullptr
  else {
    return nullptr;
  }
}

// Zero matrix convolution
const Matrix<Type> *ZeroMatConv(const size_t rows, const size_t cols,
                                const Matrix<Type> *lhs,
                                const Matrix<Type> *rhs) {

  // If both lhs and rhs matrices are zero matrices
  if (lhs->getMatType() == MatrixSpl::ZEROS ||
      rhs->getMatType() == MatrixSpl::ZEROS) {
    return MemoryManager::MatrixSplPool(rows, cols, MatrixSpl::ZEROS);
  }
  // If neither, then return nullptr
  else {
    return nullptr;
  }
}

// Zero matrix derivative convolution
const Matrix<Type> *ZeroMatDervConv(const size_t rows, const size_t cols,
                                    const Matrix<Type> *lhs,
                                    const Matrix<Type> *dlhs,
                                    const Matrix<Type> *rhs,
                                    const Matrix<Type> *drhs) {
  // If both (lhs,drhs) and (dlhs,rhs) matrices are zero matrices
  if ((lhs->getMatType() == MatrixSpl::ZEROS ||
       drhs->getMatType() == MatrixSpl::ZEROS) &&
      (dlhs->getMatType() == MatrixSpl::ZEROS ||
       rhs->getMatType() == MatrixSpl::ZEROS)) {
    return MemoryManager::MatrixSplPool(rows, cols, MatrixSpl::ZEROS);
  }
  // If (lhs,drhs) matrices are zero matrices
  else if ((lhs->getMatType() == MatrixSpl::ZEROS ||
            drhs->getMatType() == MatrixSpl::ZEROS)) {
    return rhs;
  }
  // If (dlhs,rhs) matrices are zero matrices
  else if ((dlhs->getMatType() == MatrixSpl::ZEROS ||
            rhs->getMatType() == MatrixSpl::ZEROS)) {
    return lhs;
  }
  // If neither, then return nullptr
  else {
    return nullptr;
  }
}

// Zero matrix addition numerical check
const Matrix<Type> *ZeroMatAddNum(const Matrix<Type> *lhs,
                                  const Matrix<Type> *rhs) {
  // Boolean check
  const bool lhs_bool = IsZeroMatrix(lhs);
  const bool rhs_bool = IsZeroMatrix(rhs);

  // Rows and columns of result matrix
  const size_t nrows{lhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};

  if (lhs_bool == true && rhs_bool == true) {
    return MemoryManager::MatrixSplPool(nrows, ncols, MatrixSpl::ZEROS);
  } else if (lhs_bool == true) {
    return rhs;
  } else if (rhs_bool == true) {
    return lhs;
  } else {
    return nullptr;
  }
}

// Zero matrix scalar addition numerics
const Matrix<Type> *ZeroMatScalarAddNum(Type lhs, const Matrix<Type> *rhs) {
  // Rows and columns of result matrix
  const size_t nrows{rhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};

  // Boolean check
  const bool rhs_bool = IsZeroMatrix(rhs);

  // Result matrix
  Matrix<Type> *result{nullptr};

  if ((lhs == (Type)(0)) && rhs_bool == true) {
    return MemoryManager::MatrixSplPool(nrows, ncols, MatrixSpl::ZEROS);
  } else if ((lhs == (Type)(0))) {
    return rhs;
  } else if (rhs_bool == true) {
    // Pool matrix
    MemoryManager::MatrixPool(nrows, ncols, result, lhs);
    return result;
  } else {
    return nullptr;
  }
}

// Zero matrix subtraction numerics
const Matrix<Type> *ZeroMatSubNum(const Matrix<Type> *lhs,
                                  const Matrix<Type> *rhs) {
  // Boolean check
  const bool lhs_bool = IsZeroMatrix(lhs);
  const bool rhs_bool = IsZeroMatrix(rhs);

  // Rows and columns of result matrix
  const size_t nrows{lhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};

  if (lhs_bool == true && rhs_bool == true) {
    return MemoryManager::MatrixSplPool(nrows, ncols, MatrixSpl::ZEROS);
  } else if (lhs_bool == true) {
    return rhs;
  } else if (rhs_bool == true) {
    return lhs;
  } else {
    return nullptr;
  }
}

// Zero matrix multiplication numerics
const Matrix<Type> *ZeroMatMulNum(const Matrix<Type> *lhs,
                                  const Matrix<Type> *rhs) {
  // Left matrix rows and column numbers
  const size_t lr = lhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // Boolean check
  const bool lhs_bool = IsZeroMatrix(lhs);
  const bool rhs_bool = IsZeroMatrix(rhs);
  if (lhs_bool == true || rhs_bool == true) {
    return MemoryManager::MatrixSplPool(lr, rc, MatrixSpl::ZEROS);
  } else {
    return nullptr;
  }
}

// Zero matrix scalar multiplication numerics
const Matrix<Type> *ZeroMatScalarMulNum(Type lhs, const Matrix<Type> *rhs) {
  // Rows and columns of result matrix
  const size_t nrows{rhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};

  // Boolean check
  const bool rhs_bool = IsZeroMatrix(rhs);
  if ((lhs == (Type)(0)) || rhs_bool == true) {
    return MemoryManager::MatrixSplPool(nrows, ncols, MatrixSpl::ZEROS);
  } else {
    return nullptr;
  }
}

// Zero matrix kronocker product numerics
const Matrix<Type> *ZeroMatKronNum(const Matrix<Type> *lhs,
                                   const Matrix<Type> *rhs) {
  // Left matrix rows and right matrix columns
  const size_t lr = lhs->getNumRows();
  const size_t lc = lhs->getNumColumns();
  const size_t rr = rhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // Boolean check
  const bool lhs_bool = IsZeroMatrix(lhs);
  const bool rhs_bool = IsZeroMatrix(rhs);
  if (lhs_bool == true || rhs_bool == true) {
    return MemoryManager::MatrixSplPool(lr * rr, lc * rc, MatrixSpl::ZEROS);
  } else {
    return nullptr;
  }
}

// Zero matrix Hadamard product
const Matrix<Type> *ZeroMatHadamardNum(const Matrix<Type> *lhs,
                                       const Matrix<Type> *rhs) {
  // Left matrix rows and column numbers
  const size_t lr = lhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // Boolean check
  const bool lhs_bool = IsZeroMatrix(lhs);
  const bool rhs_bool = IsZeroMatrix(rhs);
  if (lhs_bool == true || rhs_bool == true) {
    return MemoryManager::MatrixSplPool(lr, rc, MatrixSpl::ZEROS);
  } else {
    return nullptr;
  }
}

// Zero matrix convolution numerics
const Matrix<Type> *ZeroMatConvNum(const size_t rows, const size_t cols,
                                   const Matrix<Type> *lhs,
                                   const Matrix<Type> *rhs) {
  // Boolean check
  const bool lhs_bool = IsZeroMatrix(lhs);
  const bool rhs_bool = IsZeroMatrix(rhs);
  if (lhs_bool == true || rhs_bool == true) {
    return MemoryManager::MatrixSplPool(rows, cols, MatrixSpl::ZEROS);
  } else {
    return nullptr;
  }
}

// Zero matrix derivative convolution numerics
const Matrix<Type> *ZeroMatDervConvNum(const size_t rows, const size_t cols,
                                       const Matrix<Type> *lhs,
                                       const Matrix<Type> *dlhs,
                                       const Matrix<Type> *rhs,
                                       const Matrix<Type> *drhs) {
  // Boolean check
  const bool lhs_bool = IsZeroMatrix(lhs);
  const bool dlhs_bool = IsZeroMatrix(dlhs);
  const bool rhs_bool = IsZeroMatrix(rhs);
  const bool drhs_bool = IsZeroMatrix(drhs);

  // If both (lhs,drhs) and (dlhs,rhs) matrices are zero matrices
  if ((lhs_bool == true || drhs_bool == true) &&
      (dlhs_bool == true || rhs_bool == true)) {
    return MemoryManager::MatrixSplPool(rows, cols, MatrixSpl::ZEROS);
  }
  // If (lhs,drhs) matrices are zero matrices
  else if ((lhs_bool == true || drhs_bool == true)) {
    return rhs;
  }
  // If (dlhs,rhs) matrices are zero matrices
  else if ((dlhs_bool == true || rhs_bool == true)) {
    return lhs;
  }
  // If neither, then return nullptr
  else {
    return nullptr;
  }
}

namespace BaselineCPU {
  void SubZero(const Matrix<Type> *it, Matrix<Type> *&result) {
    /*
      Rows and columns of result matrix and if result is nullptr or if dimensions
      mismatch, then create a new matrix resource
    */
    const size_t nrows{it->getNumRows()};
    const size_t ncols{it->getNumColumns()};

    // Pool matrix
    MemoryManager::MatrixPool(nrows, ncols, result);

    std::transform(EXECUTION_PAR it->getMatrixPtr(),
                  it->getMatrixPtr() + it->getNumElem(), result->getMatrixPtr(),
                  [](const auto &i) { return ((Type)(-1) * i); });
  }
}
/**
 * @file include/Matrix/BinaryOps/GenericMatSum.hpp
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

#include "IMatrix.hpp"
#include "Matrix.hpp"

// Left/right side is an expression
template <typename T1, typename T2, typename... Callables>
class GenericMatSum : public IMatrix<GenericMatSum<T1, T2, Callables...>> {
private:
  // Resources
  T1 *mp_left{nullptr};
  T2 *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatSum)
  DISABLE_MOVE(GenericMatSum)

  // Verify dimensions of result matrix for addition operation
  inline constexpr bool verifyDim() const {
    // Left matrix rows
    const int lr = mp_left->getNumRows();
    // Right matrix rows
    const int rr = mp_right->getNumRows();
    // Left matrix columns
    const int lc = mp_left->getNumColumns();
    // Right matrix columns
    const int rc = mp_right->getNumColumns();
    // Condition for Matrix-Matrix addition
    return ((lr == rr) && (lc == rc));
  }

public:

  // Result
  Matrix<Type> *mp_result{nullptr};
  // Derivative result
  Matrix<Type> *mp_dresult{nullptr};

  // Block index
  const size_t m_nidx{};

  // Constructor
  GenericMatSum(T1 *u, T2 *v, Callables &&...call) : mp_left{u}, 
                                                     mp_right{v}, 
                                                     mp_result{nullptr}, 
                                                     mp_dresult{nullptr},
                                                     m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                     m_nidx{this->m_idx_count++} 
  {}

  /*
  * ======================================================================================================
  * ======================================================================================================
  * ======================================================================================================
   _   _ ___________ _____ _   _  ___   _       _____  _   _ ___________ _ _____
  ___ ______  _____ | | | |_   _| ___ \_   _| | | |/ _ \ | |     |  _  || | | |
  ___| ___ \ |   |  _  |/ _ \|  _  \/  ___| | | | | | | | |_/ / | | | | | /
  /_\ \| |     | | | || | | | |__ | |_/ / |   | | | / /_\ \ | | |\ `--.
  | | | | | | |    /  | | | | | |  _  || |     | | | || | | |  __||    /| |   |
  | | |  _  | | | | `--. \ \ \_/ /_| |_| |\ \  | | | |_| | | | || |____ \ \_/
  /\ \_/ / |___| |\ \| |___\ \_/ / | | | |/ / /\__/ /
   \___/ \___/\_| \_| \_/  \___/\_| |_/\_____/  \___/  \___/\____/\_|
  \_\_____/\___/\_| |_/___/  \____/

  *======================================================================================================
  *======================================================================================================
  *======================================================================================================
  */

  // Get number of rows
  V_OVERRIDE( size_t getNumRows() const ) { 
    return mp_left->getNumRows(); 
  }

  // Get number of columns
  V_OVERRIDE( size_t getNumColumns() const ) { 
    return mp_right->getNumColumns(); 
  }

  // Find me
  bool findMe(void *v) const { 
    BINARY_FIND_ME(); 
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type> *eval()) {
      // Check whether dimensions are correct
      ASSERT(verifyDim(), "Matrix-Matrix addition dimensions mismatch");

      // Get raw pointers to result, left and right matrices
      Matrix<Type>* left_mat = mp_left->eval();
      Matrix<Type>* right_mat = mp_right->eval();

      // Matrix-Matrix addition computation (Policy design)
      std::get<OpMat::ADD_MAT>(m_caller)(left_mat, right_mat, mp_result);

      // Return result pointer
      return mp_result;
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(const Variable &x)) {
      // Check whether dimensions are correct
      ASSERT(verifyDim(), "Matrix-Matrix addition dimensions mismatch");

      // Left and right matrices
      Matrix<Type>* dleft_mat = mp_left->devalF(x);
      Matrix<Type>* dright_mat = mp_right->devalF(x);

      // Matrix-Matrix derivative addition computation (Policy design)
      std::get<OpMat::ADD_MAT>(m_caller)(dleft_mat, dright_mat, mp_dresult);

      // Return result pointer
      return mp_dresult;
  }

    // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalMatF(Matrix<Variable> &X)) {
      // Check whether dimensions are correct
      ASSERT(verifyDim(), "Matrix-Matrix addition dimensions mismatch");

      // Left and right matrices
      Matrix<Type>* dleft_mat = mp_left->devalMatF(X);
      Matrix<Type>* dright_mat = mp_right->devalMatF(X);

      // Matrix-Matrix derivative addition computation (Policy design)
      std::get<OpMat::ADD_MAT>(m_caller)(dleft_mat, dright_mat, mp_dresult);

      // Return result pointer
      return mp_dresult;
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) {
    BINARY_MAT_RESET()
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericMatSum"; 
  }

  // Destructor
  V_DTR(~GenericMatSum()) = default;
};

// GenericMatSum with 2 typename callables
template <typename T1, typename T2>
using GenericMatSumT = GenericMatSum<T1, T2, OpMatType>;

// Function for sum computation
template <typename T1, typename T2>
const GenericMatSumT<T1, T2> &operator+(const IMatrix<T1> &u,
                                         const IMatrix<T2> &v) {
  auto tmp = Allocate<GenericMatSumT<T1, T2>>(
      const_cast<T1 *>(static_cast<const T1 *>(&u)),
      const_cast<T2 *>(static_cast<const T2 *>(&v)), OpMatObj);
  return *tmp;
}
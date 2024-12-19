/**
 * @file include/Matrix/UnaryOps/GenericMatSin.hpp
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

#include "Matrix.hpp"

// Left/right side is a Matrix
template <typename T, typename... Callables>
class GenericMatSin : public IMatrix<GenericMatSin<T, Callables...>> {
private:
  // Resources
  T* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatSin)
  DISABLE_MOVE(GenericMatSin)

  // All matrices
  inline static constexpr const size_t m_size{4};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};

  // Constructor
  constexpr GenericMatSin(T* u, Callables&&...call)  : mp_right{u}, 
                                                       m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                       m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  
  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { 
    return mp_right->getNumRows(); 
  }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { 
    return mp_right->getNumColumns(); 
  }  

  // Find me
  bool findMe(void* v) const { 
    BINARY_RIGHT_FIND_ME(); 
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {  
    // Get raw pointers to result and right matrices
    const Matrix<Type>* right_mat = mp_right->eval();
    MATRIX_SIN(right_mat, mp_arr[0]);
    return mp_arr[0];
  }

  
  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Rows and columns of function and variable
    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();

    // Right matrix derivative
    const Matrix<Type>* right_mat = mp_right->eval();
    const Matrix<Type>* dright_mat = mp_right->devalF(X);

    MATRIX_COS(right_mat, mp_arr[1]);
    MATRIX_KRON(Ones(nrows_x, ncols_x), mp_arr[1], mp_arr[2]);
    MATRIX_HADAMARD(mp_arr[2], dright_mat, mp_arr[3]);

     return mp_arr[3];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_MAT_RIGHT_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericMatSin"; 
  }

  // Destructor
  V_DTR(~GenericMatSin()) = default;

};

// GenericMatSin with 1 typename and callables
template <typename T> 
using GenericMatSinT = GenericMatSin<T, OpMatType>;

// Function for sin computation
template <typename T> 
constexpr const auto& sin(const IMatrix<T> &u) {
  auto tmp = Allocate<GenericMatSinT<T>>(const_cast<T*>(static_cast<const T*>(&u)), OpMatObj);
  return *tmp;
}
/**
 * @file include/Matrix/UnaryOps/GenericMatUnary.hpp
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

// Function computation (Found in bottom)
template <typename Func1, typename Func2>
constexpr const auto& MatUnaryFunction(Func1, Func2);

// Left/right side is a Matrix
template <typename Func1, typename Func2, typename... Callables>
class GenericMatUnary : public IMatrix<GenericMatUnary<Func1, Func2, Callables...>> {
private:
  // Resources
  mutable Matrix<Expression>* mp_right{nullptr};

  // Callables
  Func1 m_f1;
  Func2 m_f2;

  // Callables
  Tuples<Callables...> m_caller;

  // All matrices
  inline static constexpr const size_t m_size{4};
  Matrix<Type>* mp_arr[m_size]{};

  // Set operand
  template <typename T, typename = std::enable_if_t<std::is_base_of_v<MetaMatrix, T>>>
  constexpr const auto& setOperand(const T& X) const {
    mp_right = Allocate<Matrix<Expression>>(X).get();
    return *this;
  }  

public:
  // Block index
  const size_t m_nidx{};  

  // Constructor
  constexpr GenericMatUnary(Func1 f1, Func2 f2, Callables&&... call) : m_f1{f1}, m_f2{f2}, 
                                                                       m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                                       m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  
  template <typename T, typename = std::enable_if_t<std::is_base_of_v<MetaMatrix, T>>>
  constexpr const auto& operator()(const T& X) const {
    auto exp = Allocate<Matrix<Expression>>(MatUnaryFunction(m_f1, m_f2).setOperand(X));
    return *exp;
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
    UNARY_OP_MAT(right_mat, m_f1, mp_arr[0]);
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Rows and columns of function and variable
    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();

    // Right matrix derivative
    const Matrix<Type>* dright_mat = mp_right->devalF(X);
    const Matrix<Type>* right_mat = mp_right->eval();

    UNARY_OP_MAT(right_mat, m_f2, mp_arr[1]);
    MATRIX_KRON(mp_arr[1], Ones(nrows_x, ncols_x), mp_arr[2]);
    MATRIX_HADAMARD(mp_arr[2], dright_mat, mp_arr[3]);

     return mp_arr[3];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_MAT_RIGHT_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericMatUnary"; 
  }

  // Destructor
  V_DTR(~GenericMatUnary()) = default;

};

// GenericMatUnary with 2 typenames and callables
template <typename Func1, typename Func2> 
using GenericMatUnaryT = GenericMatUnary<Func1, Func2, OpMatType>;

// Function computation
template <typename Func1, typename Func2>
constexpr const auto& MatUnaryFunction(Func1 f1, Func2 f2) {
  static_assert(true == std::is_invocable_v<Func1, Type>, "Eval function is not invocable");
  static_assert(true == std::is_invocable_v<Func2, Type>, "Deval function is not invocable");
  auto tmp = Allocate<GenericMatUnaryT<Func1, Func2>>(f1, f2, OpMatObj);
  return *tmp;
}

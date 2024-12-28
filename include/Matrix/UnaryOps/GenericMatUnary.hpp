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

#define UNARY_MATRIX_OPERATOR(OPS, FUNC1, FUNC2)                               \
  template <typename T, typename... Callables>                                 \
  class GenericMat##OPS : public IMatrix<GenericMat##OPS<T, Callables...>> {   \
  private:                                                                     \
    T *mp_right{nullptr};                                                      \
    Tuples<Callables...> m_caller;                                             \
    DISABLE_COPY(GenericMat##OPS)                                              \
    DISABLE_MOVE(GenericMat##OPS)                                              \
    inline static constexpr const size_t m_size{4};                            \
    Matrix<Type> *mp_arr[m_size]{};                                            \
                                                                               \
  public:                                                                      \
    const size_t m_nidx{};                                                     \
    constexpr GenericMat##OPS(T *u, Callables &&...call)                       \
        : mp_right{u}, m_caller{std::make_tuple(                               \
                           std::forward<Callables>(call)...)},                 \
          m_nidx{this->m_idx_count++} {                                        \
      std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);                      \
    }                                                                          \
    V_OVERRIDE(size_t getNumRows() const) { return mp_right->getNumRows(); }   \
    \                                                                       
  V_OVERRIDE(size_t getNumColumns() const) {                                   \
      return mp_right->getNumColumns();                                        \
    }                                                                          \
    bool findMe(void *v) const { BINARY_RIGHT_FIND_ME(); }                     \
    \      
  V_OVERRIDE(Matrix<Type> *eval()) {                                           \
      const Matrix<Type> *right_mat = mp_right->eval();                        \
      UNARY_OP_MAT(right_mat, FUNC1, mp_arr[0]);                               \
      return mp_arr[0];                                                        \
    }                                                                          \
    V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {                    \
      const size_t nrows_x = X.getNumRows();                                   \
      const size_t ncols_x = X.getNumColumns();                                \
      const Matrix<Type> *dright_mat = mp_right->devalF(X);                    \
      const Matrix<Type> *right_mat = mp_right->eval();                        \
      UNARY_OP_MAT(right_mat, FUNC2, mp_arr[1]);                               \
      MATRIX_KRON(mp_arr[1], Ones(nrows_x, ncols_x), mp_arr[2]);               \
      MATRIX_HADAMARD(mp_arr[2], dright_mat, mp_arr[3]);                       \
      return mp_arr[3];                                                        \
    }                                                                          \
    V_OVERRIDE(void reset()) { BINARY_MAT_RIGHT_RESET(); }                     \
    V_OVERRIDE(std::string_view getType() const) {                             \
      return TOSTRING(GenericMat##OPS);                                        \
    }                                                                          \
    V_DTR(~GenericMat##OPS()) = default;                                       \
  };                                                                           \
  template <typename T>                                                        \
  using CONCAT3(GenericMat, OPS, T) = GenericMat##OPS<T, OpMatType>;           \
  template <typename T> constexpr const auto &OPS(const IMatrix<T> &u) {       \
    auto tmp = Allocate < CONCAT3(GenericMat, OPS, T) < T >>                   \
               (const_cast<T *>(static_cast<const T *>(&u)), OpMatObj);        \
    return *tmp;                                                               \
  }
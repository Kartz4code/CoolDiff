/**
 * @file include/Matrix/UnaryOps/GenericMatUnary.hpp
 *
 * @copyright 2023-2025 Karthik Murali Madhavan Rathai
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

#define UNARY_MATRIX_OPERATOR(OPS, FUNC1, FUNC2)                                                    \
template <typename T>                                                                               \
class GenericMat##OPS : public IMatrix<GenericMat##OPS<T>> {                                        \
  private:                                                                                          \
    T* mp_right{nullptr};                                                                           \
    inline static constexpr const size_t m_size{9};                                                 \
    Matrix<Type>* mp_arr[m_size]{};                                                                 \
  public:                                                                                           \
    const size_t m_nidx{};                                                                          \
    OMMatPair m_cache;                                                                              \
    constexpr GenericMat##OPS(T* u) : mp_right{u}, m_nidx{this->m_idx_count++} {                    \
        std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);                                         \
    }                                                                                               \
    V_OVERRIDE(size_t getNumRows() const) { return mp_right->getNumRows(); }                        \
    V_OVERRIDE(size_t getNumColumns() const) { return mp_right->getNumColumns(); }                  \
    bool findMe(void* v) const { BINARY_RIGHT_FIND_ME(); }                                          \
    constexpr const auto& cloneExp() const { return OPS(*mp_right); }                               \
    V_OVERRIDE(Matrix<Type>* eval()) {                                                              \
      const Matrix<Type>* right_mat = mp_right->eval();                                             \
      const size_t start = 0; const size_t end = 1;                                                 \
      std::for_each(EXECUTION_PAR mp_arr + start, mp_arr + end, [&](Matrix<Type>*& m) {             \
      if (nullptr != m) { m = ((right_mat == m) ? nullptr : m); }});                                \
      UNARY_OP_MAT(right_mat, FUNC1, mp_arr[0]);                                                    \
      return mp_arr[0];                                                                             \
    }                                                                                               \
    V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {                                         \
      const Matrix<Type>* dright_mat = mp_right->devalF(X);                                         \
      const Matrix<Type>* right_mat = mp_right->eval();                                             \
      const size_t start = 1; const size_t end = 4;                                                 \
      std::for_each(EXECUTION_PAR mp_arr + start, mp_arr + end, [&](Matrix<Type>*& m) {             \
        if (nullptr != m) {                                                                         \
          m = ((dright_mat == m) ? nullptr : m);                                                    \
          m = ((right_mat == m) ? nullptr : m);                                                     \
        }                                                                                           \
      });                                                                                           \
      const size_t nrows_x = X.getNumRows();                                                        \
      const size_t ncols_x = X.getNumColumns();                                                     \
      UNARY_OP_MAT(right_mat, FUNC2, mp_arr[1]);                                                    \
      MATRIX_KRON(mp_arr[1], CoolDiff::TensorR2::MatrixBasics::Ones(nrows_x, ncols_x), mp_arr[2]);  \
      MATRIX_HADAMARD(mp_arr[2], dright_mat, mp_arr[3]);                                            \
      return mp_arr[3];                                                                             \
    }                                                                                               \
    V_OVERRIDE(void traverse(OMMatPair* cache = nullptr)) {                                         \
      if (cache == nullptr) {                                                                       \
          cache = &m_cache;                                                                         \
          cache->reserve(g_map_reserve);                                                            \
          if (false == (*cache).empty()) {                                                          \
            (*cache).clear();                                                                       \
          }                                                                                         \
          if (false == mp_right->m_visited) {                                                       \
            mp_right->traverse(cache);                                                              \
          }                                                                                         \
          const Matrix<Type>* right_mat = mp_right->eval();                                         \
          UNARY_OP_MAT(right_mat, FUNC2, mp_arr[4]);                                                \
          const auto mp_arr4_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[4]);               \
          if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {                       \
            MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[4], (*cache)[mp_right->m_nidx]);          \
          } else {                                                                                  \
            (*cache)[mp_right->m_nidx] = mp_arr[4];                                                 \
          }                                                                                         \
          for(const auto& [k,v] : (*cache)) {                                                       \
            (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);                                \
          }                                                                                         \
          std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(),           \
                        [&](const auto& item) {                                                     \
                          const size_t rows = mp_arr[4]->getNumRows();                              \
                          const size_t cols = mp_arr[4]->getNumColumns();                           \
                          ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative");\
                          const auto idx = item.first; const auto val = item.second;                \
                          MatType*& ptr = this->m_cloned[this->incFunc()];                          \
                          MATRIX_SCALAR_MUL(mp_arr4_val, val, ptr);                                 \
                          if(auto it2 = cache->find(idx); it2 != cache->end()) {                    \
                            MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);                          \
                          } else {                                                                  \
                            (*cache)[idx] = ptr;                                                    \
                        }});                                                                        \
        }                                                                                           \
        else {                                                                                      \
          if(auto it = cache->find(m_nidx); it != cache->end()) {                                   \
            const auto cCache = it->second;                                                         \
            if (false == mp_right->m_visited) {                                                     \
              mp_right->traverse(cache);                                                            \
            }                                                                                       \
            const Matrix<Type>* right_mat = mp_right->eval();                                       \
            UNARY_OP_MAT(right_mat, FUNC2, mp_arr[5]);                                              \
            MATRIX_HADAMARD(mp_arr[5], cCache, mp_arr[6]);                                          \
            const auto mp_arr6_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[6]);             \
            if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {                     \
              MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[6], (*cache)[mp_right->m_nidx]);        \
            } else {                                                                                \
              (*cache)[mp_right->m_nidx] = mp_arr[6];                                               \
            }                                                                                       \
            for(const auto& [k,v] : (*cache)) {                                                     \
              (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);                              \
            }                                                                                       \
            std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(),         \
                          [&](const auto &item) {                                                   \
                            const size_t rows = mp_arr[6]->getNumRows();                            \
                            const size_t cols = mp_arr[6]->getNumColumns();                         \
                            ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative");\
                            const auto idx = item.first; const auto val = item.second;              \
                            MatType*& ptr = this->m_cloned[this->incFunc()];                        \
                            MATRIX_SCALAR_MUL(mp_arr6_val, val, ptr);                               \
                            if(auto it2 = cache->find(idx); it2 != cache->end()) {                  \
                              MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);                        \
                            } else {                                                                \
                              (*cache)[idx] = ptr;                                                  \
                            }                                                                       \
            });                                                                                     \
          }                                                                                         \
        }                                                                                           \
        if (false == mp_right->m_visited) {                                                         \
          mp_right->traverse(cache);                                                                \
        }                                                                                           \
    }                                                                                               \
    V_OVERRIDE(OMMatPair& getCache()) { return m_cache; }                                           \
    V_OVERRIDE(void reset()) { BINARY_MAT_RIGHT_RESET(); }                                          \
    V_OVERRIDE(std::string_view getType() const) { return TOSTRING(GenericMat##OPS); }              \
    V_DTR(~GenericMat##OPS()) = default;                                                            \
  };                                                                                                \
template <typename T>                                                                               \
using CONCAT3(GenericMat, OPS, T) = GenericMat##OPS<T>;                                             \
template <typename T> constexpr const auto& OPS(const IMatrix<T>& u) {                              \
  const auto& _u = u.cloneExp();                                                                    \
  auto tmp = Allocate<CONCAT3(GenericMat, OPS, T)<T>>(const_cast<T*>(static_cast<const T*>(&_u)));  \
  return *tmp;                                                                                      \
}


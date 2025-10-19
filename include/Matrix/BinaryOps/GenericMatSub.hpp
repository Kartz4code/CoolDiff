/**
 * @file include/Matrix/BinaryOps/GenericMatSub.hpp
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
template <typename T1, typename T2, typename... Callables>
class GenericMatSub : public IMatrix<GenericMatSub<T1, T2, Callables...>> {
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericMatSub)
    DISABLE_MOVE(GenericMatSub)
  #endif

  // Verify dimensions of result matrix for subtraction operation
  inline constexpr bool verifyDim() const {
    // Left matrix rows
    const int lr = mp_left->getNumRows();
    // Right matrix rows
    const int rr = mp_right->getNumRows();
    // Left matrix columns
    const int lc = mp_left->getNumColumns();
    // Right matrix columns
    const int rc = mp_right->getNumColumns();
    // Condition for Matrix-Matrix subtraction
    return ((lr == rr) && (lc == rc));
  }

  // All matrices
  inline static constexpr const size_t m_size{6};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache;

  // Constructor
  constexpr GenericMatSub(T1* u, T2* v, Callables&&... call) : mp_left{u}, 
                                                               mp_right{v}, 
                                                               m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                               m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { 
    return mp_left->getNumRows(); 
  }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { 
    return mp_right->getNumColumns(); 
  }

  // Find me
  bool findMe(void* v) const { 
    BINARY_FIND_ME(); 
  }

   // Clone matrix expression
  constexpr const auto& cloneExp() const {
    return (*mp_left) - (*mp_right);
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix subtraction dimensions mismatch");

    // Get raw pointers to result, left and right matrices
    const Matrix<Type>* left_mat = mp_left->eval();
    const Matrix<Type>* right_mat = mp_right->eval();

    const size_t start = 0;
    const size_t end = 1;
    std::for_each(EXECUTION_PAR mp_arr + start, mp_arr + end, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                     
        m = ((left_mat == m) ? nullptr : m);
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    // Matrix-Matrix subtraction computation (Policy design)
    MATRIX_SUB(left_mat, right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix subtraction dimensions mismatch");

    // Left and right matrices
    const Matrix<Type>* dleft_mat = mp_left->devalF(X);
    const Matrix<Type>* dright_mat = mp_right->devalF(X);

    const size_t start = 1; 
    const size_t end = 2;
    std::for_each(EXECUTION_PAR mp_arr + start, mp_arr + end, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                     
        m = ((dleft_mat == m) ? nullptr : m);
        m = ((dright_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    // Matrix-Matrix derivative subtraction computation (Policy design)
    MATRIX_SUB(dleft_mat, dright_mat, mp_arr[1]);

    // Return result pointer
    return mp_arr[1];
  }

  // Traverse
  V_OVERRIDE(void traverse(OMMatPair* cache = nullptr)) {
    // If cache is nullptr, i.e. for the first step
    if (cache == nullptr) {
      // cache is m_cache
      cache = &m_cache;
      cache->reserve(g_map_reserve);
      // Clear cache in the first entry
      if (false == (*cache).empty()) {
        (*cache).clear();
      }

      // Traverse left node
      if (false == mp_left->m_visited) {
        mp_left->traverse(cache);
      }
      // Traverse right node
      if (false == mp_right->m_visited) {
        mp_right->traverse(cache);
      }

      /* IMPORTANT: The derivative is computed here */
      const size_t n = mp_left->getNumRows();
      const auto eye_n = const_cast<MatType*>(CoolDiff::TensorR2::MatrixBasics::Eye(n));

      MATRIX_SCALAR_MUL(1, eye_n, mp_arr[2]); 
      MATRIX_SCALAR_MUL(-1, eye_n, mp_arr[3]); 

      if(auto it2 = cache->find(mp_left->m_nidx); it2 != cache->end()) {
        MATRIX_ADD((*cache)[mp_left->m_nidx], mp_arr[2], (*cache)[mp_left->m_nidx]); 
      } else {
        (*cache)[mp_left->m_nidx] = mp_arr[2];
      }

      if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
        MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[3], (*cache)[mp_right->m_nidx]); 
      } else {
        (*cache)[mp_right->m_nidx] = mp_arr[3];
      }

      // Clone the cache
      for(const auto& [k,v] : (*cache)) {
        (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);
      }

      // Modify cache for left node
      std::for_each(EXECUTION_PAR mp_left->m_cache.begin(), mp_left->m_cache.end(), 
                      [&](const auto& item) {
                      const size_t rows = mp_arr[2]->getNumRows();
                      const size_t cols = mp_arr[2]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                      const auto idx = item.first; const auto val = item.second;
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], val, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = val;
                      }
      });
  
      // Modify cache for right node
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                    [&](const auto& item) {
                      const size_t rows = mp_arr[3]->getNumRows();
                      const size_t cols = mp_arr[3]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(-1, val, ptr);
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = ptr;
                      }
      });
    } else {      
      // Cached value
      if(auto it = cache->find(m_nidx); it != cache->end()) {    
        const auto cCache = it->second;
        
        // Traverse left node
        if (false == mp_left->m_visited) {
          mp_left->traverse(cache);
        }

        // Traverse right node
        if (false == mp_right->m_visited) {
          mp_right->traverse(cache);         
        }
        
        MATRIX_SCALAR_MUL(1, cCache, mp_arr[4]); 
        MATRIX_SCALAR_MUL(-1, cCache, mp_arr[5]);
        
        const auto mp_arr4_val = (*mp_arr[4])(0,0);
        const auto mp_arr5_val = (*mp_arr[5])(0,0);

        if(auto it2 = cache->find(mp_left->m_nidx); it2 != cache->end()) {
          MATRIX_ADD((*cache)[mp_left->m_nidx], mp_arr[4], (*cache)[mp_left->m_nidx]); 
        } else {
          (*cache)[mp_left->m_nidx] = mp_arr[4];
        }

        if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
          MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[5], (*cache)[mp_right->m_nidx]); 
        } else {
          (*cache)[mp_right->m_nidx] = mp_arr[5];
        }

        // Clone the cache
        for(const auto& [k,v] : (*cache)) {
          (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);
        }

        // Modify cache for left node
        std::for_each(EXECUTION_PAR mp_left->m_cache.begin(), mp_left->m_cache.end(), 
                      [&](const auto& item) {
                      const size_t rows = mp_arr[4]->getNumRows();
                      const size_t cols = mp_arr[4]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 
                      
                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(mp_arr4_val, val, ptr);
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = ptr;
                      }
        });

        // Modify cache for right node
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                      [&](const auto &item) {
                      const size_t rows = mp_arr[5]->getNumRows();
                      const size_t cols = mp_arr[5]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(mp_arr5_val, val, ptr);
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = ptr;
                      }
        });
      }
    }
    // Traverse left/right nodes
    if (false == mp_left->m_visited) {
      mp_left->traverse(cache);
    }
    if (false == mp_right->m_visited) {
      mp_right->traverse(cache);
    } 
  }

  // Get cache
  V_OVERRIDE(OMMatPair& getCache()) {
    return m_cache;
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_MAT_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericMatSub"; 
  }

  // Destructor
  V_DTR(~GenericMatSub()) = default;
};

// GenericMatSub with 2 typename callables
template <typename T1, typename T2>
using GenericMatSubT = GenericMatSub<T1, T2, OpMatType>;

// Function for sub computation
template <typename T1, typename T2>
constexpr const auto& operator-(const IMatrix<T1>& u, const IMatrix<T2>& v) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  auto tmp = Allocate<GenericMatSubT<T1, T2>>(const_cast<T1*>(static_cast<const T1*>(&_u)),
                                              const_cast<T2*>(static_cast<const T2*>(&_v)), 
                                              OpMatObj);
  return *tmp;
}

// Matrix sub with Type (LHS)
template <typename T>
constexpr const auto& operator-(const Type& v, const IMatrix<T>& u) {
  const auto& _u = u.cloneExp();
  return (v + ((Type)(-1) * _u));
}

// Matrix sub with Type (RHS)
template <typename T>
constexpr const auto& operator-(const IMatrix<T>& u, const Type& v) {
  const auto& _u = u.cloneExp();
  return (_u + ((Type)(-1) * v));
}

// Matrix sub with scalar (LHS) - SFINAE'd
template <typename T1, typename T2, typename = CoolDiff::TensorR1::Details::IsPureMetaVariableType<T1>>
constexpr const auto& operator-(const T1& v, const IMatrix<T2>& u) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  return (_v + ((Type)(-1) * _u));
}

// Matrix sub with scalar (RHS) - SFINAE'd
template <typename T1, typename T2, typename = CoolDiff::TensorR1::Details::IsPureMetaVariableType<T2>>
constexpr const auto& operator-(const IMatrix<T1>& u, const T2& v) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  return (_u + ((Type)(-1) * _v));
}

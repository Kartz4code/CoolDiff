/**
 * @file include/Matrix/BinaryOps/GenericMatHadamard.hpp
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
#include "MatrixBasics.hpp"

// Left/right side is a Matrix
template <typename T1, typename T2, typename... Callables>
class GenericMatHadamard : public IMatrix<GenericMatHadamard<T1, T2, Callables...>> {
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericMatHadamard)
    DISABLE_MOVE(GenericMatHadamard)
  #endif

  // Verify dimensions of result matrix for Hadamard product operation
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
  inline static constexpr const size_t m_size{16};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache;

  // Constructor
  constexpr GenericMatHadamard(T1* u, T2* v, Callables&&... call) : mp_left{u}, 
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
    return (*mp_left)^(*mp_right);
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix Hadamard product dimensions mismatch");

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

    // Matrix-Matrix Hadamard product evaluation (Policy design)
    MATRIX_HADAMARD(left_mat, right_mat, mp_arr[0]);

    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix Hadamard product dimensions mismatch");

    // Left and right matrices derivatives
    const Matrix<Type>* dleft_mat = mp_left->devalF(X);
    const Matrix<Type>* dright_mat = mp_right->devalF(X);

    // Left and right matrices evaluation
    const Matrix<Type>* left_mat = mp_left->eval();
    const Matrix<Type>* right_mat = mp_right->eval();

    const size_t start = 1;
    const size_t end = 6;
    std::for_each(EXECUTION_PAR mp_arr + start, mp_arr + end, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((dleft_mat == m) ? nullptr : m);
        m = ((dright_mat == m) ? nullptr : m);
        m = ((left_mat == m) ? nullptr : m);
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();

    // L (X) I - Left matrix and identity Kronocker product (Policy design)
    MATRIX_KRON(left_mat, Ones(nrows_x, ncols_x), mp_arr[4]);
    // R (X) I - Right matrix and identity Kronocker product (Policy design)
    MATRIX_KRON(right_mat, Ones(nrows_x, ncols_x), mp_arr[5]);

    // Hadamard product with left and right derivatives (Policy design)
    MATRIX_HADAMARD(mp_arr[4], dright_mat, mp_arr[2]);
    MATRIX_HADAMARD(dleft_mat, mp_arr[5], mp_arr[3]);

    // Addition between left and right derivatives (Policy design)
    MATRIX_ADD(mp_arr[2], mp_arr[3], mp_arr[1]);

    // Return derivative result pointer
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

      // Get raw pointers to result, left and right matrices
      const Matrix<Type>* left_mat = mp_left->eval();
      const Matrix<Type>* right_mat = mp_right->eval();
      
      /* IMPORTANT: The derivative is computed here */
      MATRIX_SCALAR_MUL(1, left_mat, mp_arr[7]); 
      MATRIX_SCALAR_MUL(1, right_mat, mp_arr[6]); 

      const auto mp_arr6_val = (*mp_arr[6])(0,0);
      const auto mp_arr7_val = (*mp_arr[7])(0,0);

      if(auto it2 = cache->find(mp_left->m_nidx); it2 != cache->end()) {
        MATRIX_ADD((*cache)[mp_left->m_nidx], mp_arr[6], (*cache)[mp_left->m_nidx]); 
      } else {
        (*cache)[mp_left->m_nidx] = mp_arr[6];
      }

      if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
        MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[7], (*cache)[mp_right->m_nidx]); 
      } else {
        (*cache)[mp_right->m_nidx] = mp_arr[7];
      }

      // Clone the cache
      for(const auto& [k,v] : (*cache)) {
        (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);
      }

      // Modify cache for left node
      std::for_each(EXECUTION_PAR mp_left->m_cache.begin(), mp_left->m_cache.end(), 
                      [&](const auto& item) {
                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(mp_arr6_val, val, ptr);
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = ptr;
                      }
      });
  
      // Modify cache for right node
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                    [&](const auto& item) {
                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(mp_arr7_val, val, ptr);
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

        // Get raw pointers to result, left and right matrices
        const Matrix<Type>* left_mat = mp_left->eval();
        const Matrix<Type>* right_mat = mp_right->eval();

        /* IMPORTANT: The derivative is computed here */
        MATRIX_SCALAR_MUL(1, left_mat, mp_arr[9]);
        MATRIX_SCALAR_MUL(1, right_mat, mp_arr[8]);  

        MATRIX_HADAMARD(cCache, mp_arr[8], mp_arr[10]);
        MATRIX_HADAMARD(mp_arr[9], cCache, mp_arr[11]);
        
        const auto mp_arr10_val = (*mp_arr[10])(0,0);
        const auto mp_arr11_val = (*mp_arr[11])(0,0);

        if(auto it2 = cache->find(mp_left->m_nidx); it2 != cache->end()) {
          MATRIX_ADD((*cache)[mp_left->m_nidx], mp_arr[10], (*cache)[mp_left->m_nidx]); 
        } else {
          (*cache)[mp_left->m_nidx] = mp_arr[10];
        }

        if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
          MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[11], (*cache)[mp_right->m_nidx]); 
        } else {
          (*cache)[mp_right->m_nidx] = mp_arr[11];
        }

        // Clone the cache
        for(const auto& [k,v] : (*cache)) {
          (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);
        }

        // Modify cache for left node
        std::for_each(EXECUTION_PAR mp_left->m_cache.begin(), mp_left->m_cache.end(), 
                      [&](const auto& item) {
                        const auto idx = item.first; const auto val = item.second;
                        MatType*& ptr = this->m_cloned[this->incFunc()];
                        MATRIX_SCALAR_MUL(mp_arr10_val, val, ptr);
                        if(auto it2 = cache->find(idx); it2 != cache->end()) {
                          MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                        } else {
                          (*cache)[idx] = ptr;
                        }
        });
      
        // Modify cache for right node
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),
                      mp_right->m_cache.end(), [&](const auto &item) {
                        const auto idx = item.first; const auto val = item.second;
                        MatType*& ptr = this->m_cloned[this->incFunc()];
                        MATRIX_SCALAR_MUL(mp_arr11_val, val, ptr);
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
    return "GenericMatHadamard"; 
  }

  // Destructor
  V_DTR(~GenericMatHadamard()) = default;
};

// GenericMatHadamard with 2 typename callables
template <typename T1, typename T2>
using GenericMatHadamardT = GenericMatHadamard<T1, T2, OpMatType>;

// Function for Hadamard product computation
template <typename T1, typename T2>
constexpr const auto& operator^(const IMatrix<T1>& u, const IMatrix<T2>& v) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  auto tmp = Allocate<GenericMatHadamardT<T1, T2>>( const_cast<T1*>(static_cast<const T1*>(&_u)),
                                                    const_cast<T2*>(static_cast<const T2*>(&_v)), 
                                                    OpMatObj  );
  return *tmp;
}
/**
 * @file include/Matrix/UnaryOps/GenericMatTranspose.hpp
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
class GenericMatTranspose : public IMatrix<GenericMatTranspose<T, Callables...>> {
private:
  // Resources
  T* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatTranspose)
  DISABLE_MOVE(GenericMatTranspose)

  // All matrices
  inline static constexpr const size_t m_size{7};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache;

  // Constructor
  constexpr GenericMatTranspose(T* u, Callables&&... call) : mp_right{u}, 
                                                             m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                             m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { 
    return mp_right->getNumColumns(); 
  }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { 
    return mp_right->getNumRows(); 
  }

  // Find me
  bool findMe(void* v) const { 
    BINARY_RIGHT_FIND_ME(); 
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Get raw pointers to result and right matrices
    const Matrix<Type>* right_mat = mp_right->eval();

    const size_t start = 0;
    const size_t end = 1;
    std::for_each(EXECUTION_PAR mp_arr + start, mp_arr + end, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });  

    // Matrix transpose computation (Policy design)
    MATRIX_TRANSPOSE(right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable> &X)) {
    // Right matrix derivative
    const Matrix<Type>* dright_mat = mp_right->devalF(X);

    const size_t start = 1;
    const size_t end = 2;
    std::for_each(EXECUTION_PAR mp_arr + start, mp_arr + end, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((dright_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    // Rows and columns of function and variable
    const size_t nrows_f = mp_right->getNumRows();
    const size_t ncols_f = mp_right->getNumColumns();
    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();

    MATRIX_DERV_TRANSPOSE(nrows_f, ncols_f, nrows_x, ncols_x, dright_mat, mp_arr[1]);

    // Return result pointer
    return mp_arr[1];
  }

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

      // Traverse right node
      if (false == mp_right->m_visited) {
        mp_right->traverse(cache);
      }

      // Get raw pointers to right matrix
      const Matrix<Type>* right_mat = mp_right->eval();
      
      /* IMPORTANT: The derivative is computed here */
      // Matrix transpose
      MATRIX_TRANSPOSE(right_mat, mp_arr[2]);

      const auto mp_arr2_val = (*mp_arr[2])(0,0);

      if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
        MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[2], (*cache)[mp_right->m_nidx]); 
      } else {
        (*cache)[mp_right->m_nidx] = mp_arr[2];
      }

      for(const auto& [k,v] : (*cache)) {                                                       
        (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);                                
      }
      
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(),
              [&](const auto& item) {                                                     
                const auto idx = item.first; const auto val = item.second;                
                MATRIX_SCALAR_MUL(mp_arr2_val, val, mp_arr[3]);                           
                if(auto it2 = cache->find(idx); it2 != cache->end()) {                    
                  MATRIX_ADD((*cache)[idx], mp_arr[3], (*cache)[idx]);                    
                } else {                                                                  
                  (*cache)[idx] = mp_arr[3];                                              
              }});                                                                        
    } else {
      // Cached value
      if(auto it = cache->find(m_nidx); it != cache->end()) {
        // Cache
        const auto cCache = it->second;
        
        // Traverse right node
        if (false == mp_right->m_visited) {
          mp_right->traverse(cache);
        }

        // Get raw pointers to right matrix
        const Matrix<Type>* right_mat = mp_right->eval();

        /* IMPORTANT: The derivative is computed here */
        // Matrix transpose
        MATRIX_TRANSPOSE(right_mat, mp_arr[4]);
        // Cache multiplication
        MATRIX_HADAMARD(cCache, mp_arr[4], mp_arr[5]);

        const auto mp_arr5_val = (*mp_arr[5])(0,0);

        if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
          MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[5], (*cache)[mp_right->m_nidx]); 
        } else {
          (*cache)[mp_right->m_nidx] = mp_arr[5];
        }

        for(const auto& [k,v] : (*cache)) {                                                     
          (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);                              
        }  

        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(),         
                      [&](const auto &item) {                                                   
                        const auto idx = item.first; const auto val = item.second;              
                        MATRIX_SCALAR_MUL(mp_arr5_val, val, mp_arr[6]);                         
                        if(auto it2 = cache->find(idx); it2 != cache->end()) {                  
                          MATRIX_ADD((*cache)[idx], mp_arr[6], (*cache)[idx]);                  
                        } else {                                                                
                          (*cache)[idx] = mp_arr[6];                                            
                        }                                                                       
        });
      }
    }

    // Traverse right node
    if (false == mp_right->m_visited) {
      mp_right->traverse(cache);
    }
  }

  V_OVERRIDE(OMMatPair& getCache()) {
    return m_cache;
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_MAT_RIGHT_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericMatTranspose"; 
  }

  // Destructor
  V_DTR(~GenericMatTranspose()) = default;
};

// GenericMatTranspose with 1 typename and callables
template <typename T>
using GenericMatTransposeT = GenericMatTranspose<T, OpMatType>;

// Function for transpose computation
template <typename T> 
constexpr const auto& transpose(const IMatrix<T>& u) {
  auto tmp = Allocate<GenericMatTransposeT<T>>(const_cast<T*>(static_cast<const T*>(&u)), OpMatObj);
  return *tmp;
}
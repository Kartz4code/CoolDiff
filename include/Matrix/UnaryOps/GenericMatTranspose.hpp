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
  #if 0
    DISABLE_COPY(GenericMatTranspose)
    DISABLE_MOVE(GenericMatTranspose)
  #endif

  // All matrices
  inline static constexpr const size_t m_size{6};
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

   // Clone matrix expression
  constexpr const auto& cloneExp() const {
    return transpose(*mp_right);
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

      // Traverse right node
      if (false == mp_right->m_visited) {
        mp_right->traverse(cache);
      }

      // Get raw pointers to right matrix
      const size_t n = mp_right->getNumRows();
      const auto eye_n = const_cast<MatType*>(CoolDiff::TensorR2::MatrixBasics::Eye(n));

      /* IMPORTANT: The derivative is computed here */
      // Matrix transpose
      MATRIX_TRANSPOSE(eye_n, mp_arr[2]);

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
                const size_t rows = mp_arr[2]->getNumRows();
                const size_t cols = mp_arr[2]->getNumColumns(); 
                ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative");   
                
                const auto idx = item.first; const auto val = item.second;   
                MatType*& ptr = this->m_cloned[this->incFunc()];             
                MATRIX_SCALAR_MUL(1, val, ptr);                           
                if(auto it2 = cache->find(idx); it2 != cache->end()) {                    
                  MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);                    
                } else {                                                                  
                  (*cache)[idx] = ptr;                                              
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

        /* IMPORTANT: The derivative is computed here */
        // Matrix transpose
        MATRIX_TRANSPOSE(cCache, mp_arr[4]);

        const auto mp_arr4_val = (*mp_arr[4])(0,0);

        if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
          MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[4], (*cache)[mp_right->m_nidx]); 
        } else {
          (*cache)[mp_right->m_nidx] = mp_arr[4];
        }

        for(const auto& [k,v] : (*cache)) {                                                     
          (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);                              
        }  

        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(),         
                      [&](const auto &item) {      
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
      }
    }

    // Traverse right node
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
  const auto& _u = u.cloneExp();
  auto tmp = Allocate<GenericMatTransposeT<T>>(const_cast<T*>(static_cast<const T*>(&_u)), OpMatObj);
  return *tmp;
}
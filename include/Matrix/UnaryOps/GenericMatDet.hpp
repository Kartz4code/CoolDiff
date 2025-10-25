/**
 * @file include/Matrix/UnaryOps/GenericMatDet.hpp
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
class GenericMatDet : public IMatrix<GenericMatDet<T, Callables...>> {
private:
  // Resources
  T* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericMatDet)
    DISABLE_MOVE(GenericMatDet)
  #endif

  // Verify dimensions of result matrix for trace operation
  inline constexpr bool verifyDim() const {
    // Right matrix rows
    const int rr = mp_right->getNumRows();
    // Right matrix columns
    const int rc = mp_right->getNumColumns();
    // Condition for square matrix for trace operation
    return (rr == rc);
  }

  // All matrices
  inline static constexpr const size_t m_size{22};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache;

  // Constructor
  constexpr GenericMatDet(T* u, Callables&&... call) :  mp_right{u}, 
                                                        m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                        m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { 
    return 1; 
  }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { 
    return 1; 
  }

  // Find me
  bool findMe(void* v) const { 
    BINARY_RIGHT_FIND_ME(); 
  }

   // Clone matrix expression
  constexpr const auto& cloneExp() const {
    return det(*mp_right);
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix is not a square matrix to compute determinant");

    // Get raw pointers to result and right matrices
    const Matrix<Type>* right_mat = mp_right->eval();

    // Reset function
    const size_t start = 0;
    const size_t end = 1;
    std::for_each(EXECUTION_PAR mp_arr + start, mp_arr + end, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                     
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    // Matrix determinant
    MATRIX_DET(right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix is not a square matrix to compute determinant");

    // Right matrix derivative
    const Matrix<Type>* dright_mat = mp_right->devalF(X);
    const Matrix<Type>* right_mat = mp_right->eval();

    // Reset function
    const size_t start = 1;
    const size_t end = 11;
    std::for_each(EXECUTION_PAR mp_arr + start, mp_arr + end, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((dright_mat == m) ? nullptr : m);
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    const size_t n_size = mp_right->getNumRows(); 
    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();

    // Matrix inverse
    MATRIX_INVERSE(right_mat, mp_arr[1]); 
    // Matrix transpose
    MATRIX_TRANSPOSE(mp_arr[1], mp_arr[2]);
    // Matrix determinant
    MATRIX_DET(right_mat, mp_arr[9]);
    // Scalar multiplication
    MATRIX_SCALAR_MUL((*mp_arr[9])(0,0), mp_arr[2], mp_arr[10]);
   
    // L (X) I - Left matrix and identity Kronocker product (Policy design)
    MATRIX_KRON(mp_arr[10], CoolDiff::TensorR2::MatrixBasics::Ones(nrows_x, ncols_x), mp_arr[3]);
    // Hadamard product with left and right derivatives (Policy design)
    MATRIX_HADAMARD(mp_arr[3], dright_mat, mp_arr[4]);
    
    MATRIX_KRON(CoolDiff::TensorR2::MatrixBasics::Ones(1, n_size), 
                CoolDiff::TensorR2::MatrixBasics::Eye(nrows_x), 
                mp_arr[5]);
    
    MATRIX_KRON(CoolDiff::TensorR2::MatrixBasics::Ones(n_size, 1), 
                CoolDiff::TensorR2::MatrixBasics::Eye(ncols_x), 
                mp_arr[6]);
                
    MATRIX_MUL(mp_arr[5], mp_arr[4], mp_arr[7]);
    MATRIX_MUL(mp_arr[7], mp_arr[6], mp_arr[8]);

    // Return result pointer
    return mp_arr[8];
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
      const Matrix<Type>* right_mat = mp_right->eval();
      
      /* IMPORTANT: The derivative is computed here */
      // Matrix inverse
      MATRIX_INVERSE(right_mat, mp_arr[11]); 
      // Matrix transpose
      MATRIX_TRANSPOSE(mp_arr[11], mp_arr[12]);
      // Matrix determinant
      MATRIX_DET(right_mat, mp_arr[13]);
      
      const auto mp_arr13_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[13]); 
      
      // Scalar multiplication
      MATRIX_SCALAR_MUL(mp_arr13_val, mp_arr[12], mp_arr[14]);
      const auto mp_arr14_val = (*mp_arr[14])(0,0);

      if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
        MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[14], (*cache)[mp_right->m_nidx]); 
      } else {
        (*cache)[mp_right->m_nidx] = mp_arr[14];
      }

      for(const auto& [k,v] : (*cache)) {                                                       
        (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);                                
      }
      
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(),
              [&](const auto& item) {  
                const size_t rows = mp_arr[14]->getNumRows();
                const size_t cols = mp_arr[14]->getNumColumns(); 
                ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 
                
                const auto idx = item.first; const auto val = item.second;    
                MatType*& ptr = this->m_cloned[this->incFunc()];            
                MATRIX_SCALAR_MUL(mp_arr14_val, val, ptr);                           
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
        const auto cCache_val = CoolDiff::TensorR2::Details::ScalarSpl(cCache);

        // Traverse right node
        if (false == mp_right->m_visited) {
          mp_right->traverse(cache);
        }

        // Get raw pointers to right matrix
        const Matrix<Type>* right_mat = mp_right->eval();

        /* IMPORTANT: The derivative is computed here */
        // Matrix inverse
        MATRIX_INVERSE(right_mat, mp_arr[15]); 
        // Matrix transpose
        MATRIX_TRANSPOSE(mp_arr[15], mp_arr[16]);
        // Matrix determinant
        MATRIX_DET(right_mat, mp_arr[17]);
        const auto mp_arr17_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[17]); 

        // Scalar multiplication
        MATRIX_SCALAR_MUL(mp_arr17_val, mp_arr[16], mp_arr[18]);
        
        // Cache multiplication
        MATRIX_SCALAR_MUL(cCache_val, mp_arr[18], mp_arr[19]);
        const auto mp_arr19_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[19]);

        if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
          MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[19], (*cache)[mp_right->m_nidx]); 
        } else {
          (*cache)[mp_right->m_nidx] = mp_arr[19];
        }

        for(const auto& [k,v] : (*cache)) {                                                     
          (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);                              
        }  

        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(),         
                      [&](const auto &item) {       
                        const size_t rows = mp_arr[19]->getNumRows();
                        const size_t cols = mp_arr[19]->getNumColumns(); 
                        ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 
                        
                        const auto idx = item.first; const auto val = item.second;    
                        MatType*& ptr = this->m_cloned[this->incFunc()];          
                        MATRIX_SCALAR_MUL(mp_arr19_val, val, ptr);                         
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
    return "GenericMatDet"; 
  }

  // Destructor
  V_DTR(~GenericMatDet()) = default;
};

// GenericMatDet with 1 typename and callables
template <typename T> 
using GenericMatDetT = GenericMatDet<T, OpMatType>;

// Function for determinant computation
template <typename T> 
constexpr const auto& det(const IMatrix<T>& u) {
  const auto& _u = u.cloneExp();
  auto tmp = Allocate<GenericMatDetT<T>>(const_cast<T*>(static_cast<const T*>(&_u)), OpMatObj);
  return *tmp;
}
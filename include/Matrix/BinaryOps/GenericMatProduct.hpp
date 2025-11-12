/**
 * @file include/Matrix/BinaryOps/GenericMatProduct.hpp
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
#include "MatrixBasics.hpp"

// Left/right side is a Matrix
template <typename T1, typename T2>
class GenericMatProduct : public IMatrix<GenericMatProduct<T1, T2>> {
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericMatProduct)
    DISABLE_MOVE(GenericMatProduct)
  #endif

  // Verify dimensions of result matrix for multiplication operation
  inline constexpr bool verifyDim() const {
    // Right matrix rows
    const int rr = mp_right->getNumRows();
    // Left matrix columns
    const int lc = mp_left->getNumColumns();
    // Condition for Matrix-Matrix multiplication
    return ((lc == rr));
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
  constexpr GenericMatProduct(T1* u, T2* v) : mp_left{u}, mp_right{v}, m_nidx{this->m_idx_count++} {
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
    return ((*mp_left)*(*mp_right));
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix multiplication dimensions mismatch");

    // Get raw pointers to result, left and right matrices
    const Matrix<Type>* left_mat = mp_left->eval();
    const Matrix<Type>* right_mat = mp_right->eval();

    // Reset function
    const size_t start = 0;
    const size_t end = 1;
    std::for_each(EXECUTION_PAR mp_arr + start, mp_arr + end, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                     
        m = ((left_mat == m) ? nullptr : m);
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    // Matrix multiplication evaluation (Policy design)
    MATRIX_MUL(left_mat, right_mat, mp_arr[0]);

    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix multiplication dimensions mismatch");

    // Left and right matrices derivatives
    const Matrix<Type>* dleft_mat = mp_left->devalF(X);
    const Matrix<Type>* dright_mat = mp_right->devalF(X);

    // Left and right matrices evaluation
    const Matrix<Type>* left_mat = mp_left->eval();
    const Matrix<Type>* right_mat = mp_right->eval();
    
    // Reset function
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
    MATRIX_KRON(left_mat, CoolDiff::TensorR2::MatrixBasics::Eye(nrows_x), mp_arr[4]);
    // R (X) I - Right matrix and identity Kronocke product (Policy design)
    MATRIX_KRON(right_mat, CoolDiff::TensorR2::MatrixBasics::Eye(ncols_x), mp_arr[5]);

    // Product with left and right derivatives (Policy design)
    MATRIX_MUL(mp_arr[4], dright_mat, mp_arr[2]);
    MATRIX_MUL(dleft_mat, mp_arr[5], mp_arr[3]);

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
      MATRIX_TRANSPOSE(left_mat, mp_arr[7]); 
      MATRIX_TRANSPOSE(right_mat, mp_arr[6]); 

      const auto mp_arr6_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[6]);
      const auto mp_arr7_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[7]);

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
                      const size_t rows = mp_arr[6]->getNumRows();
                      const size_t cols = mp_arr[6]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

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
                      const size_t rows = mp_arr[7]->getNumRows();
                      const size_t cols = mp_arr[7]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

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
        MATRIX_TRANSPOSE(left_mat, mp_arr[9]);
        MATRIX_TRANSPOSE(right_mat, mp_arr[8]);  

        MATRIX_MUL(cCache, mp_arr[8], mp_arr[10]);
        MATRIX_MUL(mp_arr[9], cCache, mp_arr[11]);

        const auto mp_arr10_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[10]);
        const auto mp_arr11_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[11]);

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
                        const size_t rows = mp_arr[10]->getNumRows();
                        const size_t cols = mp_arr[10]->getNumColumns(); 
                        ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

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
                      const size_t rows = mp_arr[11]->getNumRows();
                      const size_t cols = mp_arr[11]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

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
    return "GenericMatProduct"; 
  }

  // Destructor
  V_DTR(~GenericMatProduct()) = default;
};

// Left is Type and right is a matrix
template <typename T>
class GenericMatScalarProduct : public IMatrix<GenericMatScalarProduct<T>> {
private:
  // Resources
  Type m_left{};
  T* mp_right{nullptr};

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericMatScalarProduct)
    DISABLE_MOVE(GenericMatScalarProduct)
  #endif

  // All matrices
  inline static constexpr const size_t m_size{5};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache;

  // Constructor
  constexpr GenericMatScalarProduct(Type u, T* v) : m_left{u}, mp_right{v}, m_nidx{this->m_idx_count++} {
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

   // Clone matrix expression
  constexpr const auto& cloneExp() const {
    return m_left*(*mp_right);
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Get raw pointers to result and right matrices
    const Matrix<Type>* right_mat = mp_right->eval();

    std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    // Matrix-Scalar multiplication computation (Policy design)
    MATRIX_SCALAR_MUL(m_left, right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Right matrix derivative
    const Matrix<Type>* dright_mat = mp_right->devalF(X);

    std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((dright_mat == m) ? nullptr : m);
      }                                                                          
    });

    // Matrix-Scalar multiplication computation (Policy design)
    MATRIX_SCALAR_MUL(m_left, dright_mat, mp_arr[1]);

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
      
      /* IMPORTANT: The derivative is computed here */
      const size_t n = mp_right->getNumRows();
      const auto eye_n = const_cast<MatType*>(CoolDiff::TensorR2::MatrixBasics::Eye(n));

      MATRIX_SCALAR_MUL(m_left, eye_n, mp_arr[2]);

      if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
        MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[2], (*cache)[mp_right->m_nidx]); 
      } else {
        (*cache)[mp_right->m_nidx] = mp_arr[2];
      }

      // Clone the cache
      for(const auto& [k,v] : (*cache)) {
        (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);
      }

      // Modify cache for right node
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                    [&](const auto& item) {
                      const size_t rows = mp_arr[2]->getNumRows();
                      const size_t cols = mp_arr[2]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(m_left, val, ptr);
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
        
        // Traverse right node
        if (false == mp_right->m_visited) {
          mp_right->traverse(cache);         
        }
        
        MATRIX_SCALAR_MUL(m_left, cCache, mp_arr[3]); 
        const auto mp_arr3_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[3]);

        if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
          MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[3], (*cache)[mp_right->m_nidx]); 
        } else {
          (*cache)[mp_right->m_nidx] = mp_arr[3];
        }

        // Clone the cache
        for(const auto& [k,v] : (*cache)) {
          (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);
        }

        // Modify cache for right node
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                      [&](const auto &item) {
                      const size_t rows = mp_arr[3]->getNumRows();
                      const size_t cols = mp_arr[3]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(mp_arr3_val, val, ptr);
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = ptr;
                      }
        });
      }
    }
    // Traverse right nodes
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
    return "GenericMatScalarProduct";
  }

  // Destructor
  V_DTR(~GenericMatScalarProduct()) = default;
};

// Left is Expression/Variable/Parameter and right is a matrix
template <typename T1, typename T2>
class GenericMatScalarProductExp : public IMatrix<GenericMatScalarProductExp<T1, T2>> { 
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericMatScalarProductExp)
    DISABLE_MOVE(GenericMatScalarProductExp)
  #endif

  // All matrices
  inline static constexpr const size_t m_size{13};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache;
  OMPair m_cache_v; 

  // Constructor
  constexpr GenericMatScalarProductExp(T1* u, T2* v) : mp_left{u}, mp_right{v}, m_nidx{this->m_idx_count++} {
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

   // Clone matrix expression
  constexpr const auto& cloneExp() const {
    return (*mp_left)*(*mp_right);
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Get raw pointers to result and right matrices
    const Matrix<Type>* right_mat = mp_right->eval();

    std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    const Type val = CoolDiff::TensorR1::Eval((*mp_left));

    // Matrix-Scalar addition computation (Policy design)
    MATRIX_SCALAR_MUL(val, right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Right matrix derivative and evaluation
    const Matrix<Type>* dright_mat = mp_right->devalF(X);
    const Matrix<Type>* right_mat = mp_right->eval();

    std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((dright_mat == m) ? nullptr : m);
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    // Left derivative and evaluation
    CoolDiff::TensorR2::Details::DevalR((*mp_left), X, mp_arr[2]);

    // Evaluate left expression
    const Type val = CoolDiff::TensorR1::Eval((*mp_left));

    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();
    const size_t nrows_f = getNumRows();
    const size_t ncols_f = getNumColumns();


    MATRIX_KRON(CoolDiff::TensorR2::MatrixBasics::Ones(nrows_f, ncols_f), mp_arr[2], mp_arr[5]);
    MATRIX_KRON(right_mat, CoolDiff::TensorR2::MatrixBasics::Ones(nrows_x, ncols_x), mp_arr[6]);

    // Product with left and right derivatives (Policy design)
    MATRIX_SCALAR_MUL(val, dright_mat, mp_arr[4]);
    MATRIX_HADAMARD(mp_arr[5], mp_arr[6], mp_arr[3]);

    // Addition between left and right derivatives (Policy design)
    MATRIX_ADD(mp_arr[4], mp_arr[3], mp_arr[1]);

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
      
      const size_t n = mp_right->getNumRows();
      const auto eye_n = const_cast<MatType*>(CoolDiff::TensorR2::MatrixBasics::Eye(n));
      const Type left = CoolDiff::TensorR1::Eval((*mp_left));
      const Matrix<Type>* right_mat = mp_right->eval();

      /* IMPORTANT: The derivative is computed here */
      MATRIX_TRACE(right_mat, mp_arr[7]);
      MATRIX_SCALAR_MUL(left, eye_n, mp_arr[8]);
      
      const auto mp_arr7_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[7]);
      const auto mp_arr8_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[8]);
      
      if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
        MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[8], (*cache)[mp_right->m_nidx]); 
      } else {
        (*cache)[mp_right->m_nidx] = mp_arr[8];
      }

      // Clone the cache
      for(const auto& [k,v] : (*cache)) {
        (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);
      }

      // Modify cache for left node
      std::for_each(EXECUTION_PAR m_cache_v.begin(), m_cache_v.end(), 
                      [&](const auto& item) {
                      const size_t rows = mp_arr[7]->getNumRows();
                      const size_t cols = mp_arr[7]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(mp_arr7_val*val, CoolDiff::TensorR2::MatrixBasics::Eye(1), ptr);
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = ptr;
                      }
      });
  
      // Modify cache for right node
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                    [&](const auto& item) {
                      const size_t rows = mp_arr[8]->getNumRows();
                      const size_t cols = mp_arr[8]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(mp_arr8_val, val, ptr);
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

        // Traverse right node
        if (false == mp_right->m_visited) {
          mp_right->traverse(cache);
        }

        const Type left = CoolDiff::TensorR1::Eval((*mp_left));
        const Matrix<Type>* right_mat = mp_right->eval();

        /* IMPORTANT: The derivative is computed here */
        MATRIX_TRANSPOSE(cCache, mp_arr[9]);
        MATRIX_MUL(mp_arr[9], right_mat, mp_arr[10]);
        MATRIX_TRACE(mp_arr[10], mp_arr[11]);

        MATRIX_SCALAR_MUL(left, cCache, mp_arr[12]);

        const auto mp_arr11_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[11]);
        const auto mp_arr12_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[12]);

        if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
          MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[12], (*cache)[mp_right->m_nidx]); 
        } else {
          (*cache)[mp_right->m_nidx] = mp_arr[12];
        }

        // Clone the cache
        for(const auto& [k,v] : (*cache)) {
          (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);
        }

        // Modify cache for left node
        std::for_each(EXECUTION_PAR m_cache_v.begin(), m_cache_v.end(), 
                      [&](const auto& item) {
                        const size_t rows = mp_arr[11]->getNumRows();
                        const size_t cols = mp_arr[11]->getNumColumns(); 
                        ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                        const auto idx = item.first; const auto val = item.second;
                        MatType*& ptr = this->m_cloned[this->incFunc()];
                        MATRIX_SCALAR_MUL(mp_arr11_val*val, CoolDiff::TensorR2::MatrixBasics::Eye(1), ptr);
                        if(auto it2 = cache->find(idx); it2 != cache->end()) {
                          MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                        } else {
                          (*cache)[idx] = ptr;
                        }
        });
      
        // Modify cache for right node
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),
                      mp_right->m_cache.end(), [&](const auto &item) {
                      const size_t rows = mp_arr[12]->getNumRows();
                      const size_t cols = mp_arr[12]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(mp_arr12_val, val, ptr);
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = ptr;
                      }
        });
      }
    }

    // If scalar reverse derivative cache is empty, run it once
    if(true == m_cache_v.empty()) {
      Expression exp{*mp_left};
      CoolDiff::TensorR1::PreComp(exp);
      m_cache_v = exp.m_cache;
    }

    // Traverse right node
    if (false == mp_right->m_visited) {
      mp_right->traverse(cache);
    }
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) {
    m_cache_v.clear(); 
    BINARY_MAT_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) {
    return "GenericMatScalarProductExp";
  }

  // Destructor
  V_DTR(~GenericMatScalarProductExp()) = default;
};

// GenericMatProduct with 2 typename callables
template <typename T1, typename T2>
using GenericMatProductT = GenericMatProduct<T1, T2>;

// GenericMatScalarProduct with 1 typename and callables
template <typename T>
using GenericMatScalarProductT = GenericMatScalarProduct<T>;

// GenericMatScalarProductExp with 2 typename and callables
template <typename T1, typename T2>
using GenericMatScalarProductExpT = GenericMatScalarProductExp<T1, T2>;

// Function for product computation
template <typename T1, typename T2>
constexpr const auto& operator*(const IMatrix<T1>& u, const IMatrix<T2>& v) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  auto tmp = Allocate<GenericMatProductT<T1, T2>>(const_cast<T1*>(static_cast<const T1*>(&_u)),
                                                  const_cast<T2*>(static_cast<const T2*>(&_v)));
  return *tmp;
}

// Function for product computation
template <typename T>
constexpr const auto& operator*(Type u, const IMatrix<T>& v) {
  const auto& _v = v.cloneExp();
  auto tmp = Allocate<GenericMatScalarProductT<T>>(u, const_cast<T *>(static_cast<const T*>(&_v)));
  return *tmp;
}

template <typename T>
constexpr const auto& operator*(const IMatrix<T>& v, Type u) {
  const auto& _v = v.cloneExp();
  return (u * _v);
}

template <typename T>
constexpr const auto& operator/(const IMatrix<T>& v, Type u) {
  const auto& _v = v.cloneExp();
  return (((Type)(1)/u) * _v);
}

template <typename T1, typename T2>
constexpr const auto& operator/(const IMatrix<T1>& u, const IMatrix<T2>& v) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  return (_u*inv(_v));
}

// Matrix multiplication with scalar (LHS) - SFINAE'd
template <typename T1, typename T2, typename = CoolDiff::TensorR1::Details::IsPureMetaVariableType<T1>>
constexpr const auto& operator*(const T1& v, const IMatrix<T2>& u) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();

  auto tmp = Allocate<GenericMatScalarProductExpT<T1, T2>>( const_cast<T1*>(static_cast<const T1*>(&_v)),
                                                            const_cast<T2*>(static_cast<const T2*>(&_u)) );
  return *tmp;
}

template <typename T1, typename T2, typename = CoolDiff::TensorR1::Details::IsPureMetaVariableType<T1>>
constexpr const auto& operator/(const IMatrix<T2>& u, const T1& v) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  return (_u * (1/_v));
}

// Matrix sum with scalar (RHS) - SFINAE'd
template <typename T1, typename T2, typename = CoolDiff::TensorR1::Details::IsPureMetaVariableType<T2>>
constexpr const auto& operator*(const IMatrix<T1>& u, const T2& v) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  return (_v * _u);
}
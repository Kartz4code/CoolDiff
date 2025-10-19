/**
 * @file include/Matrix/BinaryOps/GenericMatSum.hpp
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

// Left/right side is a Matrix expression
template <typename T1, typename T2, typename... Callables>
class GenericMatSum : public IMatrix<GenericMatSum<T1, T2, Callables...>> {
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericMatSum)
    DISABLE_MOVE(GenericMatSum)
  #endif

  // Verify dimensions of result matrix for addition operation
  inline constexpr bool verifyDim() const {
    // Left matrix rows and columns
    const size_t lr = mp_left->getNumRows();
    const size_t lc = mp_left->getNumColumns();

    // Right matrix rows and columns
    const size_t rr = mp_right->getNumRows();
    const size_t rc = mp_right->getNumColumns();

    // Condition for Matrix-Matrix addition
    return ((lr == rr) && (lc == rc));
  }

  // All internal matrices
  inline static constexpr const size_t m_size{6};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache{};

  // Constructor
  constexpr GenericMatSum(T1* u, T2* v, Callables&&... call) :  mp_left{u}, 
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
    return (*mp_left) + (*mp_right);
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix addition dimensions mismatch");

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

    // Matrix-Matrix addition computation (Policy design)
    MATRIX_ADD(left_mat, right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix addition dimensions mismatch");

    // Left and right matrices
    const Matrix<Type>* dleft_mat = mp_left->devalF(X);
    const Matrix<Type>* dright_mat = mp_right->devalF(X);

    // Reset function
    const size_t start = 1;
    const size_t end = 2;
    std::for_each(EXECUTION_PAR mp_arr + start, mp_arr + end, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((dleft_mat == m) ? nullptr : m);
        m = ((dright_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    // Matrix-Matrix derivative addition computation (Policy design)
    MATRIX_ADD(dleft_mat, dright_mat, mp_arr[1]);

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
      MATRIX_SCALAR_MUL(1, eye_n, mp_arr[3]); 

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
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], val, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = val;
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
        MATRIX_SCALAR_MUL(1, cCache, mp_arr[5]); 

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
    return "GenericMatSum"; 
  }

  // Destructor
  V_DTR(~GenericMatSum()) = default;
};

// Left is Type (scalar) and right is a matrix
template <typename T, typename... Callables>
class GenericMatScalarSum : public IMatrix<GenericMatScalarSum<T, Callables...>> {
private:
  // Resources
  Type m_left{};
  T* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericMatScalarSum)
    DISABLE_MOVE(GenericMatScalarSum)
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
  constexpr GenericMatScalarSum(Type u, T* v, Callables&&... call) : m_left{u}, 
                                                                     mp_right{v}, 
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

   // Clone matrix expression
  constexpr const auto& cloneExp() const {
    return m_left + (*mp_right);
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

    // Matrix-Scalar addition computation (Policy design)
    MATRIX_SCALAR_ADD(m_left, right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Right matrix derivative
    mp_arr[1] = mp_right->devalF(X);
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

      MATRIX_SCALAR_MUL(1, eye_n, mp_arr[2]);

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
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], val, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = val;
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
        
        MATRIX_SCALAR_MUL(1, cCache, mp_arr[3]); 
        const auto mp_arr3_val = (*mp_arr[3])(0,0);

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
    return "GenericMatScalarSum"; 
  }

  // Destructor
  V_DTR(~GenericMatScalarSum()) = default;
};

// Left is Expression/Variable/Parameter (scalar) and right is a matrix
template <typename T1, typename T2, typename... Callables>
class GenericMatScalarSumExp : public IMatrix<GenericMatScalarSumExp<T1, T2, Callables...>> {
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericMatScalarSumExp)
    DISABLE_MOVE(GenericMatScalarSumExp)
  #endif

  // All matrices
  inline static constexpr const size_t m_size{9};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache{};
  OMPair m_cache_v; 

  // Constructor
  constexpr GenericMatScalarSumExp(T1* u, T2* v, Callables&&... call) : mp_left{u}, 
                                                                        mp_right{v}, 
                                                                        m_caller{std::make_tuple(
                                                                        std::forward<Callables>(call)...)},
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

   // Clone matrix expression
  constexpr const auto& cloneExp() const {
    return (*mp_left) + (*mp_right);
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Get raw pointers to result and right matrices
    const Matrix<Type>* right_mat = mp_right->eval();
    const Type val = CoolDiff::TensorR1::Eval((*mp_left));

    std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                   
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    // Matrix-Scalar addition computation (Policy design)
    MATRIX_SCALAR_ADD(val, right_mat, mp_arr[0]);

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

    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();
    const size_t nrows_f = getNumRows();
    const size_t ncols_f = getNumColumns();

    // Derivative of expression w.r.t to variable matrix (Reverse mode)
    CoolDiff::TensorR2::Details::DevalR((*mp_left), X, mp_arr[2]);

    // Kronecker product with ones and add with right derivatives
    MATRIX_KRON(CoolDiff::TensorR2::MatrixBasics::Ones(nrows_f, ncols_f), mp_arr[2], mp_arr[3]);
    MATRIX_ADD(mp_arr[3], dright_mat, mp_arr[1]);

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
      
      const size_t n = mp_right->getNumRows();
      const auto eye_n = const_cast<MatType*>(CoolDiff::TensorR2::MatrixBasics::Eye(n));

      /* IMPORTANT: The derivative is computed here */
      MATRIX_SCALAR_MUL(n, eye_n, mp_arr[4]);
      MATRIX_SCALAR_MUL(1, eye_n, mp_arr[5]);
      
      const auto mp_arr4_val = (*mp_arr[4])(0,0);
      const auto mp_arr5_val = (*mp_arr[5])(0,0);

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
      std::for_each(EXECUTION_PAR m_cache_v.begin(), m_cache_v.end(), 
                      [&](const auto& item) {
                      const size_t rows = mp_arr[4]->getNumRows();
                      const size_t cols = mp_arr[4]->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(mp_arr4_val*val, CoolDiff::TensorR2::MatrixBasics::Eye(1), ptr);
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = ptr;
                      }
      });
  
      // Modify cache for right node
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                    [&](const auto& item) {
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
    } else {
      // Cached value
      if(auto it = cache->find(m_nidx); it != cache->end()) {
        const auto cCache = it->second;

        // Traverse right node
        if (false == mp_right->m_visited) {
          mp_right->traverse(cache);
        }

        const size_t rows = cCache->getNumRows(); 
        const size_t cols = cCache->getNumColumns();

        /* IMPORTANT: The derivative is computed here */
        MATRIX_MUL(CoolDiff::TensorR2::MatrixBasics::Ones(1, rows), cCache, mp_arr[6]);
        MATRIX_MUL(mp_arr[6], CoolDiff::TensorR2::MatrixBasics::Ones(cols, 1), mp_arr[7]);

        MATRIX_SCALAR_MUL(1, cCache, mp_arr[8]); 

        const auto mp_arr7_val = (*mp_arr[7])(0,0);
        const auto mp_arr8_val = (*mp_arr[8])(0,0);

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
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),
                      mp_right->m_cache.end(), [&](const auto &item) {
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
      }
    }

    // If scalar reverse derivative cache is empty, run it once
    if(true == m_cache_v.empty()) {
      Expression exp{*mp_left};
      CoolDiff::TensorR1::PreComp(exp);
      m_cache_v = exp.m_cache;
    }

    // Traverse right nodes
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
    return "GenericMatScalarSumExp";
  }

  // Destructor
  V_DTR(~GenericMatScalarSumExp()) = default;
};

// GenericMatSum with 2 typename and callables
template <typename T1, typename T2>
using GenericMatSumT = GenericMatSum<T1, T2, OpMatType>;

// GenericMatScalarSum with 1 typename and callables
template <typename T>
using GenericMatScalarSumT = GenericMatScalarSum<T, OpMatType>;

// GenericMatScalarSumExp with 2 typename and callables
template <typename T1, typename T2>
using GenericMatScalarSumExpT = GenericMatScalarSumExp<T1, T2, OpMatType>;

// Function for sum computation
template <typename T1, typename T2>
constexpr const auto& operator+(const IMatrix<T1>& u, const IMatrix<T2>& v) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  auto tmp = Allocate<GenericMatSumT<T1, T2>>(const_cast<T1*>(static_cast<const T1*>(&_u)),
                                              const_cast<T2*>(static_cast<const T2*>(&_v)), 
                                              OpMatObj);
  return *tmp;
}

// Function for sum computation
template <typename T>
constexpr const auto& operator+(Type u, const IMatrix<T>& v) {
  const auto& _v = v.cloneExp();
  auto tmp = Allocate<GenericMatScalarSumT<T>>(u, const_cast<T*>(static_cast<const T*>(&_v)), OpMatObj);
  return *tmp;
}

template <typename T>
constexpr const auto& operator+(const IMatrix<T>& v, Type u) {
  const auto& _v = v.cloneExp();
  return u + _v;
}

// Matrix sum with scalar (LHS) - SFINAE'd
template <typename T1, typename T2, typename = CoolDiff::TensorR1::Details::IsPureMetaVariableType<T1>>
constexpr const auto& operator+(const T1& v, const IMatrix<T2>& u) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();

  auto tmp = Allocate<GenericMatScalarSumExpT<T1, T2>>(const_cast<T1*>(static_cast<const T1*>(&_v)),
                                                      const_cast<T2*>(static_cast<const T2*>(&_u)), 
                                                      OpMatObj);
  return *tmp;
}

// Matrix sum with scalar (RHS) - SFINAE'd
template <typename T1, typename T2, typename = CoolDiff::TensorR1::Details::IsPureMetaVariableType<T2>>
constexpr const auto& operator+(const IMatrix<T1>& u, const T2& v) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  return _v + _u;
}
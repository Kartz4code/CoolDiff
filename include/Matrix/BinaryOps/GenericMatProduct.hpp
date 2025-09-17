/**
 * @file include/Matrix/BinaryOps/GenericMatProduct.hpp
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
#include "MatrixHelper.hpp"

// Left/right side is a Matrix
template <typename T1, typename T2, typename... Callables>
class GenericMatProduct : public IMatrix<GenericMatProduct<T1, T2, Callables...>> {
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatProduct)
  DISABLE_MOVE(GenericMatProduct)

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
  inline static constexpr const size_t m_size{20};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache;

  // Constructor
  constexpr GenericMatProduct(T1* u, T2* v, Callables&&... call) : mp_left{u}, 
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

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix multiplication dimensions mismatch");

    // Get raw pointers to result, left and right matrices
    const Matrix<Type>* left_mat = mp_left->eval();
    const Matrix<Type>* right_mat = mp_right->eval();

    std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [&](Matrix<Type>*& m) {
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
    
    std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [&](Matrix<Type>*& m) {
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
    MATRIX_KRON(left_mat, Eye(nrows_x), mp_arr[4]);
    // R (X) I - Right matrix and identity Kronocke product (Policy design)
    MATRIX_KRON(right_mat, Eye(ncols_x), mp_arr[5]);

    // Product with left and right derivatives (Policy design)
    MATRIX_MUL(mp_arr[4], dright_mat, mp_arr[2]);
    MATRIX_MUL(dleft_mat, mp_arr[5], mp_arr[3]);

    // Addition between left and right derivatives (Policy design)
    MATRIX_ADD(mp_arr[2], mp_arr[3], mp_arr[1]);

    // Return derivative result pointer
    return mp_arr[1];
  }

  virtual void traverse(OMMatPair* cache = nullptr) override {
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

      // Modify cache for left node
      std::for_each(EXECUTION_PAR mp_left->m_cache.begin(), mp_left->m_cache.end(), 
                    [&](const auto& item) {
                      const auto idx = item.first; const auto val = item.second;
                      
                      const size_t nrows1 = val->getNumRows();
                      const size_t ncols1 = val->getNumColumns();
                      const size_t nrows2 = mp_arr[6]->getNumRows();
                      const size_t ncols2 = mp_arr[6]->getNumColumns();
                      
                      if(ncols1 == nrows2) {
                        MATRIX_MUL(val, mp_arr[6], mp_arr[8]);
                      } else {
                        MATRIX_MUL(mp_arr[6], val, mp_arr[8]);
                      }

                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], mp_arr[8], (*cache)[idx]);
                      } else {
                        (*cache)[idx] = mp_arr[8];
                      }
      });
  
      // Modify cache for right node
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                    [&](const auto& item) {
                      const auto idx = item.first; const auto val = item.second;

                      const size_t nrows1 = val->getNumRows();
                      const size_t ncols1 = val->getNumColumns();
                      const size_t nrows2 = mp_arr[7]->getNumRows();
                      const size_t ncols2 = mp_arr[7]->getNumColumns();

                      if(ncols1 == nrows2) {
                        MATRIX_MUL(val, mp_arr[7], mp_arr[9]);
                      } else {
                        MATRIX_MUL(mp_arr[7], val, mp_arr[9]);
                      }

                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], mp_arr[9], (*cache)[idx]);
                      } else {
                        (*cache)[idx] = mp_arr[9];
                      }
      });
    } else {
      // Traverse left node
      if (false == mp_left->m_visited) {
        mp_left->traverse(cache);
      }
      // Traverse right node
      if (false == mp_right->m_visited) {
        mp_right->traverse(cache);
      }

      // Cached value
      if(auto it = cache->find(m_nidx); it != cache->end()) {
        const auto cCache = it->second;

        // Get raw pointers to result, left and right matrices
        const Matrix<Type>* left_mat = mp_left->eval();
        const Matrix<Type>* right_mat = mp_right->eval();

        /* IMPORTANT: The derivative is computed here */
        MATRIX_TRANSPOSE(left_mat, mp_arr[10]);
        MATRIX_TRANSPOSE(right_mat, mp_arr[11]);  

        MATRIX_MUL(cCache, mp_arr[11], mp_arr[12]);
        MATRIX_MUL(mp_arr[10], cCache, mp_arr[13]);

        if(auto it2 = cache->find(mp_left->m_nidx); it2 != cache->end()) {
          MATRIX_ADD((*cache)[mp_left->m_nidx], mp_arr[12], (*cache)[mp_left->m_nidx]); 
        } else {
          (*cache)[mp_left->m_nidx] = mp_arr[12];
        }

        if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
          MATRIX_ADD((*cache)[mp_right->m_nidx], mp_arr[13], (*cache)[mp_right->m_nidx]); 
        } else {
          (*cache)[mp_right->m_nidx] = mp_arr[13];
        }

        // Modify cache for left node
        std::for_each(EXECUTION_PAR mp_left->m_cache.begin(), mp_left->m_cache.end(), 
                      [&](const auto& item) {
                        const auto idx = item.first; const auto val = item.second;

                        
                        const size_t nrows1 = val->getNumRows();
                        const size_t ncols1 = val->getNumColumns();
                        const size_t nrows2 = mp_arr[12]->getNumRows();
                        const size_t ncols2 = mp_arr[12]->getNumColumns();
                        
                        if(ncols1 == nrows2) {
                          MATRIX_MUL(val, mp_arr[12], mp_arr[14]);
                        } else if (ncols2 == nrows1) {
                          MATRIX_MUL(mp_arr[12], val, mp_arr[14]);
                        }


                        if(auto it2 = cache->find(idx); it2 != cache->end()) {
                          MATRIX_ADD((*cache)[idx], mp_arr[14], (*cache)[idx]);
                        } else {
                          (*cache)[idx] = val;
                        }
        });
      
        // Modify cache for right node
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),
                      mp_right->m_cache.end(), [&](const auto &item) {
                        const auto idx = item.first; const auto val = item.second;

                        if(auto it2 = cache->find(idx); it2 != cache->end()) {
                          const size_t nrows1 = val->getNumRows();
                          const size_t ncols1 = val->getNumColumns();
                          const size_t nrows2 = mp_arr[13]->getNumRows();
                          const size_t ncols2 = mp_arr[13]->getNumColumns();

                          if(ncols1 == nrows2) {
                            MATRIX_MUL(val, mp_arr[13], mp_arr[15]);
                          } else if (ncols2 == nrows1) {
                            MATRIX_MUL(mp_arr[13], val, mp_arr[15]);
                          }

                          MATRIX_ADD((*cache)[idx], mp_arr[15], (*cache)[idx]);
                        } else {
                          (*cache)[idx] = val;
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

  virtual OMMatPair& getCache() override {
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
template <typename T, typename... Callables>
class GenericMatScalarProduct : public IMatrix<GenericMatScalarProduct<T, Callables...>> {
private:
  // Resources
  Type m_left{};
  T* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatScalarProduct)
  DISABLE_MOVE(GenericMatScalarProduct)

  // All matrices
  inline static constexpr const size_t m_size{2};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache;

  // Constructor
  constexpr GenericMatScalarProduct(Type u, T* v, Callables&&... call) : m_left{u}, 
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
template <typename T1, typename T2, typename... Callables>
class GenericMatScalarProductExp : public IMatrix<GenericMatScalarProductExp<T1, T2, Callables...>> {
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatScalarProductExp)
  DISABLE_MOVE(GenericMatScalarProductExp)

  // All matrices
  inline static constexpr const size_t m_size{7};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache;

  // Constructor
  constexpr GenericMatScalarProductExp(T1* u, T2* v, Callables&&... call) : mp_left{u}, 
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

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Get raw pointers to result and right matrices
    const Matrix<Type>* right_mat = mp_right->eval();

    std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    const Type val = Eval((*mp_left));

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
    DevalR((*mp_left), X, mp_arr[2]);

    // Evaluate left expression
    const Type val = Eval((*mp_left));

    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();
    const size_t nrows_f = getNumRows();
    const size_t ncols_f = getNumColumns();


    MATRIX_KRON(Ones(nrows_f, ncols_f), mp_arr[2], mp_arr[5]);
    MATRIX_KRON(right_mat, Ones(nrows_x, ncols_x), mp_arr[6]);

    // Product with left and right derivatives (Policy design)
    MATRIX_SCALAR_MUL(val, dright_mat, mp_arr[4]);
    MATRIX_HADAMARD(mp_arr[5], mp_arr[6], mp_arr[3]);

    // Addition between left and right derivatives (Policy design)
    MATRIX_ADD(mp_arr[4], mp_arr[3], mp_arr[1]);

    // Return result pointer
    return mp_arr[1];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_MAT_RIGHT_RESET(); 
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
using GenericMatProductT = GenericMatProduct<T1, T2, OpMatType>;

// GenericMatScalarProduct with 1 typename and callables
template <typename T>
using GenericMatScalarProductT = GenericMatScalarProduct<T, OpMatType>;

// GenericMatScalarProductExp with 2 typename and callables
template <typename T1, typename T2>
using GenericMatScalarProductExpT = GenericMatScalarProductExp<T1, T2, OpMatType>;

// Function for product computation
template <typename T1, typename T2>
constexpr const auto& operator*(const IMatrix<T1>& u, const IMatrix<T2>& v) {
  auto tmp = Allocate<GenericMatProductT<T1, T2>>(const_cast<T1*>(static_cast<const T1*>(&u)),
                                                  const_cast<T2*>(static_cast<const T2*>(&v)), 
                                                  OpMatObj);
  return *tmp;
}

// Function for product computation
template <typename T>
constexpr const auto& operator*(Type u, const IMatrix<T>& v) {
  auto tmp = Allocate<GenericMatScalarProductT<T>>(u, const_cast<T *>(static_cast<const T*>(&v)), OpMatObj);
  return *tmp;
}

template <typename T>
constexpr const auto& operator*(const IMatrix<T>& v, Type u) {
  return (u * v);
}

template <typename T>
constexpr const auto& operator/(const IMatrix<T>& v, Type u) {
  return (((Type)(1)/u) * v);
}

template <typename T1, typename T2>
constexpr const auto& operator/(const IMatrix<T1>& u, const IMatrix<T2>& v) {
  return (u*inv(v));
}

// Matrix multiplication with scalar (LHS) - SFINAE'd
template <typename T1, typename T2, typename = ExpType<T1>>
constexpr const auto& operator*(const T1& v, const IMatrix<T2>& u) {
  auto tmp = Allocate<GenericMatScalarProductExpT<T1, T2>>(const_cast<T1*>(static_cast<const T1*>(&v)),
                                                           const_cast<T2*>(static_cast<const T2*>(&u)), 
                                                           OpMatObj);
  return *tmp;
}

template <typename T1, typename T2, typename = ExpType<T1>>
constexpr const auto& operator/(const IMatrix<T2>& u, const T1& v) {
  return (u * (1/v));
}

// Matrix sum with scalar (RHS) - SFINAE'd
template <typename T1, typename T2, typename = ExpType<T2>>
constexpr const auto& operator*(const IMatrix<T1>& u, const T2& v) {
  return (v * u);
}
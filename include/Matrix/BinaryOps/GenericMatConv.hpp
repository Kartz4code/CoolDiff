/**
 * @file include/Matrix/BinaryOps/GenericMatConv.hpp
 *
 * @copyright 2023-2026 Karthik Murali Madhavan Rathai
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
class GenericMatConv : public IMatrix<GenericMatConv<T1, T2>> {
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  const size_t m_stride_x{1}, m_stride_y{1};
  const size_t m_pad_x{0}, m_pad_y{0};

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericMatConv)
    DISABLE_MOVE(GenericMatConv)
  #endif

  // Verify dimensions of result matrix for convolution operation
  inline constexpr bool verifyDim() const {
    // Dimension of the result of convolution operator must be strictly non-negative
    return ((int)getNumRows() > 0 || (int)getNumColumns() > 0);
  }

  // All matrices
  inline static constexpr const size_t m_size{4};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache;

  // Constructor
  GenericMatConv( T1* u, T2* v, const size_t stride_x, const size_t stride_y,
                  const size_t pad_x, const size_t pad_y  ) : mp_left{u}, 
                                                              mp_right{v}, 
                                                              m_stride_x{stride_x}, 
                                                              m_stride_y{stride_y},
                                                              m_pad_x{pad_x}, 
                                                              m_pad_y{pad_y}, 
                                                              m_nidx{this->m_idx_count++} {
    // Stride must be strictly non-negative
    ASSERT(((int)m_stride_x > 0) && ((int)m_stride_y > 0), "Stride is not strictly non-negative");
    // Padding must be positive
    ASSERT(((int)pad_x >= 0) && ((int)pad_y >= 0), "Stride is not positive");
    // Fill with nullptrs
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows (After convolution)
  V_OVERRIDE(size_t getNumRows() const) {
    return (((mp_left->getNumRows() + (2 * m_pad_x) - mp_right->getNumRows()) / m_stride_x) + 1);
  }

  // Get number of columns (After convolution)
  V_OVERRIDE(size_t getNumColumns() const) {
    return (((mp_left->getNumColumns() + (2 * m_pad_y) - mp_right->getNumColumns()) / m_stride_y) + 1);
  }

  // Find me
  bool findMe(void* v) const { 
    BINARY_FIND_ME(); 
  }

  // Clone matrix expression
  constexpr const auto& cloneExp() const {
    return conv(*mp_left, *mp_right, m_stride_x, m_stride_y, m_pad_x, m_pad_y);
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix convolution dimensions invalid");

    // Get raw pointers to result, left and right matrices
    const Matrix<Type>* left_mat = mp_left->eval();
    const Matrix<Type>* right_mat = mp_right->eval();

    std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((left_mat == m) ? nullptr : m);
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    });

    // Matrix convolution
    MATRIX_CONV(m_stride_x, m_stride_y, m_pad_x, m_pad_y, left_mat, right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix convolution dimensions invalid");

    // Left and right matrices derivatives
    Matrix<Type>* dleft_mat = mp_left->devalF(X);
    Matrix<Type>* dright_mat = mp_right->devalF(X);

    // Left and right matrices evaluation
    Matrix<Type>* left_mat = mp_left->eval();
    Matrix<Type>* right_mat = mp_right->eval();

    std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [&](Matrix<Type>*& m) {
      if (nullptr != m) {                                                        
        m = ((dleft_mat == m) ? nullptr : m);
        m = ((dright_mat == m) ? nullptr : m);
        m = ((left_mat == m) ? nullptr : m);
        m = ((right_mat == m) ? nullptr : m);                                                               
      }                                                                          
    }); 
    
    // Derivative matrix dimensions
    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();

    // Matrix convolution derivative
    MATRIX_DERV_CONV(nrows_x, ncols_x, m_stride_x, m_stride_y, m_pad_x, m_pad_y,
                    left_mat, dleft_mat, right_mat, dright_mat, mp_arr[1]);

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

      const auto mp_right_val = CoolDiff::TensorR2::Details::ScalarSpl(right_mat);
      const auto mp_left_val = CoolDiff::TensorR2::Details::ScalarSpl(left_mat);

      if(auto it2 = cache->find(mp_left->m_nidx); it2 != cache->end()) {
        MATRIX_ADD((*cache)[mp_left->m_nidx], right_mat, (*cache)[mp_left->m_nidx]); 
      } else {
        (*cache)[mp_left->m_nidx] = const_cast<Matrix<Type>*>(right_mat);
      }

      if(auto it2 = cache->find(mp_right->m_nidx); it2 != cache->end()) {
        MATRIX_ADD((*cache)[mp_right->m_nidx], left_mat, (*cache)[mp_right->m_nidx]); 
      } else {
        (*cache)[mp_right->m_nidx] = const_cast<Matrix<Type>*>(left_mat);
      }

      // Clone the cache
      for(const auto& [k,v] : (*cache)) {
        (*cache)[k] = v->clone(this->m_cloned[this->incFunc()]);
      }

      // Modify cache for left node
      std::for_each(EXECUTION_PAR mp_left->m_cache.begin(), mp_left->m_cache.end(), 
                      [&](const auto& item) {
                      const size_t rows = right_mat->getNumRows();
                      const size_t cols = right_mat->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(mp_right_val, val, ptr);
                      if(auto it2 = cache->find(idx); it2 != cache->end()) {
                        MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                      } else {
                        (*cache)[idx] = ptr;
                      }
      });
  
      // Modify cache for right node
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                    [&](const auto& item) {
                      const size_t rows = left_mat->getNumRows();
                      const size_t cols = left_mat->getNumColumns(); 
                      ASSERT((rows == 1) && (cols == 1), "Matrix expression not scalar for reverse mode derivative"); 

                      const auto idx = item.first; const auto val = item.second;
                      MatType*& ptr = this->m_cloned[this->incFunc()];
                      MATRIX_SCALAR_MUL(mp_left_val, val, ptr);
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

        // Right matrix rows and columns size (Kernel)
        const size_t rrows = right_mat->getNumRows();
        const size_t rcols = right_mat->getNumColumns();
        
        // Left matrix rows and columns size (Input)
        const size_t lrows = left_mat->getNumRows();
        const size_t lcols = left_mat->getNumColumns();

        // Cache rows and columns size (Pullback)
        const size_t crows = cCache->getNumRows();
        const size_t ccols = cCache->getNumColumns(); 

        /* IMPORTANT: The derivative is computed here */
        // Typically, right_mat will be less than cCache dimension, so no need for padding
        MATRIX_CONV(m_stride_x, m_stride_y, m_pad_x, m_pad_y, cCache, right_mat, mp_arr[2]);

        // Find the best number of zero padding on the input to match the output dimensions (right_mat)
        const size_t px = (m_stride_x*(rrows - 1) + crows - lrows)/2; 
        const size_t py = (m_stride_y*(rcols - 1) + ccols - lcols)/2;

        MATRIX_CONV(m_stride_x, m_stride_y, px, py, left_mat, cCache, mp_arr[3]);

        const auto mp_arr2_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[2]);
        const auto mp_arr3_val = CoolDiff::TensorR2::Details::ScalarSpl(mp_arr[3]);

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
                        MatType*& ptr = this->m_cloned[this->incFunc()];
                        MATRIX_SCALAR_MUL(mp_arr2_val, val, ptr);
                        if(auto it2 = cache->find(idx); it2 != cache->end()) {
                          MATRIX_ADD((*cache)[idx], ptr, (*cache)[idx]);
                        } else {
                          (*cache)[idx] = ptr;
                        }
        });
      
        // Modify cache for right node
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),
                      mp_right->m_cache.end(), [&](const auto &item) {
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
    return "GenericMatConv"; 
  }

  // Destructor
  V_DTR(~GenericMatConv()) = default;
};

// GenericMatConv with 2 typename callables
template <typename T1, typename T2>
using GenericMatConvT = GenericMatConv<T1, T2>;

// Function for sub computation
template <typename T1, typename T2>
constexpr const auto& conv( const IMatrix<T1>& u, const IMatrix<T2>& v,
                            const size_t stride_x = 1, const size_t stride_y = 1,
                            const size_t pad_x = 0, const size_t pad_y = 0  ) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  auto tmp = Allocate<GenericMatConvT<T1, T2>>( const_cast<T1*>(static_cast<const T1*>(&_u)),
                                                const_cast<T2*>(static_cast<const T2*>(&_v)), 
                                                stride_x, stride_y, pad_x, pad_y );
  return *tmp;
}
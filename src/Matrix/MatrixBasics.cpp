/**
 * @file src/Matrix/MatrixBasics.cpp
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

#include "MatrixBasics.hpp"
#include "Matrix.hpp"

namespace CoolDiff {
    namespace TensorR2 {
        namespace MatrixBasics {
          // Numerical Eye matrix
          const Matrix<Type>* Eye(std::string_view alloc_type, const size_t n) {
            // Eye matrix registry
            static Map<std::string_view, UnOrderedMap<size_t, Matrix<Type>*>> eye_register_base;

            // Get the registry for the particular allocator type
            auto& eye_register = eye_register_base[alloc_type];

            // Result matrix
            Matrix<Type>* result{nullptr};

            // Find in registry of special matrices
            if (auto it = eye_register.find(n); it != eye_register.end()) {
              return it->second;
            } else {

              // Pool matrix
              MemoryManager::MatrixPool(result, n, n, alloc_type);

              // Vector of indices
              const auto idx = CoolDiff::Common::Range<size_t>(0, n);
              std::for_each(EXECUTION_PAR idx.begin(), idx.end(), [&](const size_t i) { (*result)(i, i) = (Type)(1); });

              // Register and return result
              eye_register[n] = result;
              return result;
            }
          }
          const Matrix<Type>& EyeRef(std::string_view alloc_type, const size_t n) {
            return *Eye(alloc_type, n);
          }

          // Numerical Zero matrix
          const Matrix<Type>* Zeros(std::string_view alloc_type, const size_t n, const size_t m) {
            // Zeros matrix registry
            static Map<std::string_view, UnOrderedMap<Pair<size_t, size_t>, Matrix<Type>*>> zeros_register_base;

            // Get the registry for the particular allocator type
            auto& zeros_register = zeros_register_base[alloc_type];

            // Result matrix
            Matrix<Type>* result{nullptr};

            // Find in registry of special matrices
            if (auto it = zeros_register.find({n, m}); it != zeros_register.end()) {
              return it->second;
            } else {
              // Pool matrix
              MemoryManager::MatrixPool(result, n, m, alloc_type);
              // Register and return result
              zeros_register[{n, m}] = result;
              return result;
            }
          }
          const Matrix<Type>* Zeros(std::string_view alloc_type, const size_t n) { 
            return Zeros(alloc_type, n, n); 
          }
          const Matrix<Type>& ZerosRef(std::string_view alloc_type, const size_t n, const size_t m) {
            return *Zeros(alloc_type, n,m);
          }
          const Matrix<Type>& ZerosRef(std::string_view alloc_type, const size_t n) {
            return ZerosRef(alloc_type, n, n);
          }

          // Numerical Ones matrix
          const Matrix<Type>* Ones(std::string_view alloc_type, const size_t n, const size_t m) {
            // Zeros matrix registry
            static Map<std::string_view, UnOrderedMap<Pair<size_t, size_t>, Matrix<Type>*>> ones_register_base;

            // Get the registry for the particular allocator type
            auto& ones_register = ones_register_base[alloc_type];

            // Result matrix
            Matrix<Type>* result{nullptr};

            // Find in registry of special matrices
            if (auto it = ones_register.find({n, m}); it != ones_register.end()) {
              return it->second;
            } else {
              // Pool matrix
              MemoryManager::MatrixPool(result, n, m, alloc_type, (Type)1);
              // Register and return result
              ones_register[{n, m}] = result;
              return result;
            }
          }
          const Matrix<Type>* Ones(std::string_view alloc_type, const size_t n) { 
            return Ones(alloc_type, n, n); 
          }
          const Matrix<Type>& OnesRef(std::string_view alloc_type, const size_t n, const size_t m) {
            return *Ones(alloc_type, n,m);
          }
          const Matrix<Type>& OnesRef(std::string_view alloc_type, const size_t n) {
            return OnesRef(alloc_type, n,n);
          }

        }
    }
}


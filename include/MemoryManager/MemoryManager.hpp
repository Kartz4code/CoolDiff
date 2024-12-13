/**
 * @file include/MemoryManager/MemoryManager.hpp
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

#include "MetaMatrix.hpp"
#include "MetaVariable.hpp"

#include "IMatrix.hpp"

class MemoryManager {
private:
  // Allocate friend function
  template <typename T, typename... Args>
  friend SharedPtr<T> Allocate(Args &&...args);

  // Vector of deleted or to be deleted MetaVariable resources
  inline static Vector<SharedPtr<MetaVariable>> m_del_ptr;

  // Vector of deleted or to be deleted MetaMatrix resources
  inline static Vector<SharedPtr<MetaMatrix>> m_del_mat_ptr;
  // A special vector for Matrix<Type> which is used dynamically throughout the
  // process
  inline static Vector<SharedPtr<Matrix<Type>>> m_del_mat_type_ptr;

public:
  // Get size of memory allocated
  static size_t size();

  // Special matrix pool allocation
  static Matrix<Type> *MatrixSplPool(const size_t, const size_t,
                                     const MatrixSpl &);
  // Matrix pool allocation
  static void MatrixPool(const size_t, const size_t, Matrix<Type> *&,
                         const Type & = (Type)0);
};

// Delete resource
template <typename T> void DelPtr(T *ptr) {
  if (nullptr != ptr) {
    delete ptr;
    ptr = nullptr;
  }
}

// Scalar allocator
template <typename T, typename... Args>
inline SharedPtr<T> Allocate(Args &&...args) {
  const size_t size = sizeof(T);
  const size_t align = 0;

  // Allocate with custom deleter
  SharedPtr<T> tmp{new T(std::forward<Args>(args)...), DelPtr<T>};

  // Push the allocated object into stack to clear it later
  if constexpr (std::is_base_of_v<MetaVariable, T>) {
    MemoryManager::m_del_ptr.push_back(tmp);
  } else if constexpr (std::is_base_of_v<MetaMatrix, T>) {
    if constexpr (std::is_same_v<T, Matrix<Type>>) {
      MemoryManager::m_del_mat_type_ptr.push_back(tmp);
    } else {
      MemoryManager::m_del_mat_ptr.push_back(tmp);
    }
  }

  return tmp;
}
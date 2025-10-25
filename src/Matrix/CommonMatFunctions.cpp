/**
 * @file src/Matrix/CommonMatFunctions.cpp
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

#include "CommonMatFunctions.hpp"

namespace CoolDiff {
   namespace Common {
      const size_t Factorial(const size_t n) {
         static UnOrderedMap<size_t, size_t> map{{1,1}}; 
         if(auto it = map.find(n); it == map.end()) {
            map[n] = n*Factorial(n-1); 
         }
         return map[n]; 
      }
   }
}

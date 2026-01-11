/**
 * @file include/Matrix/GlobalParameters.hpp
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

#include <unordered_map>

#if defined(USE_CUDA_BACKEND)
  #include <cuda_runtime.h>
#endif

// Global parameters
namespace CoolDiff {
  class GlobalParameters {
      public:
          // Backend handlers
          enum class HandlerType {
              EIGEN, CUDA
          };

      private:
          // Handler 
          inline static HandlerType m_ht{HandlerType::EIGEN};

      public:
          // Getters and setters
          static void setHandler(HandlerType);
          static HandlerType getHandler();
       
      #if defined(USE_CUDA_BACKEND)    
        private:     
            // Map of device and properties 
            inline static std::unordered_map<int, cudaDeviceProp> m_device_prop;
            
        public:
            // Getters of GPU properties
            static int getNumGPUDevices(); 
            static cudaDeviceProp getDeviceProperties(int);
      #endif
  };
}

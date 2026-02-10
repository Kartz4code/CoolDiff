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
#include <string_view>

#if defined(USE_CUDA_BACKEND)
  #include <cuda_runtime.h>
#endif

// Global parameters
namespace CoolDiff {
  class GlobalParameters {
      public:
        // CPU backend handlers
        enum class CPUHandlerType {
            EIGEN, NAIVE
        };

        // GPU backend handlers
        enum class GPUHandlerType {
            CUDA    
        };

      private:
          // Handler 
          inline static CPUHandlerType m_ht_cpu{CPUHandlerType::EIGEN};
          inline static GPUHandlerType m_ht_gpu{GPUHandlerType::CUDA};

      public:
          // Getters and setters
          static void setCPUHandler(CPUHandlerType);
          static CPUHandlerType getCPUHandler();

          static void setGPUHandler(GPUHandlerType);
          static GPUHandlerType getGPUHandler();

          // Is the memory strategy in CPU/GPU space
          static bool isCPUSpace(std::string_view);
          static bool isGPUSpace(std::string_view);
        
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

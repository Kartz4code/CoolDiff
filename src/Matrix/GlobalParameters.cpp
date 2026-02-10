/**
 * @file src/Matrix/GlobalParameters.hpp
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

 #include "GlobalParameters.hpp"
 
namespace CoolDiff {
  // Set CPU handler
  void GlobalParameters::setCPUHandler(CPUHandlerType ht) {
    m_ht_cpu = ht;
  }

  // Get CPU handler
  GlobalParameters::CPUHandlerType GlobalParameters::getCPUHandler() {
    return m_ht_cpu;
  }

  // Set GPU handler
  void GlobalParameters::setGPUHandler(GPUHandlerType ht) {
    m_ht_gpu = ht;
  }

  // Get GPU handler
  GlobalParameters::GPUHandlerType GlobalParameters::getGPUHandler() {
    return m_ht_gpu;
  }

  // Is the memory strategy in CPU
  bool GlobalParameters::isCPUSpace(std::string_view strategy) {
    if("CPUMemoryStrategy" == strategy) {
        return true;
    } else {
        return false;
    }
  }

  // Is the memory strategy in GPU space
  bool GlobalParameters::isGPUSpace(std::string_view strategy) {
    if("GPUPinnedMemoryStrategy" == strategy) {
        return true;
    } else {
        return false;
    }
  }

  #if defined(USE_CUDA_BACKEND) 
    // Get number of GPU devices
    int GlobalParameters::getNumGPUDevices() {
      int ndevices{}; cudaGetDeviceCount(&ndevices);
      return ndevices;
    }

    // Get device properties
    cudaDeviceProp GlobalParameters::getDeviceProperties(int i) {
      if(auto it = m_device_prop.find(i); it != m_device_prop.end()) {
        return m_device_prop[i];
      } else {
        cudaDeviceProp cdp; cudaGetDeviceProperties(&cdp, i);
        m_device_prop[i] = cdp; 
        return cdp; 
      }
    }
  #endif
}
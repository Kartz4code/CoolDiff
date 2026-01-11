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
 
// Set handler
namespace CoolDiff {
  void GlobalParameters::setHandler(HandlerType ht) {
    m_ht = ht;
  }

  // Get handler
  GlobalParameters::HandlerType GlobalParameters::getHandler() {
    return m_ht;
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
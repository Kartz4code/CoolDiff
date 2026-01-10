/**
 * @file include/Matrix/UnaryFunctions.hpp
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

#include <cuda.h>
#include <cuda_runtime.h>

#include <unordered_map>
#include <cmath>

// Function pointer of unary functions
template<typename T>
using FunctionTypeCuda = T(*)(T);

// Sin function
template<typename T>
__host__ __device__ T Sin(T x) { 
    return (T)(std::sin(x)); 
}

// Derivative of Sin function
template<typename T>
__host__ __device__ T DSin(T x) { 
    return (T)(std::cos(x)); 
}

// Cos function
template<typename T>
__host__ __device__ T Cos(T x) { 
    return (T)(std::cos(x)); 
}

// Derivative of Cos function 
template<typename T>
__host__ __device__ T DCos(T x) { 
    return (T)(-1*std::sin(x)); 
}

// Exp function
template<typename T>
__host__ __device__ T Exp(T x) { 
    return (T)(std::exp(x)); 
}

// Log function
template<typename T>
__host__ __device__ T Log(T x) { 
    return (T)(std::log(x)); 
}

// Derivative of Log function
template<typename T>
__host__ __device__ T DLog(T x) { 
    return (T)((T)1/(x)); 
}

// Square function 
template<typename T>
__host__ __device__ T Sqrt(T x) { 
    return (T)std::sqrt(x); 
}

// Derivative of square function 
template<typename T>
__host__ __device__ T DSqrt(T x) { 
    return (T)((T)(1)/(2*std::sqrt(x))); 
}

// Tanh function 
template<typename T>
__host__ __device__ T Tanh(T x) {  
    return (T)std::tanh(x); 
}

// Derivative of tanh function 
template<typename T>
__host__ __device__ T DTanh(T x) { 
    T y = std::tanh(x);
    return (T)((T)1 - y*y); 
}

// Sigmoid function 
template<typename T>
__host__ __device__ T Sigmoid(T x) { 
    T res = ((T)(1)) / (((T)(1)) + std::exp(-x));
    return res;
}

// Derivative of sigmoid function 
template<typename T>
__host__ __device__ T DSigmoid(T x) { 
    T res = ((T)(1)) / (((T)(1)) + std::exp(-x));
    return (T)(res * (((T)(1)) - res));
}

// ReLU function 
template<typename T>
__host__ __device__ T ReLU(T x) { 
    return (T)((x >= (T)(0)) ? x : 0);
}

// Derivative of ReLU function 
template<typename T>
__host__ __device__ T DReLU(T x) { 
    return (T)((x >= (T)(0)) ? 1 : 0);
}

// Leaky ReLU function 
#define NEG_SLOPE() 0.01

template<typename T>
__host__ __device__ T LeakyReLU(T x) { 
    return (T)((x >= (T)(0)) ? x : NEG_SLOPE()*x);
}

// Derivative of Leaky ReLU function 
template<typename T>
__host__ __device__ T DLeakyReLU(T x) { 
    return (T)((x >= (T)(0)) ? 1 : NEG_SLOPE());
}

// Abs function 
template<typename T>
__host__ __device__ T Abs(T x) { 
    return (T)std::abs(x);
}

// Derivative of Abs function 
template<typename T>
__host__ __device__ T DAbs(T x) { 
    return (T)(((T)(0) < x) - (x < (T)(0)));
}

// List of device functions
enum DEVICE_FUNCTIONS {
    SIN = 0, DSIN, 
    COS, DCOS, 
    EXP, 
    LOG, DLOG,  
    SQRT, DSQRT, 
    TANH, DTANH, 
    SIGMOID, DSIGMOID,
    RELU, DRELU, 
    LEAKYRELU, DLEAKYRELU, 
    ABS, DABS,
    COUNT
};

// List of all device functions
template<typename T>
__device__ FunctionTypeCuda<T> device_functions[DEVICE_FUNCTIONS::COUNT];

// Device function intialization
template<typename T>
__global__ void DeviceFunctionsInit() {
    /*    <----------- Function ----------->                                   <----------- Derivative ----------->           */
    device_functions<T>[DEVICE_FUNCTIONS::SIN]  = Sin<T>;            device_functions<T>[DEVICE_FUNCTIONS::DSIN] = DSin<T>;
    device_functions<T>[DEVICE_FUNCTIONS::COS]  = Cos<T>;            device_functions<T>[DEVICE_FUNCTIONS::DCOS] = DCos<T>;
    device_functions<T>[DEVICE_FUNCTIONS::EXP]  = Exp<T>;
    device_functions<T>[DEVICE_FUNCTIONS::LOG]  = Log<T>;            device_functions<T>[DEVICE_FUNCTIONS::DLOG] = DLog<T>;
    device_functions<T>[DEVICE_FUNCTIONS::SQRT] = Sqrt<T>;           device_functions<T>[DEVICE_FUNCTIONS::DSQRT] = DSqrt<T>;
    device_functions<T>[DEVICE_FUNCTIONS::TANH] = Tanh<T>;           device_functions<T>[DEVICE_FUNCTIONS::DTANH] = DTanh<T>;
    device_functions<T>[DEVICE_FUNCTIONS::SIGMOID] = Sigmoid<T>;     device_functions<T>[DEVICE_FUNCTIONS::DSIGMOID] = DSigmoid<T>;
    device_functions<T>[DEVICE_FUNCTIONS::RELU] = ReLU<T>;           device_functions<T>[DEVICE_FUNCTIONS::DRELU] = DReLU<T>;
    device_functions<T>[DEVICE_FUNCTIONS::LEAKYRELU] = LeakyReLU<T>; device_functions<T>[DEVICE_FUNCTIONS::DLEAKYRELU] = DLeakyReLU<T>;
    device_functions<T>[DEVICE_FUNCTIONS::ABS] = Abs<T>;             device_functions<T>[DEVICE_FUNCTIONS::DABS] = DAbs<T>;
}

// Map of all device functions and its enum
template<typename T>
std::unordered_map<FunctionTypeCuda<T>, int> device_functions_map{  /* <-- Function -->             <-- Derivative-->           */
                                                                    {Sin<T>, SIN},                  {DSin<T>, DSIN}, 
                                                                    {Cos<T>, COS},                  {DCos<T>, DCOS}, 
                                                                    {Exp<T>, EXP},
                                                                    {Log<T>, LOG},                  {DLog<T>, DLOG},
                                                                    {Sqrt<T>, SQRT},                {DSqrt<T>, DSQRT},
                                                                    {Tanh<T>, TANH},                {DTanh<T>, DTANH},
                                                                    {Sigmoid<T>, SIGMOID},          {DSigmoid<T>, DSIGMOID},
                                                                    {ReLU<T>, RELU},                {DReLU<T>, DRELU},
                                                                    {LeakyReLU<T>, LEAKYRELU},      {DLeakyReLU<T>, DLEAKYRELU},
                                                                    {Abs<T>, ABS},                  {DAbs<T>, DABS}};
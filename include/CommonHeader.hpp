#pragma once 

#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <tuple>
#include <thread>
#include <type_traits>
#include <chrono>
#include <cmath>
#include <algorithm> 
#include "Operators.hpp"


#ifndef BUILD_TYPE
    #define SCALAR_TYPE double
    #define USE_COMPLEX_MATH
    #define USE_ROBIN_HOOD_MAP
    #define USE_VIRTUAL_FUNCTIONS
#endif

// Enable/disable copy/move operators
#ifndef ENABLE_COPY_MOVE
    #define DISABLE_COPY(X) X(const X&) = delete;\
                            X& operator=(const X&) = delete;

    #define DISABLE_MOVE(X) X(X&&) noexcept = delete;\
                            X& operator=(X&&) noexcept = delete;
#else
    #define DISABLE_COPY(X)
    #define DISABLE_MOVE(X)
#endif

// Eval/Deval left operator
#define EVAL_L() (*mp_left->symEval())
#define DEVAL_L(X) (*mp_left->symDeval(X))

// Eval/Deval right operator
#define EVAL_R() (*mp_right->symEval())
#define DEVAL_R(X) (*mp_right->symDeval(X))

// Unary reset
#define UNARY_RESET()\
this->m_visited = false;\
if (false == m_cache.empty()) {\
    m_cache.clear();\
}\
MetaVariable::resetTemp();\
if(nullptr != mp_left){\
mp_left->reset();\
}

// Unary find me 
#define UNARY_FIND_ME()\
if(static_cast<void*>(mp_left) == v) {\
    return true;\
} else if (mp_left->findMe(v) == true) {\
    return true;\
}\
return false;

// Binary reset
#define BINARY_RESET()\
this->m_visited = false;\
if (false == m_cache.empty()) {\
    m_cache.clear();\
}\
MetaVariable::resetTemp();\
mp_left->reset();\
mp_right->reset();

// Binary find me 
#define BINARY_FIND_ME()\
if(static_cast<void*>(mp_left) == v) {\
    return true;\
} else if (static_cast<void*>(mp_right) == v) {\
    return true;\
} else {\
    if(mp_left->findMe(v) == true) {\
        return true;\
    } else if(mp_right->findMe(v) == true) {\
        return true;\
    }\
}\
return false;

// Binary right reset 
#define BINARY_RIGHT_RESET()\
this->m_visited = false;\
if (false == m_cache.empty()) {\
    m_cache.clear();\
}\
MetaVariable::resetTemp();\
mp_right->reset();

// Binary right find me
#define BINARY_RIGHT_FIND_ME()\
if (static_cast<void*>(mp_right) == v) {\
    return true;\
} else {\
    if(mp_right->findMe(v) == true) {\
        return true;\
    }\
}\
return false; 

// Binary left reset 
#define BINARY_LEFT_RESET()\
this->m_visited = false;\
if (false == m_cache.empty()) {\
    m_cache.clear();\
}\
MetaVariable::resetTemp();\
mp_left->reset();

// Binary left find me 
#define BINARY_LEFT_FIND_ME()\
if (static_cast<void*>(mp_left) == v) {\
    return true;\
} else {\
    if(mp_left->findMe(v) == true) {\
        return true;\
    }\
}\
return false;

// Map reserve size
constexpr const size_t g_map_reserve{ 32 };
// Constant size for vector of generic holder
constexpr const size_t g_vec_init{ 32 };

// Predeclare a few classes
class VarWrap;
class Variable;
class Parameter;
class Expression;

// Typedef double as Type (TODO: Replace Type with a class/struct based on variants to support multiple types)
using Real = SCALAR_TYPE;
#if defined(USE_COMPLEX_MATH)
    #include <complex>
    using Type = std::complex<Real>;

    Type operator+(Real, const Type&);
    Type operator+(const Type&, Real);

    Type operator-(Real, const Type&);
    Type operator-(const Type&, Real);

    Type operator*(Real, const Type&);
    Type operator*(const Type&, Real);

    Type operator/(Real, const Type&);
    Type operator/(const Type&, Real);

    bool operator!=(const Type&, Real);
    bool operator!=(Real, const Type&);

    bool operator==(const Type&, Real);
    bool operator==(Real, const Type&);
#else
    using Type = Real;
#endif

// Ordered map between size_t and Type 
#if defined(USE_ROBIN_HOOD_MAP)
    #include <robin_hood.h>
    using OMPair = robin_hood::unordered_flat_map<size_t, Type>;
    // A generic unorderedmap
    template<typename T, typename U>
    using UnOrderedMap = robin_hood::unordered_flat_map<T, U>;
#else
    #include <unordered_map>
    using OMPair = std::unordered_map<size_t, Type>;
    // A generic unorderedmap
    template<typename T, typename U>
    using UnOrderedMap = std::unordered_map<T, U>;
#endif

// A generic vector type
template<typename T>
using Vector = std::vector<T>;

// A generic variadic tuple type
template<typename... Args>
using Tuples = std::tuple<Args...>;

// A generic shared pointer
template<typename T>
using SharedPtr = std::shared_ptr<T>;

// Delete resource
template<typename T>
void DelPtr(T* ptr) {
    if (ptr != nullptr) {
        delete ptr;
        ptr = nullptr;
    }
}

#if defined(USE_CUSTOM_FUNCTIONS)
    // Operations enum (Order matters!)
    enum Op : size_t {
        ADD = 0, SUB, MUL, DIV,
        SIN, COS, TAN, SINH, COSH, TANH,
        ASIN, ACOS, ATAN, ASINH, ACOSH, ATANH,
        ABS, SQRT, EXP, LOG, POW,
        RELU,
        COUNT
    };
    
    // Operation type (Order matters!)
    #define OpType std::plus<Type>, std::minus<Type>, std::multiplies<Type>, std::divides<Type>,\
                Sin<Type>, Cos<Type>, Tan<Type>, Sinh<Type>, Cosh<Type>, Tanh<Type>,\
                ASin<Type>, ACos<Type>, ATan<Type>, ASinh<Type>, ACosh<Type>, ATanh<Type>

    // Operation objects (Order matters!)
    #define OpObj  std::plus<Type>(), std::minus<Type>(), std::multiplies<Type>(), std::divides<Type>(),\
                Sin<Type>(), Cos<Type>(), Tan<Type>(), Sinh<Type>(), Cosh<Type>(), Tanh<Type>(),\
                ASin<Type>(), ACos<Type>(), ATan<Type>(), ASinh<Type>(), ACosh<Type>(), ATanh<Type>()
#else
    struct X007{};
    #define OpType X007
    #define OpObj OpType()
#endif

#if defined(USE_VIRTUAL_FUNCTIONS)
    #define V_OVERRIDE(X) X override
    #define V_DTR(X) virtual X
    #define V_PURE(X) virtual X = 0
#endif

// Convert to string
template<typename T>
std::string ToString(const T& value) {
    // if complex number
    if constexpr(std::is_same_v<T, std::complex<Real>>) {
        return std::move("(" + std::to_string(value.real()) + "," + std::to_string(value.imag()) + ")");
    }
    else {
        return std::move(std::to_string(value));
    }
}
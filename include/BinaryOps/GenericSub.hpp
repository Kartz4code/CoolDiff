#pragma once 

#include "IVariable.hpp"
#include "Variable.hpp"

// Left/right side is an expression
template <typename T1, typename T2, typename... Callables>
class GenericSub : public IVariable<GenericSub<T1, T2, Callables...>> {
private:
    // Resources
    T1* mp_left{ nullptr };
    T2* mp_right{ nullptr };

    // Callables
    Tuples<Callables...> m_caller;

    // Disable copy and move constructors/assignments
    DISABLE_COPY(GenericSub)
    DISABLE_MOVE(GenericSub)

public:
    // Block index
    const size_t m_nidx{};
    // Cache for reverse AD
    OMPair m_cache;

    
    // Constructor
    GenericSub(T1* u, T2* v, Callables&&... call) : mp_left{ u },
                                                    mp_right{ v },
                                                    m_caller{ std::make_tuple(std::forward<Callables>(call)...) },
                                                    m_nidx{ this->m_idx_count++ }
    {}


    /*
    * ======================================================================================================
    * ======================================================================================================
    * ======================================================================================================
     _   _ ___________ _____ _   _  ___   _       _____  _   _ ___________ _     _____  ___ ______  _____
    | | | |_   _| ___ \_   _| | | |/ _ \ | |     |  _  || | | |  ___| ___ \ |   |  _  |/ _ \|  _  \/  ___|
    | | | | | | | |_/ / | | | | | / /_\ \| |     | | | || | | | |__ | |_/ / |   | | | / /_\ \ | | |\ `--.
    | | | | | | |    /  | | | | | |  _  || |     | | | || | | |  __||    /| |   | | | |  _  | | | | `--. \
    \ \_/ /_| |_| |\ \  | | | |_| | | | || |____ \ \_/ /\ \_/ / |___| |\ \| |___\ \_/ / | | | |/ / /\__/ /
     \___/ \___/\_| \_| \_/  \___/\_| |_/\_____/  \___/  \___/\____/\_| \_\_____/\___/\_| |_/___/  \____/

    *======================================================================================================
    *======================================================================================================
    *======================================================================================================
    */

    // Symbolic evaluation
    V_OVERRIDE( Variable* symEval() ) {
        if (nullptr == this->mp_tmp) {  
            auto tmp = Allocate<Expression>((EVAL_L()) - (EVAL_R()));
            this->mp_tmp = tmp.get();
        }
        return this->mp_tmp;
    }

    // Symbolic differentiation
    V_OVERRIDE( Variable* symDeval(const Variable& var) ) {
        // Static derivative computation
        if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end()) {
            auto tmp = Allocate<Expression>((DEVAL_L(var)) - (DEVAL_R(var)));
            this->mp_dtmp[var.m_nidx] = tmp.get();
        }
        return this->mp_dtmp[var.m_nidx];
    }

    // Eval in run-time
    V_OVERRIDE( Type eval() ) {
        // Returned evaluation
        const Type u = mp_left->eval();
        const Type v = mp_right->eval();
        return (u - v);
    }

    // Deval 1st in run-time for forward derivative
    V_OVERRIDE( Type devalF(const Variable& var) ) {
        // Return derivative of -: ud - vd
        const Type du = mp_left->devalF(var);
        const Type dv = mp_right->devalF(var);
        return (du - dv);
    }

    // Traverse run-time
    V_OVERRIDE( void traverse(OMPair* cache = nullptr) ) {
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

            /* IMPORTANT: The derivative is computed here */
            (*cache)[mp_left->m_nidx] += (Type)1;
            (*cache)[mp_right->m_nidx] += (Type)(-1);

            // Modify cache for left node
            for (const auto& [idx, val] : mp_left->m_cache) {
                (*cache)[idx] += val;
            }
            // Modify cache for right node
            for (const auto& [idx, val] : mp_right->m_cache) {
                (*cache)[idx] += ((Type)(-1) * val);
            }
        }
        else {
            // Cached value
            const Type cCache = (*cache)[m_nidx];

            // Traverse left node
            if (false == mp_left->m_visited) {
                mp_left->traverse(cache);
            }
            // Traverse right node
            if (false == mp_right->m_visited) {
                mp_right->traverse(cache);
            }

            /* IMPORTANT: The derivative is computed here */
            (*cache)[mp_left->m_nidx] += cCache;
            (*cache)[mp_right->m_nidx] += ((Type)(-1) * cCache);

            // Modify cache for left node
            if (cCache != 0) {
                for (const auto& [idx, val] : mp_left->m_cache) {
                    (*cache)[idx] += (val * cCache);
                }
                // Modify cache for right node
                for (const auto& [idx, val] : mp_right->m_cache) {
                    (*cache)[idx] += ((Type)(-1) * val * cCache);
                }
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

    // Get m_cache
    V_OVERRIDE( OMPair& getCache() ) {
        return m_cache;
    }

    // Reset visit run-time
    V_OVERRIDE( void reset() ) {
        BINARY_RESET();
    }

    // Get type
    V_OVERRIDE( std::string_view getType() const ) {
        return "GenericSub";
    }

    // Find me 
    V_OVERRIDE( bool findMe(void* v) const )  {
        BINARY_FIND_ME();
    }

    // Destructor
    V_DTR( ~GenericSub() ) = default;
};

// Left side is a number
template <typename T, typename... Callables>
class GenericSub<Type, T, Callables...> : public IVariable<GenericSub<Type, T, Callables...>> {
private:
    // Resources
    Type mp_left{0};
    T* mp_right{ nullptr };
    
    // Callables
    Tuples<Callables...> m_caller;

    // Disable copy and move constructors/assignments
    DISABLE_COPY(GenericSub)
    DISABLE_MOVE(GenericSub)


public:
    // Block index
    const size_t m_nidx{};
    // Cache for reverse AD
    OMPair m_cache;

    // Constructor
    GenericSub(const Type& u, T* v, Callables&&... call) : mp_left{ u },
                                                           mp_right{ v },
                                                           m_caller{ std::make_tuple(std::forward<Callables>(call)...) },
                                                           m_nidx{ this->m_idx_count++ }
    {}


    /*
    * ======================================================================================================
    * ======================================================================================================
    * ======================================================================================================
     _   _ ___________ _____ _   _  ___   _       _____  _   _ ___________ _     _____  ___ ______  _____
    | | | |_   _| ___ \_   _| | | |/ _ \ | |     |  _  || | | |  ___| ___ \ |   |  _  |/ _ \|  _  \/  ___|
    | | | | | | | |_/ / | | | | | / /_\ \| |     | | | || | | | |__ | |_/ / |   | | | / /_\ \ | | |\ `--.
    | | | | | | |    /  | | | | | |  _  || |     | | | || | | |  __||    /| |   | | | |  _  | | | | `--. \
    \ \_/ /_| |_| |\ \  | | | |_| | | | || |____ \ \_/ /\ \_/ / |___| |\ \| |___\ \_/ / | | | |/ / /\__/ /
     \___/ \___/\_| \_| \_/  \___/\_| |_/\_____/  \___/  \___/\____/\_| \_\_____/\___/\_| |_/___/  \____/

    *======================================================================================================
    *======================================================================================================
    *======================================================================================================
    */

    // Symbolic evaluation
    V_OVERRIDE( Variable* symEval() ) {
        if (nullptr == this->mp_tmp) {
            auto tmp = Allocate<Expression>((mp_left) - (EVAL_R()));
            this->mp_tmp = tmp.get();
        }
        return this->mp_tmp;
    }

    // Symbolic differentiation
    V_OVERRIDE( Variable* symDeval(const Variable& var) ) {
        // Static derivative computation
        if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end()) {
            auto tmp = Allocate<Expression>((Type)(-1) * (DEVAL_R(var)));
            this->mp_dtmp[var.m_nidx] = tmp.get();
        }
        return this->mp_dtmp[var.m_nidx];
    }

    // Eval in run-time
    V_OVERRIDE( Type eval() ) {
        // Returned evaluation
        const Type v = mp_right->eval();
        return (mp_left - v);
    }

    // Deval 1st in run-time for forward derivative
    V_OVERRIDE( Type devalF(const Variable& var) ) {
        // Return derivative of -: -vd
        const Type dv = ((Type)(-1) * mp_right->devalF(var));
        return dv;
    }

    // Traverse run-time
    V_OVERRIDE( void traverse(OMPair* cache = nullptr) ) {
        // If cache is nullptr, i.e. for the first step
        if (cache == nullptr) {
            // cache is m_cache
            cache = &m_cache;
            cache->reserve(g_map_reserve);
            // Clear cache in the first entry
            if (false == (*cache).empty()) {
                (*cache).clear();
            }

            // Traverse right node
            if (false == mp_right->m_visited) {
                mp_right->traverse(cache);
            }

            /* IMPORTANT: The derivative is computed here */
            (*cache)[mp_right->m_nidx] += (Type)(-1);
            
            // Modify cache for right node
            for (const auto& [idx, val] : mp_right->m_cache) {
                (*cache)[idx] += ((Type)(-1) * val);
            }
        }
        else {
            // Cached value
            const Type cCache = (*cache)[m_nidx];

            // Traverse right node
            if (false == mp_right->m_visited) {
                mp_right->traverse(cache);
            }

            /* IMPORTANT: The derivative is computed here */
            (*cache)[mp_right->m_nidx] += ((Type)(-1) * cCache);

            // Modify cache for right node
            if (cCache != 0) {
                for (const auto& [idx, val] : mp_right->m_cache) {
                    (*cache)[idx] += ((Type)(-1) * val * cCache);
                }
            }
        }
        // Traverse left/right nodes
        if (false == mp_right->m_visited) {
            mp_right->traverse(cache);
        }
    }

    // Get m_cache
    V_OVERRIDE( OMPair& getCache() ) {
        return m_cache;
    }

    // Reset visit run-time
    V_OVERRIDE( void reset() ) {
        BINARY_RIGHT_RESET();
    }

    // Get type
    V_OVERRIDE( std::string_view getType() const ) {
        return "GenericSub";
    }

    // Find me 
    V_OVERRIDE( bool findMe(void* v) const ) {
        BINARY_RIGHT_FIND_ME();
    }

    // Destructor
    V_DTR( ~GenericSub() ) = default;
};

// Right side is a number
template <typename T, typename... Callables>
class GenericSub<T, Type, Callables...> : public IVariable<GenericSub<T, Type, Callables...>> {
private:
    // Resources
    T* mp_left{ nullptr };
    Type mp_right{0};

    // Callables
    Tuples<Callables...> m_caller;

    // Disable copy and move constructors/assignments
    DISABLE_COPY(GenericSub)
    DISABLE_MOVE(GenericSub)

public:
    // Block index
    const size_t m_nidx{};
    // Cache for reverse AD
    OMPair m_cache;

    // Constructor
    GenericSub(T* u, const Type& v, Callables&&... call) : mp_left{ u },
                                                           mp_right{ v },
                                                           m_caller{ std::make_tuple(std::forward<Callables>(call)...) },
                                                           m_nidx{ this->m_idx_count++ }
    {}

    /*
    * ======================================================================================================
    * ======================================================================================================
    * ======================================================================================================
     _   _ ___________ _____ _   _  ___   _       _____  _   _ ___________ _     _____  ___ ______  _____
    | | | |_   _| ___ \_   _| | | |/ _ \ | |     |  _  || | | |  ___| ___ \ |   |  _  |/ _ \|  _  \/  ___|
    | | | | | | | |_/ / | | | | | / /_\ \| |     | | | || | | | |__ | |_/ / |   | | | / /_\ \ | | |\ `--.
    | | | | | | |    /  | | | | | |  _  || |     | | | || | | |  __||    /| |   | | | |  _  | | | | `--. \
    \ \_/ /_| |_| |\ \  | | | |_| | | | || |____ \ \_/ /\ \_/ / |___| |\ \| |___\ \_/ / | | | |/ / /\__/ /
     \___/ \___/\_| \_| \_/  \___/\_| |_/\_____/  \___/  \___/\____/\_| \_\_____/\___/\_| |_/___/  \____/

    *======================================================================================================
    *======================================================================================================
    *======================================================================================================
    */

    // Symbolic evaluation
    V_OVERRIDE( Variable* symEval() ) {
        if (nullptr == this->mp_tmp) {
            auto tmp = Allocate<Expression>((EVAL_L()) - (mp_right));
            this->mp_tmp = tmp.get();
        }
        return this->mp_tmp;
    }

    // Symbolic differentiation
    V_OVERRIDE( Variable* symDeval(const Variable& var) ) {
        // Static derivative computation
        if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end()) {
            auto tmp = Allocate<Expression>((DEVAL_L(var)));
            this->mp_dtmp[var.m_nidx] = tmp.get();
        }
        return this->mp_dtmp[var.m_nidx];
    }

    // Eval in run-time
    V_OVERRIDE( Type eval() ) {
        // Returned evaluation
        const Type u = mp_left->eval();
        return (u - mp_right);
    }

    // Deval 1st in run-time for forward derivative
    V_OVERRIDE( Type devalF(const Variable& var) ) {
        // Return derivative of -: udd
        const Type du = mp_left->devalF(var);
        return (du);
    }

    // Traverse run-time
    V_OVERRIDE( void traverse(OMPair* cache = nullptr) ) {
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

            /* IMPORTANT: The derivative is computed here */
            (*cache)[mp_left->m_nidx] += (Type)1;

            // Modify cache for left node
            for (const auto& [idx, val] : mp_left->m_cache) {
                (*cache)[idx] += val;
            }
        }
        else {
            // Cached value
            const Type cCache = (*cache)[m_nidx];

            // Traverse left node
            if (false == mp_left->m_visited) {
                mp_left->traverse(cache);
            }

            /* IMPORTANT: The derivative is computed here */
            (*cache)[mp_left->m_nidx] += cCache;
            
            // Modify cache for left node
            if (cCache != 0) {
                for (const auto& [idx, val] : mp_left->m_cache) {
                    (*cache)[idx] += (val * cCache);
                }
            }
        }
        // Traverse left/right nodes
        if (false == mp_left->m_visited) {
            mp_left->traverse(cache);
        }
    }

    // Get m_cache
    V_OVERRIDE( OMPair& getCache() ) {
        return m_cache;
    }

    // Reset visit run-time
    V_OVERRIDE( void reset() ) {
        BINARY_LEFT_RESET();
    }

    // Get type
    V_OVERRIDE( std::string_view getType() const ) {
        return "GenericSub";
    }

    // Find me 
    V_OVERRIDE( bool findMe(void* v) const ) {
        BINARY_LEFT_FIND_ME();
    }

    // Destructor
    V_DTR( ~GenericSub() ) = default;
};

// GenericSub with 2 typename callables 
template<typename T1, typename T2>
using GenericSubT1 = GenericSub<T1, T2, OpType>;

// GenericSub with 1 typename callables 
template<typename T>
using GenericSubT2 = GenericSub<Type, T, OpType>;

template<typename T>
using GenericSubT3 = GenericSub<T, Type, OpType>;

// Function for subtraction computation
template <typename T1, typename T2>
const GenericSubT1<T1, T2>& operator-(const IVariable<T1>& u, const IVariable<T2>& v) {
    auto tmp = Allocate<GenericSubT1<T1, T2>>(const_cast<T1*>(static_cast<const T1*>(&u)), 
                                              const_cast<T2*>(static_cast<const T2*>(&v)), 
                                              OpObj);
    return *tmp;
}

// Left side is a number (subtraction)
template <typename T>
const GenericSubT2<T>& operator-(const Type& u, const IVariable<T>& v) {
    auto tmp = Allocate<GenericSubT2<T>>(u, const_cast<T*>(static_cast<const T*>(&v)), OpObj);
    return *tmp;
}

// Right side is a number (subtraction)
template <typename T>
const GenericSubT3<T>& operator-(const IVariable<T>& u, const Type& v) {
    auto tmp = Allocate<GenericSubT3<T>>(const_cast<T*>(static_cast<const T*>(&u)), v, OpObj);
    return *tmp;
}

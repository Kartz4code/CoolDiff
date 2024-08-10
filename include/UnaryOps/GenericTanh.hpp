 #pragma once 

#include "IVariable.hpp"
#include "Variable.hpp"

template <typename T, typename... Callables>
class GenericTanh : public IVariable<GenericTanh<T, Callables...>> {
private:
    // Resources
    T* mp_left{ nullptr };

    // Callables
    Tuples<Callables...> m_caller;

    // Disable copy and move constructors/assignments
    DISABLE_COPY(GenericTanh)
    DISABLE_MOVE(GenericTanh)


public:
    // Block index
    const size_t m_nidx{};
    // Cache for reverse AD
    OMPair m_cache;

    // Constructor
    GenericTanh(T* u, Callables&&... call) : mp_left{ u },
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
            auto tmp = Allocate<Expression>(tanh(EVAL_L()));
            this->mp_tmp = tmp.get();
        }
        return this->mp_tmp;
    }

    // Symbolic Differentiation
    V_OVERRIDE( Variable* symDeval(const Variable& var) ) {
        // Static derivative computation
        if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end()) {
            auto tmp = Allocate<Expression>(((Type)(1) - (tanh(EVAL_L())*tanh(EVAL_L())))*(DEVAL_L(var)));
            this->mp_dtmp[var.m_nidx] = tmp.get();
        }
        return this->mp_dtmp[var.m_nidx];
    }

    // Eval in run-time
    V_OVERRIDE( Type eval() ) {
        // Returned evaluation
        const Type u = mp_left->eval();
        return (std::tanh(u));
    }

    // Deval in run-time for forward derivative
    V_OVERRIDE( Type devalF(const Variable& var) ) {
        // Return derivative of tanh: (1-tanh(u)*tanh(u))*ud
        const Type du = mp_left->devalF(var);
        const Type u = mp_left->eval();
        return (((Type)(1) - (std::tanh(u) * std::tanh(u)))*du);
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
           const Type uRT = mp_left->eval(); 
           const Type u = ((Type)(1) - (std::tanh(uRT) * std::tanh(uRT)));
           (*cache)[mp_left->m_nidx] += u;

           // Modify cache for left node
           if (u != 0) {
               for (const auto& [idx, val] : mp_left->m_cache) {
                   (*cache)[idx] += (val * u);
               }
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
           const Type uRT = mp_left->eval(); 
           const Type u = ((Type)(1) - (std::tanh(uRT) * std::tanh(uRT)));
           const Type ustar = (u * cCache);
           (*cache)[mp_left->m_nidx] += (ustar);

           // Modify cache for left node
           if (ustar != 0) {
               for (const auto& [idx, val] : mp_left->m_cache) {
                   (*cache)[idx] += (val * ustar);
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
        UNARY_RESET(); 
    }

    // Get type
    V_OVERRIDE( std::string_view getType() const ) {
        return "GenericTanh";
    }

    // Find me 
    V_OVERRIDE( bool findMe(void* v) const )  {
        UNARY_FIND_ME();
    }

    // Destructor
    V_DTR( ~GenericTanh() ) = default;
};

// Generic tanh with 1 typename callables 
template<typename T>
using GenericTanhT = GenericTanh<T, OpType>;

// Function for tanh computation
template <typename T>
const GenericTanhT<T>& tanh(const IVariable<T>& u) {
    auto tmp = Allocate<GenericTanhT<T>>(const_cast<T*>(static_cast<const T*>(&u)), OpObj);
    return *tmp;
}
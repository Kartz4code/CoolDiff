
#include "MetaVariable.hpp"
#include "Variable.hpp"

// Reset temporaries
void MetaVariable::resetTemp() {
    if(this->mp_tmp != nullptr) {
        this->mp_tmp->reset();
    }
    for(auto& [k,v] : this->mp_dtmp) {
        if(v != nullptr) {   
            v->reset();
        }
    }
    this->m_visited = false;
}
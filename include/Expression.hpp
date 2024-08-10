#pragma once 

#include "Variable.hpp"
#include "Parameter.hpp"

class Expression : public Variable {
    public:
        Expression();

        Expression(const Expression&);

        template <typename T>
        Expression(const IVariable<T>& expr) {
            Variable::m_nidx = this->m_idx_count++;
            // Reserve a buffer of expressions
            Variable::m_gh_vec.reserve(g_vec_init);
            // Emplace the expression in a generic holder
            Variable::m_gh_vec.emplace_back(&static_cast<const T&>(expr));
        }

        /* Copy assignment for expression evaluation - e.g.Variable x = x1 + x2 + x3; */
        template <typename T>
        Expression& operator=(const IVariable<T>& expr) {
            if(static_cast<const T&>(expr).findMe(this) == false) {
                m_gh_vec.clear();
            }
            // Emplace the expression in a generic holder
            Variable::m_gh_vec.emplace_back(&static_cast<const T&>(expr));
            return *this;
        }

        // Symbolic differentiation of expression
        Expression& SymDiff(const Variable&);
    
        // Get type
        V_OVERRIDE( std::string_view getType() const );

        V_DTR( ~Expression() );
};

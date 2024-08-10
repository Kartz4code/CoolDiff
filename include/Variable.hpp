#pragma once

#include "IVariable.hpp"
#include "VarWrap.hpp"

class Expression;

// Variable Expression is a wrapper around the variable class
class Variable : public IVariable<Variable> {
private:
    friend class Expression;
    /* Constructor for expression evaluation - e.g.Variable x = x1 + x2 + x3;
       Dummy variable is created and it is counted negatively
    */
    template <typename T>
    Variable(const IVariable<T>& expr) : m_nidx{ this->m_idx_count++ } {
        // Reserve a buffer of expressions
        m_gh_vec.reserve(g_vec_init);
        // Emplace the expression in a generic holder
        m_gh_vec.emplace_back(&static_cast<const T&>(expr));
    }

    /* Copy assignment for expression evaluation - e.g.Variable x = x1 + x2 + x3; */
    template <typename T>
    Variable& operator=(const IVariable<T>& expr) {
        // Emplace the expression in a generic holder
        m_gh_vec.emplace_back(&static_cast<const T&>(expr));
        return *this;
    }

    // Getters and setters
    // Get/Set value
    void setValue(Type);
    Type getValue() const;

    // Get/Set dvalue
    void setdValue(Type);
    Type getdValue() const;

    // Set string sets the expression (which can also be a variable name)
    void setExpression(const std::string&);
    const std::string& getExpression() const;

protected:
    // Underlying symbolic variable
    VarWrap m_var{};
    // Type-time value
    Type m_value_var{};

    // Collection of meta variable expressions
    Vector<MetaVariable*> m_gh_vec{};

    // Exposed to user to compute symbolic differentiation
    Expression SymDiff(const Variable&);


public:
    // Static variable for one seed
    static Variable t1;
    // Static variable for zero seed
    static Variable t0;

    // Block index
    size_t m_nidx{};
    // Cache for reverse 1st AD
    OMPair m_cache{};

    // Constructors
    Variable();
    // Copy constructor
    Variable(const Variable&);
    // Move constructor
    Variable(Variable&&) noexcept;
    // Copy assignment from one variable to another
    Variable& operator=(const Variable&);
    // Move assignment from one variable to another
    Variable& operator=(Variable&&) noexcept;

    // Constructors for Type values
    Variable(const Type&);
    // Assignment to Type
    Variable& operator=(const Type&);

    // To output stream (TODO)
    friend std::ostream& operator<<(std::ostream&, Variable&);

    // Reset impl
    void resetImpl();

    // Deval in run-time for reverse derivative
    Type devalR(const Variable&);

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

    // Evaluate value and derivative value in run-time
    V_OVERRIDE( Type eval() );
    // Evaluate 1st derivative in forward mode
    V_OVERRIDE( Type devalF(const Variable&) );  

    // Traverse tree 
    V_OVERRIDE( void traverse(OMPair* = nullptr) );
    // Get the map of derivatives
    V_OVERRIDE( OMPair& getCache() );

    // Evaluate variable and its derivative value in run-time
    V_OVERRIDE( Variable* symEval() );
    V_OVERRIDE( Variable* symDeval(const Variable&) );

    // Reset all visited flags
    V_OVERRIDE( void reset() );

    // Get type
    V_OVERRIDE( std::string_view getType() const );
 
    // Find me 
    V_OVERRIDE( bool findMe(void*) const );

    // Destructor
    V_DTR( ~Variable() );
};


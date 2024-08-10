#pragma once 

#include "Variable.hpp"
#include "Expression.hpp"
#include "Parameter.hpp"

// Binary operators
#include "GenericSum.hpp"
#include "GenericProduct.hpp"
#include "GenericSub.hpp"
#include "GenericDiv.hpp"
#include "GenericPow.hpp"

// Unary operator
#include "GenericSin.hpp"
#include "GenericCos.hpp"
#include "GenericTan.hpp"
#include "GenericSqrt.hpp"
#include "GenericExp.hpp"
#include "GenericLog.hpp"
#include "GenericASin.hpp"
#include "GenericACos.hpp"
#include "GenericATan.hpp"
#include "GenericSinh.hpp"
#include "GenericCosh.hpp"
#include "GenericTanh.hpp"
#include "GenericASinh.hpp"
#include "GenericACosh.hpp"
#include "GenericATanh.hpp"

// Main evaluate function
Type Eval(Expression&);
// Main forward mode AD computation
Type DevalF(Expression&, const Variable&);

// Main precomputation of reverse mode AD computation
void PreComp(Expression&);
// Main reverse mode AD computation
Type DevalR(Expression&, const Variable&);

// Main reverse mode AD table
OMPair& PreCompCache(const Expression&);

// Symbolic Expression 
Expression& SymDiff(Expression&, const Variable&);

// Create new expression
template<typename T, typename = std::enable_if_t<std::is_base_of_v<MetaVariable,T>>>
Expression& CreateExpr(const T& exp) {
    auto tmp = Allocate<Expression>(exp);
    return *tmp;
}

// Free function for creating new expression
Expression& CreateExpr(const Type& = 0);

// Create new variable
Variable& CreateVar(const Type&);

// Create new parameter
Parameter& CreateParam(const Type&);
#include "CommonFunctions.hpp"

// Evaluate function
Type Eval(Expression& exp) {
    // Reset graph/tree
    exp.resetImpl();
    // Return evaluation value
    return exp.eval();
}
// Forward mode algorithmic differentiation
Type DevalF(Expression& exp, const Variable& x) {
    // Reset visited flags in the tree
    exp.resetImpl();
    // Return forward differentiation value
    return exp.devalF(x);
}

// Main precomputation of reverse mode AD computation 1st
void PreComp(Expression& exp) {
    // Reset visited flags in the tree
    exp.resetImpl();
    // Traverse tree
    exp.traverse();
}

// Main reverse mode AD computation 1st
Type DevalR(Expression& exp, const Variable& x) {
    // Return reverse differentiation value
    return exp.devalR(x);
}

// Main reverse mode AD table 1st
OMPair& PreCompCache(Expression& exp) {
    // Reset flags
    exp.resetImpl();
    // Traverse evaluation graph/tree
    exp.traverse();
    //Return cache
    return exp.getCache();
}

// Symbolic Expression 
Expression& SymDiff(Expression& exp, const Variable& var) {
    // Reset graph/tree
    return exp.SymDiff(var);
}

// Free function for creating new expression
Expression& CreateExpr(const Type& val) {
    // Allocate a temporary parameter
    auto param = Allocate<Parameter>(val);
    // Create expression
    auto tmp = Allocate<Expression>(*param);
    return *tmp;
}

// Create new variable
Variable& CreateVar(const Type& val) {
    auto tmp = Allocate<Variable>(val);
    return *tmp;
}

// Create new parameter
Parameter& CreateParam(const Type& val) {
    auto tmp = Allocate<Parameter>(val);
    return *tmp;
}


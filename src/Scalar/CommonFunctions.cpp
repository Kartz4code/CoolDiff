/**
 * @file src/Scalar/CommonFunctions.cpp
 *
 * @copyright 2023-2024 Karthik Murali Madhavan Rathai
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

#include "CommonFunctions.hpp"
#include "Matrix.hpp"

// Evaluate function
Type Eval(Expression &exp)
{
    // Reset graph/tree
    exp.resetImpl();
    // Return evaluation value
    return exp.eval();
}
// Forward mode algorithmic differentiation
Type DevalF(Expression &exp, const Variable &x)
{
    // Reset visited flags in the tree
    exp.resetImpl();
    // Return forward differentiation value
    return exp.devalF(x);
}

// Main precomputation of reverse mode AD computation 1st
void PreComp(Expression &exp)
{
    // Reset visited flags in the tree
    exp.resetImpl();
    // Traverse tree
    exp.traverse();
}

// Main reverse mode AD computation 1st
Type DevalR(Expression &exp, const Variable &x)
{
    // Return reverse differentiation value
    return exp.devalR(x);
}

// Main reverse mode AD table 1st
OMPair &PreCompCache(Expression &exp)
{
    // Reset flags
    exp.resetImpl();
    // Traverse evaluation graph/tree
    exp.traverse();
    //Return cache
    return exp.getCache();
}

// Derivative of expression
Type Deval(Expression &exp, const Variable &x, ADMode ad)
{
    if (ad == ADMode::FORWARD)
    {
        return DevalF(exp, x);
    }
    else
    {
        PreComp(exp);
        return DevalR(exp, x);
    }
}

// Jacobian forward mode
Matrix<Type> &JacobF(Expression &exp, const Vector<Variable> &vec)
{
    const size_t rows = vec.size();
    auto &result = CreateMatrix<Type>(rows, 1);
    for (size_t i{}; i < rows; ++i)
    {
        result(i, 1) = DevalF(exp, vec[i]);
    }
    return result;
}

// Jacobian reverse mode
Matrix<Type> &JacobR(Expression &exp, const Vector<Variable> &vec)
{
    const size_t rows = vec.size();
    auto &result = CreateMatrix<Type>(rows, 1);

    // Precompute
    PreComp(exp);
    for (size_t i{}; i < rows; ++i)
    {
        result(i, 1) = DevalR(exp, vec[i]);
    }
    return result;
}

// Jacobian of expression
Matrix<Type> &Jacobian(Expression &exp, const Vector<Variable> &vec, ADMode ad)
{
    if (ad == ADMode::FORWARD)
    {
        return JacobF(exp, vec);
    }
    else
    {
        return JacobR(exp, vec);
    }
}

// Symbolic Jacobian of expression
Matrix<Expression> &JacobianSym(Expression &exp, const Vector<Variable> &vec)
{
    const size_t rows = vec.size();
    auto &result = CreateMatrix<Expression>(rows, 1);
    for (size_t i{}; i < rows; ++i)
    {
        result(i, 1) = SymDiff(exp, vec[i]);
    }
    return result;
}

// Hessian forward mode
Matrix<Type> &HessF(Expression &exp, const Vector<Variable> &vec)
{
    const size_t dim = vec.size();
    auto &result = CreateMatrix<Type>(dim, dim);
    Matrix<Expression> firstSym(dim, 1);

    // Exploit Hessian symmetry
    for (size_t i{}; i < dim; ++i)
    {
        firstSym[i] = SymDiff(exp, vec[i]);
        for (size_t j{}; j < dim; ++j)
        {
            if (i < j)
            {
                result(i, j) = DevalF(firstSym[i], vec[j]);
                result(j, i) = result(i, j);
            }
            else if (i == j)
            {
                result(i, j) = DevalF(firstSym[i], vec[j]);
            }
        }
    }

    return result;
}

// Hessian reverse mode
Matrix<Type> &HessR(Expression &exp, const Vector<Variable> &vec)
{
    const size_t dim = vec.size();
    auto &result = CreateMatrix<Type>(dim, dim);
    Matrix<Expression> firstSym(dim, 1);

    // Exploit Hessian symmetry
    for (size_t i{}; i < dim; ++i)
    {
        firstSym[i] = SymDiff(exp, vec[i]);
        // Precompute
        PreComp(firstSym[i]);
        for (size_t j{}; j < dim; ++j)
        {
            if (i < j)
            {
                result(i, j) = DevalR(firstSym[i], vec[j]);
                result(j, i) = result(i, j);
            }
            else if (i == j)
            {
                result(i, j) = DevalR(firstSym[i], vec[j]);
            }
        }
    }

    return result;
}

// Hessian of expression
Matrix<Type> &Hessian(Expression &exp, const Vector<Variable> &vec, ADMode ad)
{
    if (ad == ADMode::FORWARD)
    {
        return HessF(exp, vec);
    }
    else
    {
        return HessR(exp, vec);
    }
}

// Symbolic Hessian of expression
Matrix<Expression> &HessianSym(Expression &exp, const Vector<Variable> &vec)
{
    const size_t dim = vec.size();
    auto &result = CreateMatrix<Expression>(dim, dim);

    // Exploit Hessian symmetry
    for (size_t i{}; i < dim; ++i)
    {
        for (size_t j{}; j < dim; ++j)
        {
            if (i < j)
            {
                result(i, j) = SymDiff(SymDiff(exp, vec[i]), vec[j]);
                result(j, i) = result(i, j);
            }
            else if (i == j)
            {
                result(i, j) = SymDiff(SymDiff(exp, vec[i]), vec[j]);
            }
        }
    }

    return result;
}


// Symbolic Expression
Expression &SymDiff(Expression &exp, const Variable &var)
{
    // Reset graph/tree
    return exp.SymDiff(var);
}

// Free function for creating new expression
Expression &CreateExpr(const Type &val)
{
    // Allocate a temporary parameter
    auto param = Allocate<Parameter>(val);
    // Create expression
    auto tmp = Allocate<Expression>(*param);
    return *tmp;
}

// Create new variable
Variable &CreateVar(const Type &val)
{
    auto tmp = Allocate<Variable>(val);
    return *tmp;
}

// Create new parameter
Parameter &CreateParam(const Type &val)
{
    auto tmp = Allocate<Parameter>(val);
    return *tmp;
}

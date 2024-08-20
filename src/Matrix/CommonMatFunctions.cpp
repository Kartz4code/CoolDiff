#include "CommonMatFunctions.hpp"

// Is the matrix zero
bool IsZeroMatrix(const Matrix<Type>& m) {
    auto* it = m.getMatrixPtr();
    const size_t n = m.getNumElem();
    return std::all_of(it, it+n, [](Type i) { return (i == (Type)(0)); } );
}

// Is the matrix identity
bool IsEyeMatrix(const Matrix<Type>& m) {
    // If matrix is rectangular, return false
    if(false == IsSquareMatrix(m)) {
        return false;
    }
    const size_t rows = m.getNumRows();
    for(size_t i{}; i < rows; ++i) {
        if(m(i,i) != (Type)(1)) {
            return false;
        }
    }
    return true;
}

// Is the matrix ones
bool IsOnesMatrix(const Matrix<Type>& m) {
    auto* it = m.getMatrixPtr();
    const size_t n = m.getNumElem();
    return std::all_of(it, it+n, [](Type i) { return (i == (Type)(1)); } );
}

// Is the matrix square
bool IsSquareMatrix(const Matrix<Type>& m) {
    return (m.getNumColumns() == m.getNumRows());
}

// Is the matrix diagonal?
bool IsDiagMatrix(const Matrix<Type>& m) {
    // If matrix is rectangular, return false
    if(false == IsSquareMatrix(m)) {
        return false;
    }
    const size_t rows = m.getNumRows();
    const size_t cols = m.getNumColumns();
    for(size_t i{}; i < rows; ++i) {
        for(size_t j{}; j < cols; ++j) {
            if((i != j) && (m(i,j) != (Type)(0))) {
                return false; 
            }
        }
    }
    return true;
}

// Find type of matrix
size_t FindMatType(const Matrix<Type>& m) {
    size_t result{}; 
    // Zero matrix check
    if(true == IsZeroMatrix(m)) {
        result |= MatrixSpl::ZEROS;
    }
    // One matrix check
    if(true == IsOnesMatrix(m)) {
        result |= MatrixSpl::ONES;
    }
    // Identity matrix check
    if(true == IsEyeMatrix(m)) {
        result |= MatrixSpl::EYE;
    }
    // Diagonal matrix check
    if(true == IsDiagMatrix(m)) {
        result |= MatrixSpl::DIAG;
    }
    return result;
}
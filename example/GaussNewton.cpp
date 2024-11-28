/**
 * @file example/GaussNewton.cpp
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

#include "CoolDiff.hpp"

// Number of data points
#define N 5

class GaussNewton {
  private:
    // Parameter matrices for input and output data
    Matrix<Parameter>* m_PX{nullptr};
    Matrix<Parameter>* m_PY{nullptr};

    // Data size, X, Y dataset
    size_t m_size{(size_t)(-1)}; 
    Matrix<Type>* m_X{nullptr};
    Matrix<Type>* m_Y{nullptr}; 
    
    // Oracle
    OracleScalar* m_oracle{nullptr};

    // Jacobian transpose, Residual
    Matrix<Type>* m_jt{nullptr};
    Matrix<Type>* m_res{nullptr};

    // Set data
    void setData(const size_t i) {
        for(size_t j{}; j < m_X->getNumColumns(); ++j) {
          (*m_PX)(0,j) = (*m_X)(i,j); 
        }
        for(size_t j{}; j < m_Y->getNumColumns(); ++j) {
          (*m_PY)(0,j) = (*m_Y)(i,j);
        }
    }

    // Compute jacobians for individual data and store it in m_jt matrix
    void computeJt() {
      const size_t var_size = m_oracle->getVariableSize();
      if(nullptr == m_jt) {
        m_jt = Matrix<Type>::MatrixFactory::CreateMatrixPtr(var_size, m_size);
      }
      for(size_t i{}; i < m_size; ++i) {
        const Pair<size_t,size_t> row_select{0,var_size-1};
        const Pair<size_t,size_t> col_select{i,i};
        setData(i);
        m_jt->setBlockMat(row_select, col_select, &m_oracle->jacobian());  
      }
    }

    // Compute residual for individual data and store it in m_res matrix
    void computeRes() {
      if(nullptr == m_res) {
        m_res = Matrix<Type>::MatrixFactory::CreateMatrixPtr(m_size, 1);
      }
      for(size_t i{}; i < m_size; ++i) {
         setData(i);
         (*m_res)(i,0) = m_oracle->eval();
      }
    }

  public:
    GaussNewton() = default; 

    // Set data (X,Y,size)
    GaussNewton& setData(Matrix<Type>* X, Matrix<Type>* Y, const size_t size) {
      m_size = size;
      m_X = X;
      m_Y = Y;
      return *this;
    }

    // Set data parameters
    GaussNewton& setParameters(Matrix<Parameter>* PX, Matrix<Parameter>* PY) {
      m_PX = PX;
      m_PY = PY;
      return *this;
    }

    // Set oracle
    GaussNewton& setOracle(OracleScalar* oracle) {
      m_oracle = oracle;
      return *this;
    } 

    // Get jacobian
    Matrix<Type>* getJt() {
      computeJt();
      return m_jt;
    }

    // Get residual
    Matrix<Type>* getRes() {
      computeRes();
      return m_res;
    }

    // Get objective
    Type computeObj() {
      // Compute residual and return accumulated value
      computeRes(); 
      return std::accumulate(m_res->getMatrixPtr(), 
                             m_res->getMatrixPtr() + m_res->getNumElem(), 
                             (Type)(0));    
    }
};



void func2() {
  Matrix<Variable> V(3,1);
  V[0] = 3.14159/2; V[1] = 1; V[2] = 2;

  Matrix<Expression> R(2,2);
  Matrix<Expression> t(2,1); 

  Matrix<Parameter> X(2,1);
  Matrix<Parameter> Y(2,1);

  X[0] = 1; X[1] = 3;
  Y[0] = 4; Y[1] = 5;

  R(0,0) = cos( V[0] ); R(0,1) = sin( V[0] );
  R(1,0) = -sin( V[0] ); R(1,1) = cos( V[0] );

  t[0] = V[1]; t[1] = V[2]; 

  Matrix<Expression> J = transpose(Y - (R*X + t))*(Y - (R*X + t));
  Matrix<Expression> J1 = J;
   
  OracleMatrix oracle{J1,V};
  std::cout << oracle.eval() << "\n";
  std::cout << oracle.jacobian() << "\n";

  TIME_IT_US(oracle.jacobian());

  V[0] = 2; V[1] = 9; V[2] = 10;
  std::cout << oracle.eval() << "\n";
  std::cout << oracle.jacobian() << "\n";
}

void func() {
  // Create X-Y pairs of data of size N;
  Matrix<Type> X(N,1); 
  Matrix<Type> Y(N,1);

  // Nonlinear function f(x) = 2*x*x + x + 1
  for(size_t i{}; i < N; ++i) {
    X(i,0) = i; 
    Y(i,0) = 2*X(i,0)*X(i,0) + X(i,0) + 1;
  }

  // Parameters to store data (number of input/output variables ) 
  Matrix<Parameter> PX(1,1);
  Matrix<Parameter> PY(1,1);

  // Variables (coefficients)
  Matrix<Variable> B(3,1);

  // Objective 
  Expression L = pow((PY(0,0) - (B(0,0)*PX(0,0)*PX(0,0) + B(1,0)*PX(0,0) + B(2,0))), 2);
  OracleScalar oracle(L,B);

  GaussNewton gn;
  gn.setData(&X,&Y,N)
    .setOracle(&oracle)
    .setParameters(&PX,&PY);

  Matrix<Type>* res{nullptr};
  for(int i{}; i < 4; ++i) {
    TIME_IT_US(*gn.getJt());
      //MatrixMul(gn.getJt(), gn.getRes(), res);  
      //std::cout << *res << "\n\n";
    B(0,0) = 2*i; B(1,0) = 3*i; B(2,0) = 5*i;
  }
}

int main(int argc, char** argv) {
  func2();
  func();
  return 0;
}
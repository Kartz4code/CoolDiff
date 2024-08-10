# CoolDiff :sunglasses:: A Symbolic/Auto differentiation C++ tool 
---
* **CoolDiff** is a C++ 17 library dedicated to compute both automatic as well as symbolic differentiation of mathematical expressions. 

* **CoolDiff** APIs provide an easy and simple way of modeling and computing scalar mathematical expression values, forward/reverse (adjoint) mode derivatives and symbolic derivatives (objects). 

By default, the numerical type for computation is *complex numbers* and this allows the tool to perform complex analysis as well.   

---
## Classes :white_check_mark:

The main classes involved in **CoolDiff** library are 
1. **Expression** - Expression objects are used to model  mathematical expressions.
2. **Variable** - Variable objects are the optimization/decision/control variables that are of interest for computing the derivatives. 
3. **Parameter** - Parameter objects are used to model scalar variables that are either constant or vary over run-time (e.g. parameter varying system).   
4. **Type** - Type objects are used to model scalar variables, however, unlike Parameters, the value of Type objects *cannot* be changed over run-time. 

---
## Mathematical Operators :white_check_mark:





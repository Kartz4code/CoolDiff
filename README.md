
# CoolDiff :sunglasses:: A Symbolic/Auto differentiation C++ tool  

---

* **CoolDiff** is a C++ 17 library dedicated to compute both automatic as well as symbolic differentiation of mathematical expressions.

* **CoolDiff** APIs provide an easy and simple way of modeling and evaluating scalar and matrix expressions, computing forward/reverse(adjoint) mode derivatives and symbolic derivatives (objects).
  
---

## Building the CoolDiff library :white_check_mark:

The following steps will build the library  

1. Clone the git repository using

 - `git clone https://github.com/Kartz4code/CoolDiff.git`

2. Build the library using following steps

- `cmake -S . -B build`
- `cmake --build build -j16 `

3. Execute the Gaussian example to verify the success of the build
- `./build/example/Gaussian`

Link the generated **CoolDiff** library with your project and include the header `CoolDiff.hpp` to utilize the **CoolDiff** APIs. 

The example folder has few examples to exhibit the use case of the library.  

---
## Classes & Types :white_check_mark:

The main classes/types involved in **CoolDiff** library are:
1. **Expression** - Expression objects are used to model  mathematical expressions.
2. **Variable** - Variable objects are the optimization/decision/control variables that are of interest for computing the derivatives. 
3. **Parameter** - Parameter objects are used to model scalar variables that are either constant or vary over run-time (e.g. parameter varying system).   
4. **Type** - Type objects are used to model scalar variables, however, unlike Parameters, the value of Type objects *cannot* be changed over run-time. 

---
## Mathematical Operators :white_check_mark:

### 1. Binary operators

The list of implemented binary operators in **CoolDiff** library are listed in the following table 

| Function (Operator) | LHS operand type | RHS operand type |
| ----------- | ----------- | ----------- |  
| **Addition** (+) | Variable/Parameter/Expression | Variable/Parameter/Expression
| **Addition** (+) | Type | Variable/Parameter/Expression
| **Addition** (+) | Variable/Parameter/Expression | Type
| **Subtraction** (-) | Variable/Parameter/Expression | Variable/Parameter/Expression
| **Subtraction** (-) | Variable/Parameter/Expression | Type
| **Subtraction** (-) | Type | Variable/Parameter/Expression
| **Multiplication** (*) | Variable/Parameter/Expression | Variable/Parameter/Expression
| **Multiplication** (*) | Type | Variable/Parameter/Expression
| **Multiplication** (*) | Variable/Parameter/Expression | Type
| **Division** (/) | Variable/Parameter/Expression | Variable/Parameter/Expression
| **Division** (/) | Type | Variable/Parameter/Expression
| **Division** (/) | Variable/Parameter/Expression | Type
| **Exponentiation** (pow) | Variable/Parameter/Expression | Variable/Parameter/Expression
| **Exponentiation** (pow) | Type | Variable/Parameter/Expression
| **Exponentiation** (pow) | Variable/Parameter/Expression | Type


### 2. Unary operators

The list of implemented unary operators in **CoolDiff** library are listed in the following table 

| Function (Operator) | Operand type |
| ----------- | ----------- |
| **Sin** (sin) | Variable/Parameter/Expression
| **Cos** (cos) | Variable/Parameter/Expression
| **Tan** (tan) | Variable/Parameter/Expression
| **Asin** (asin) | Variable/Parameter/Expression
| **Acos** (acos) | Variable/Parameter/Expression
| **Atan** (atan) | Variable/Parameter/Expression
| **Sinh** (sinh) | Variable/Parameter/Expression
| **Cosh** (cosh) | Variable/Parameter/Expression
| **Tanh** (tanh) | Variable/Parameter/Expression
| **Asinh** (asinh) | Variable/Parameter/Expression
| **Acosh** (acosh) | Variable/Parameter/Expression
| **Atanh** (atanh) | Variable/Parameter/Expression
| **Exp** (exp) | Variable/Parameter/Expression
| **Log** (log) | Variable/Parameter/Expression
| **Sqrt** (sqrt) | Variable/Parameter/Expression

---
## Important APIs :white_check_mark:

The list of 8 important APIs and along with its description are listed in the following table.  

| API | Description | 1st Argument type | 2nd Argument type | Output type |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| **Eval** | Evaluates the given expression | Expression& | - | Type |
| **DevalF** | Computes the forward AD derivative for the given expression and variable | Expression& | const Variable& | Type |
| **PreComp** | Precomputes all the reverse AD (adjoint) derivatives for the given expression | Expression& | - | - |
| **DevalR** | Returns the reverse AD derivative for the given expression and variable | Expression& | const Variable& | Type |
| **SymDiff** | Computes the symbolic derivative for the given expression and variable | Expression& | const Variable& | Expression& |
| **CreateExpr** | Factory method to create Expression object on heap | const Type& | - | Expression& |
| **CreateVar** | Factory method to create Variable object  on heap | const Type& | - | Expression& |
| **CreateParam** | Factory method to create Parameter object on heap | const Type& | - | Expression& |

---

# About CoolDiff 

**CoolDiff** is currently in its nascent stage. I will constantly make contributions to make this library a better one. Kindly, feel free to try it and in case if you encounter any bugs, issues or errors let me know :smile:. Your feedback would constanly motivate me to make this software a better one.

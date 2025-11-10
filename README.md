
# CoolDiff :sunglasses:: A Symbolic/Automatic differentiation C++ tool  

---

* **CoolDiff** is a C++ 17 library dedicated to compute both automatic as well as symbolic differentiation of scalar and matrix mathematical expressions.

* **CoolDiff** APIs provide an easy and simple way of modeling and evaluating scalar and matrix expressions, computing forward/reverse(adjoint) mode derivatives and symbolic derivatives (for scalar expressions).
  
---

## Building the CoolDiff library :white_check_mark:

The following steps will build the library  

1. Clone the git repository using

 - `git clone https://github.com/Kartz4code/CoolDiff.git`

2. Build the library using following steps

- `cmake -S . -B build`
- `cmake --build build -j${nproc}`

1. Install the **CoolDiff** library using 
- `sudo cmake --install build`

Link the generated **CoolDiff** library with your project and include the header `CoolDiff.hpp` to utilize the **CoolDiff** APIs (See CMakeLists.txt in the examples folder). 

Kindly refer to the examples folder on usage of **CoolDiff** APIs for your application.

---

# About CoolDiff 

**CoolDiff** is currently in its nascent stage. I will constantly make contributions to make this library a better one. Kindly, feel free to try it and in case if you encounter any bugs, issues or errors let me know :smile:. Your feedback would constanly motivate me to make this software a better one.

#pragma once 

#include "Layer.hpp"

// Base class definition for custom implementation
#define BASE_CLASS_DEF(CLAZZ)   using Base = NeuralNet< CLAZZ >;\
                                using Base::Base;                     \
                                using Base::W; using Base::b;         \
                                using Base::X; using Base::Y;


template<size_t N, typename... Args>
auto GetLayer(const Tuples<Args...>& tuple) {
    return std::get<N>(tuple);
}

// GDOptimizer
void GDOptimizer(Matrix<Type>& X, Matrix<Type>& dX, const Type& alpha) {
    Matrix<Type>* dX_ptr = &dX;
    Matrix<Type>* X_ptr = &X;

    CoolDiff::TensorR2::MatOperators::MatrixScalarMul(alpha, &dX, dX_ptr);
    CoolDiff::TensorR2::MatOperators::MatrixAdd(&X, dX_ptr, X_ptr);
}


// Templatized neural net: T -> for CRTP, N -> Layer depth
template<typename T>
class NeuralNet {
    private:
        // CRTP derived function
        inline constexpr T& derived() { 
            return static_cast<T&>(*this); 
        }

        // Vector of layers
        Vector<Layer> m_layers; 

    protected:
        // Inputs/Outputs
        Matrix<Type> X, Y;

    public:
        // Default constructor
        NeuralNet(const Pair<size_t, size_t>& Xdim, const Pair<size_t, size_t>& Ydim) {
            X = Matrix<Type>::MatrixFactory::CreateMatrix(Xdim.first, Xdim.second);
            Y = Matrix<Type>::MatrixFactory::CreateMatrix(Ydim.first, Ydim.second);
        }

        // Add layer to NeuralNet
        NeuralNet& addLayer(const Layer& layer) {
            m_layers.push_back(layer);
            return *this;
        }

        // Get size
        const size_t size() const {
            return m_layers.size();
        }

        // Get weights parameter for nth layer
        Matrix<Type>& W(const size_t n) {
            std::string msg = "Weights not defined at Layer: " + std::to_string(n);
            ASSERT((n < size()), msg);
            return m_layers[n].W();
        }  

        // Get bias parameter for nth layer
        Matrix<Type>& b(const size_t n) {
            std::string msg = "Bias not defined at Layer: " + std::to_string(n);
            ASSERT((n < size()), msg);
            return m_layers[n].b();
        }  

        // CRTP function - networkLayers (Returns tuple of layers)
        auto networkLayers() {
            return derived().networkLayers();
        }

        // CRTP function - error (Returns error/objective)
        auto error() {
            return derived().error();
        } 
    
        // Predict output for a test input and the tuple of layers for the Mth layer
        template<size_t M, typename... Args>
        auto& predict(Matrix<Type>& X_test, Tuples<Args...>& tuple) {
            static_assert((M < std::tuple_size_v<Tuples<Args...>>), "Accessing network layer beyond M");
            // Copy test data into X
            X.copyData(X_test);
            return CoolDiff::TensorR2::Eval(std::get<M>(tuple));
        }

        void train(Matrix<Type>& X_data, Matrix<Type>& Y_data) {
            Matrix<Expression> err = error();
            Type alpha = -0.00001;

            for(int i{}; i < 50; ++i) {
                // Copy X and Y data into X and Y 
                X.copyData(X_data); 
                Y.copyData(Y_data);

                std::cout << "[ERROR]: " << CoolDiff::TensorR2::Eval(err) << "\n";
                CoolDiff::TensorR2::PreComp(err);

                // Dispatch to thread
                Vector<std::thread> thread_vec;
                for(size_t i{}; i < size(); ++i) {
                    auto& dW = CoolDiff::TensorR2::DevalR(err, W(i));
                    auto& db = CoolDiff::TensorR2::DevalR(err, b(i));
                    thread_vec.emplace_back(std::thread(GDOptimizer, std::ref(W(i)), std::ref(dW), alpha));
                    thread_vec.emplace_back(std::thread(GDOptimizer, std::ref(b(i)), std::ref(db), alpha));
                }

                // Join threads
                for(auto& item : thread_vec) {
                    if(true == item.joinable()) {
                        item.join();
                    }
                }

            }
        }

        // NeuralNet default destructor
        ~NeuralNet() = default;
};


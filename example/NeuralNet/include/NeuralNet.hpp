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
        // Thread vector
        Vector<std::thread> m_thread_vec; 

        // Reset values for mini batch
        void resetMiniBatch(bool threading) {
            if(true == threading) {
                // Dispatch threads
                for(size_t r{}; r < size(); ++r) {
                    m_thread_vec.emplace_back(std::thread(CoolDiff::TensorR2::Details::ResetZero, std::ref(WBatch(r))));
                    m_thread_vec.emplace_back(std::thread(CoolDiff::TensorR2::Details::ResetZero, std::ref(bBatch(r))));
                }
                // Join threads
                for(auto& item : m_thread_vec) {
                    if(true == item.joinable()) {
                        item.join();
                    }
                }
            } else {
                for(size_t r{}; r < size(); ++r) {
                    CoolDiff::TensorR2::Details::ResetZero(WBatch(r));
                    CoolDiff::TensorR2::Details::ResetZero(bBatch(r));
                }
            }
        }

        // Accumulate gradients
        void accumulateGradients(Matrix<Expression>& err, bool threading) {
            // Precompute the errors
            CoolDiff::TensorR2::PreComp(err);

            if(true == threading) {
                // Dispatch threads
                for(size_t r{}; r < size(); ++r) {
                    auto& dW = CoolDiff::TensorR2::DevalR(err, W(r));
                    auto& db = CoolDiff::TensorR2::DevalR(err, b(r));
                    m_thread_vec.emplace_back(std::thread(CoolDiff::TensorR2::MatOperators::MatrixAdd, WBatch(r), &dW, std::ref(WBatch(r))));
                    m_thread_vec.emplace_back(std::thread(CoolDiff::TensorR2::MatOperators::MatrixAdd, bBatch(r), &db, std::ref(bBatch(r))));
                }
                // Join threads
                for(auto& item : m_thread_vec) {
                    if(true == item.joinable()) {
                        item.join();
                    }
                }
            } else {
                for(size_t r{}; r < size(); ++r) {
                    auto& dW = CoolDiff::TensorR2::DevalR(err, W(r));
                    auto& db = CoolDiff::TensorR2::DevalR(err, b(r));
                    CoolDiff::TensorR2::MatOperators::MatrixAdd(WBatch(r), &dW, WBatch(r));
                    CoolDiff::TensorR2::MatOperators::MatrixAdd(bBatch(r), &db, bBatch(r));
                }
            }
        }

        // optimize
        void optimize(const Type& alpha, bool threading) {
            if(true == threading) {
                // Dispatch threads
                for(size_t r{}; r < size(); ++r) {
                    auto& dW = (*WBatch(r));
                    auto& db = (*bBatch(r));
                    m_thread_vec.emplace_back(std::thread(GDOptimizer, std::ref(W(r)), std::ref(dW), alpha));
                    m_thread_vec.emplace_back(std::thread(GDOptimizer, std::ref(b(r)), std::ref(db), alpha));
                }
                // Join threads
                for(auto& item : m_thread_vec) {
                    if(true == item.joinable()) {
                        item.join();
                    }
                }
            } else {
                for(size_t r{}; r < size(); ++r) {
                    auto& dW = (*WBatch(r));
                    auto& db = (*bBatch(r));
                    GDOptimizer(W(r), dW, alpha);
                    GDOptimizer(b(r), db, alpha);
                }
            }
        }

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

        // Get weights parameter for nth layer
        Matrix<Type>*& WBatch(const size_t n) {
            std::string msg = "Weights not defined at Layer: " + std::to_string(n);
            ASSERT((n < size()), msg);
            return m_layers[n].WBatch();
        }  

        // Get bias parameter for nth layer
        Matrix<Type>*& bBatch(const size_t n) {
            std::string msg = "Bias not defined at Layer: " + std::to_string(n);
            ASSERT((n < size()), msg);
            return m_layers[n].bBatch();
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
        auto& predict(const Matrix<Type>& X_test, Tuples<Args...>& tuple) {
            static_assert((M < std::tuple_size_v<Tuples<Args...>>), "Accessing network layer beyond M");
            // Copy test data into X
            X.copyData(X_test);
            return CoolDiff::TensorR2::Eval(std::get<M>(tuple));
        }

        
        void train( Matrix<Type>& X_data, Matrix<Type>& Y_data, 
                    const Type& rate = -0.01, const size_t batch_size = 1, const size_t epoch = 1, 
                    bool threading = false, bool display_stats = true ) {

            // Matrix error
            Matrix<Expression> err = error();
            // Learning rate (normalized by batch size)
            Type alpha = (rate/((Type)batch_size)); 

            // Data size
            size_t data_size = Y_data.getNumRows(); 
            // Stats logger
            std::ostringstream oss; 

            // Loop over the epochs 
            for(size_t i{}; i < epoch; ++i) {
                Type acc_err{};
                // Loop over the batch size
                for(size_t j{}; j < data_size; j += batch_size) {
                    // Reset all batch matrices to zero for the next batch
                    resetMiniBatch(threading);

                    // Loop over the data set for a fixed batch
                    for(size_t k{j}; (k < (j + batch_size)) && (k < data_size-1); ++k) {
                        // Copy X and Y data into X and Y 
                        X.copyData(X_data.getRow(k)); 
                        Y.copyData(Y_data.getRow(k));
                        
                        // Accumulate gradients
                        accumulateGradients(err, threading);

                        // Accumulate error 
                        acc_err += CoolDiff::TensorR2::Eval(err)(0,0);
                    }

                    // Mini batch update
                    optimize(alpha, threading);
                }

                // Display stats
                if(true == display_stats) {
                    oss << "       <------------------------------ Computation stats ------------------------------>       \n"
                        << std::string(100, '-') << "\n" << std::string(3, ' ')
                        << "| [EPOCH]: " << std::to_string(i) << " |"
                        << " [LOSS ERROR]: " << std::to_string(acc_err) << " |"
                        << " [BATCH SIZE]: " << std::to_string(batch_size) << " | "
                        << " [LEARNING RATE]: " << std::to_string(alpha) << " |\n"
                        << std::string(100, '-') << "\n";

                    std::cout << oss.str() << "\n";
                    oss.clear(); oss.str("");
                }
            }
        }

        // NeuralNet default destructor
        ~NeuralNet() = default;
};


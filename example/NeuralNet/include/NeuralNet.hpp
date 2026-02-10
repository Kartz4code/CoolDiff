#pragma once 

#include "Layer.hpp"

// Base class definition for custom implementation
#define BASE_CLASS_DEF(CLAZZ)   using Base = NeuralNet< CLAZZ >;\
                                using Base::Base;                     \
                                using Base::W; using Base::b;         \
                                using Base::X; using Base::Y;

// GDOptimizer
void GDOptimizer(Matrix<Type>& X, Matrix<Type>& dX, const Type& alpha) {
    Matrix<Type>* dX_ptr = &dX;
    Matrix<Type>* X_ptr = &X;

    MATRIX_SCALAR_MUL(alpha, &dX, dX_ptr);
    MATRIX_ADD(&X, dX_ptr, X_ptr);
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
                for(size_t r{}; r < layerSize(); ++r) {
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
                for(size_t r{}; r < layerSize(); ++r) {
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
                for(size_t r{}; r < layerSize(); ++r) {
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
                for(size_t r{}; r < layerSize(); ++r) {
                    auto& dW = CoolDiff::TensorR2::DevalR(err, W(r));
                    auto& db = CoolDiff::TensorR2::DevalR(err, b(r));
                    MATRIX_ADD(WBatch(r), &dW, WBatch(r));
                    MATRIX_ADD(bBatch(r), &db, bBatch(r));
                }
            }
        }

        // Optimize
        void optimize(const Type& alpha, bool threading) {
            if(true == threading) {
                // Dispatch threads
                for(size_t r{}; r < layerSize(); ++r) {
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
                for(size_t r{}; r < layerSize(); ++r) {
                    auto& dW = (*WBatch(r));
                    auto& db = (*bBatch(r));
                    GDOptimizer(W(r), dW, alpha);
                    GDOptimizer(b(r), db, alpha);
                }
            }
        }

        // Classification accuracy computation for internal use (no or infrequent restructuring of the network batch size)
        template<typename Z>
        Type accuracy(Z& net, const Matrix<Type>& X_data, const Matrix<Type>& Y_data, const size_t batch_size) {
            size_t count{}; 
            const size_t data_size = X_data.getNumRows(); 
            const size_t feature_size = X_data.getNumColumns();
            const size_t label_size = Y_data.getNumColumns();

            // Unrandomize dropout
            randomizeDropout<false>();

            // Data scan size
            const size_t scan_size = (data_size-batch_size);
            for(size_t i{}; i < scan_size; i+=batch_size) {               
                // Predicted and real values
                X.setMatrixPtr(X_data.getRowPtr(i));
                Y.setMatrixPtr(Y_data.getRowPtr(i));

                // Get prediction from final layer
                auto& Yhat = CoolDiff::TensorR2::Eval(GetFinalLayer(net));

                // Identify estimated vs real classes and increment for positive count
                for(size_t j{}; j < batch_size; ++j) {
                    const auto Yhat_ptr = Yhat.getRowPtr(j);
                    const auto Y_ptr = Y.getRowPtr(j);

                    const size_t Yhat_class = std::distance(Yhat_ptr, std::max_element(Yhat_ptr, Yhat_ptr + label_size));
                    const size_t Y_class = std::distance(Y_ptr, std::max_element(Y_ptr, Y_ptr + label_size));

                    // Increment count if values match up between predicted and real
                    if(Yhat_class == Y_class) {
                        count += 1;
                    }
                }
            }

            // Convert to percentage and return
            return Type((count/(Type)(scan_size))*100.0);
        }

        // Randomize dropout
        template<bool Randomize>
        void randomizeDropout() {
            if constexpr(true == Randomize) {
                // Set the radnomized dropout matrices with bernoulli distribution 
                for(auto& [p, W] : m_dropout_mat) {
                    CoolDiff::TensorR2::Details::FillRandomValues<BernoulliDistribution>((*W), p);
                    // Transform the matrix with normalization factor of (1/p)
                    std::transform(EXECUTION_PAR W->getMatrixPtr(), W->getMatrixPtr() + W->getNumElem(),
                                                 W->getMatrixPtr(), [&p](auto& item) { return ((Type)1/p)*item; });
                }
            } else {
                // Set the non-randomized dropout matirces to Ones(rows, cols)
                for(auto& [p, W] : m_dropout_mat) {
                    std::fill(EXECUTION_PAR W->getMatrixPtr(), W->getMatrixPtr() + W->getNumElem(), 1);
                }
            }
        }

        // Dropout probability
        Vector<Pair<Type, Matrix<Type>*>> m_dropout_mat;
        // Dropout pool
        Vector<Matrix<Type>*> m_dropout_pool;

    protected:
        // Inputs/Output
        Matrix<Type> X{1, 1, nullptr};
        Matrix<Type> Y{1, 1, nullptr};

        // Input/Ouput dimensions
        Pair<size_t, size_t> m_iodim;
        size_t m_batch_size{1};

        // Dropout layer is internal to neural network
        template<typename Z>
        constexpr const auto& Dropout(const IMatrix<Z>& X, const Type p) {
            // Dimension of X matrix          
            const size_t xrows = X.getNumRows();
            const size_t xcols = X.getNumColumns();

            // Predicate function
            auto predicate = [xrows, xcols](const auto& item) { return ((item->getNumRows() == xrows) && 
                                                                        (item->getNumColumns() == xcols)); };

            // Reuse from pool, if possible
            if(auto it = std::find_if(EXECUTION_PAR m_dropout_pool.begin(), m_dropout_pool.end(), predicate); 
                    it != m_dropout_pool.end()) {
                Matrix<Type>* W = *it;
                m_dropout_mat.push_back({(1-p), W});
                m_dropout_pool.erase(it);
                return (*W)^X;
            } else {
                Matrix<Type>* W = Matrix<Type>::MatrixFactory::CreateMatrixPtr(xrows, xcols, "GPUPinnedMemoryStrategy");
                m_dropout_mat.push_back({(1-p), W});
                return (*W)^X;
            }
        }

    public:
        // Default constructor
        NeuralNet(Pair<size_t, size_t> iodim) : m_iodim{iodim} {}

        // Add layer to NeuralNet
        NeuralNet& addLayer(const Layer& layer) {
            m_layers.push_back(layer);
            return *this;
        }

        // Get size
        const size_t layerSize() const {
            return m_layers.size();
        }

        // Get weights parameter for nth layer
        Matrix<Type>& W(const size_t n) {
            std::string msg = "Weights not defined at Layer: " + std::to_string(n);
            ASSERT((n < layerSize()), msg);
            return m_layers[n].W();
        }  

        // Get bias parameter for nth layer
        Matrix<Type>& b(const size_t n) {
            std::string msg = "Bias not defined at Layer: " + std::to_string(n);
            ASSERT((n < layerSize()), msg);
            return m_layers[n].b();
        }  

        // Get weights parameter for nth layer
        Matrix<Type>*& WBatch(const size_t n) {
            std::string msg = "Weights not defined at Layer: " + std::to_string(n);
            ASSERT((n < layerSize()), msg);
            return m_layers[n].WBatch();
        }  

        // Get bias parameter for nth layer
        Matrix<Type>*& bBatch(const size_t n) {
            std::string msg = "Bias not defined at Layer: " + std::to_string(n);
            ASSERT((n < layerSize()), msg);
            return m_layers[n].bBatch();
        } 

        // CRTP function - networkLayers (Returns tuple of layers)
        auto networkLayers(const size_t batch_size) {
            // If m_dropout_mat not empty
            if(true != m_dropout_mat.empty()) {
                // Push to m_dropout_pool 
                for(auto& [p, mat] : m_dropout_mat) {
                    m_dropout_pool.push_back(mat);
                }
                // Clear m_dropout_mat
                m_dropout_mat.clear();
            }

            m_batch_size = batch_size;
            X = Matrix<Type>(m_batch_size, m_iodim.first, nullptr, "GPUPinnedMemoryStrategy");
            Y = Matrix<Type>(m_batch_size, m_iodim.second, nullptr, "GPUPinnedMemoryStrategy");
            return derived().networkLayers(m_batch_size);
        }

        // CRTP function - error (Returns error/objective)
        template<typename Z>
        auto error(Z& net) {
            return derived().error(net);
        } 
    
        // Predict output for a test input and the tuple of layers for the Mth layer
        template<size_t M, typename... Args>
        auto& predict(const Matrix<Type>& X_test, Tuples<Args...>& tuple) {
            static_assert((M < std::tuple_size_v<Tuples<Args...>>), "Accessing network layer beyond M");
            return std::get<M>(tuple);
        }

        // Get Mth layer
        template<size_t M, typename... Args>
        auto& GetLayer(Tuples<Args...>& tuple) {
            static_assert((M < std::tuple_size_v<Tuples<Args...>>), "Accessing network layer beyond M");
            return std::get<M>(tuple);
        }

        // Get final layer
        template<typename... Args>
        auto& GetFinalLayer(Tuples<Args...>& tuple) {
            constexpr const size_t N = (std::tuple_size_v<Tuples<Args...>>-1);
            return GetLayer<N>(tuple);
        }

        // Classification accuracy computation for external use, i.e. varying batch size (test case)
        Type accuracy(const Matrix<Type>& X_data, const Matrix<Type>& Y_data, const size_t batch_size) {
            size_t count{}; 
            const size_t data_size = X_data.getNumRows(); 
            const size_t feature_size = X_data.getNumColumns();
            const size_t label_size = Y_data.getNumColumns();

            // Restructure network layer to match batch size
            auto net = networkLayers(batch_size);

            // Unrandomize dropout
            randomizeDropout<false>();

            // Data scan size
            const size_t scan_size = (data_size-batch_size);
            for(size_t i{}; i < scan_size; i+=batch_size) {               
                // Predicted and real values
                X.setMatrixPtr(X_data.getRowPtr(i));
                Y.setMatrixPtr(Y_data.getRowPtr(i));

                // Get prediction from final layer
                auto& Yhat = CoolDiff::TensorR2::Eval(GetFinalLayer(net));

                // Identify estimated vs real classes and increment for positive count
                for(size_t j{}; j < batch_size; ++j) {
                    const auto Yhat_ptr = Yhat.getRowPtr(j);
                    const auto Y_ptr = Y.getRowPtr(j);

                    const size_t Yhat_class = std::distance(Yhat_ptr, std::max_element(Yhat_ptr, Yhat_ptr + label_size));
                    const size_t Y_class = std::distance(Y_ptr, std::max_element(Y_ptr, Y_ptr + label_size));

                    // Increment count if values match up between predicted and real
                    if(Yhat_class == Y_class) {
                        count += 1;
                    }
                }
            }

            // Convert to percentage and return
            return Type((count/(Type)(scan_size))*100.0);
        }

        
        void train(Matrix<Type>& X_data, Matrix<Type>& Y_data, const Type& rate = -0.01, 
                   const size_t epoch = 1, bool threading = false, bool display_stats = true) {

            // Network layer and error function (Must be in loop for adaptive batch update)
            auto net = networkLayers(m_batch_size);
            Matrix<Expression> err = error(net);

            // Learning rate (normalized by batch size)
            Type alpha = (rate/((Type)m_batch_size)); 

            // Data size
            size_t data_size = Y_data.getNumRows();

            // Stats logger
            std::ostringstream oss; 

            // Loop over the epochs 
            for(size_t i{}; i < epoch; ++i) {
            
                // Accumulated error
                Type acc_err{};

                // Loop over the batch size
                for(size_t j{}; j < data_size; j += m_batch_size) {
                    // Logic for training the last batch of the dataset
                    if((j + m_batch_size) > data_size) {
                        j = data_size - m_batch_size;
                    } 

                    // Randomize dropout 
                    randomizeDropout<true>();

                    // Reset all batch matrices to zero for the next batch
                    resetMiniBatch(threading);

                    // Set matrix pointers
                    X.setMatrixPtr(X_data.getRowPtr(j));
                    Y.setMatrixPtr(Y_data.getRowPtr(j));

                    // Accumulate gradients
                    accumulateGradients(err, threading);

                    // Accumulate error 
                    if(true == display_stats) {
                        acc_err += CoolDiff::TensorR2::Details::ScalarSpl(&CoolDiff::TensorR2::Eval(err));
                    }
                    // Mini batch update
                    optimize(alpha, threading);
                }

                // Display stats
                if(true == display_stats) {
                    Type acc{};
                    acc = accuracy(net, X_data, Y_data, m_batch_size);
                    oss << "       <------------------------------ Computation stats ------------------------------>       \n"
                        << std::string(100, '-') << "\n" << std::string(10, ' ')
                        << "| [EPOCH]: " << std::to_string(i) << " |"
                        << " [LOSS VALUE]: " << std::to_string(acc_err) << " |"
                        << " [ACCURACY]: " << std::to_string(acc) << "% |\n"
                        << std::string(100, '-') << "\n";

                    std::cout << oss.str() << "\n";
                    oss.clear(); oss.str("");
                }
            };
        }

        // NeuralNet default destructor
        ~NeuralNet() = default;
};


#pragma once 

#include "CoolDiff.hpp"
#include <random>

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> dis(-1, +1); 

static void FillRandomWeights(MatType& M) {
    for(int i{}; i < M.getNumRows(); ++i) {
        for(int j{}; j < M.getNumColumns(); ++j) {
            M(i,j) = dis(gen);
        }
    }
}

// Layer class
class Layer {
    private:
        // Layer weights and bias parameters
        Matrix<Type> m_W;
        Matrix<Type> m_b;

        public:
            // Default constructor
            Layer() = default; 

            class LayerFactory {
                public:
                    LayerFactory() = default;
                    static Layer CreateLayer(const size_t);
                    static Layer CreateLayer(const size_t, const size_t);
                    ~LayerFactory() = default;
            };

        
        // Get weight parameters for the layer
        Matrix<Type>& W();

        // Get bias parameters for the layer 
        Matrix<Type>& b();

        // Get weight matrix size
        Pair<size_t, size_t> WDim() const {
            return { m_W.getNumRows(), m_W.getNumColumns() };
        }

        // Get bias matrix size
        Pair<size_t, size_t> bDim() const {
            return { m_b.getNumRows(), m_b.getNumColumns() };
        }

        // Default destructor
        ~Layer() = default;
};

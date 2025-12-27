#pragma once 

#include "CoolDiff.hpp"

// Layer class
class Layer {
    private:
        // Layer weights and bias parameters
        Matrix<Type> m_W;
        Matrix<Type> m_b;

        // Layer weights and bias copy for batch operations
        Matrix<Type>* m_WBatch{nullptr};
        Matrix<Type>* m_bBatch{nullptr};

        // Layer dimensions
        Pair<size_t, size_t> m_dim;

        public:
            // Default constructor
            Layer() = default; 

            class LayerFactory {
                public:
                    LayerFactory() = default;
                    // Create layer of size (N X N) for weights and (N X 1) for bias with randomization
                    template<template <typename> class T, typename U, typename... Args> 
                    static Layer CreateLayer(const U N, Args&&... args) {
                        // Check whether U is an integral type
                        static_assert(true == std::is_integral_v<U>, "N is not an integral type");
                        return CreateLayer<T>(N, N, std::forward<Args>(args)...);
                    }

                    // Create layer of size (N X M) for weights and (M x 1) for bias with randomization
                    template<template <typename> class T, typename U, typename V, typename... Args>
                    static Layer CreateLayer(const U N, const V M, Args&&... args) {
                        // Check whether U and V are integral types
                        static_assert(true == std::is_integral_v<U>, "N is not an integral type");
                        static_assert(true == std::is_integral_v<V>, "M is not an integral type");

                        Layer result; 

                        result.m_W = Matrix<Type>::MatrixFactory::CreateMatrix(N, M);
                        result.m_b = Matrix<Type>::MatrixFactory::CreateMatrix(N, 1);

                        result.m_dim = {N, M};

                        CoolDiff::TensorR2::Details::FillRandomValues<T>(result.m_W, std::forward<Args>(args)...);
                        CoolDiff::TensorR2::Details::FillRandomValues<T>(result.m_b, std::forward<Args>(args)...);

                        return result;
                    }

                    // Create layer of size (N X N) for weights and (N X 1) for bias with no randomization
                    static Layer CreateLayer(const size_t);

                    // Create layer of size (N X M) for weights and (M x 1) for bias with no randomization
                    static Layer CreateLayer(const size_t, const size_t);


                    ~LayerFactory() = default;
            };

        
        // Get weight parameters for the layer
        Matrix<Type>& W();

        // Get bias parameters for the layer 
        Matrix<Type>& b();

        // Get weight parameters for the layer for batch processing
        Matrix<Type>*& WBatch();

        // Get bias parameters for the layer for batch processing
        Matrix<Type>*& bBatch();

        // Get weight matrix size
        Pair<size_t, size_t> WDim() const;

        // Get bias matrix size
        Pair<size_t, size_t> bDim() const;

        // Default destructor
        ~Layer() = default;
};

#include "Layer.hpp"

// Create layer of size (N X N) for weights and (N X 1) for bias with no randomization
Layer Layer::LayerFactory::CreateLayer(const size_t N) {
    return CreateLayer(N, N);
}

// Create layer of size (N X M) for weights and (M x 1) for bias with no randomization
Layer Layer::LayerFactory::CreateLayer(const size_t N, const size_t M) {
    Layer result; 

    result.m_W = Matrix<Type>::MatrixFactory::CreateMatrixPtr(N, M, "GPUPinnedMemoryStrategy");
    result.m_b = Matrix<Type>::MatrixFactory::CreateMatrixPtr(N, 1, "GPUPinnedMemoryStrategy");

    result.m_dim = {N, M};
    
    return result;
}

// Get weight parameters for the layer
Matrix<Type>& Layer::W() {
    return *m_W;
}

// Get bias parameters for the layer 
Matrix<Type>& Layer::b() {
return *m_b;
}

// Get weight parameters for the layer for batch processing
Matrix<Type>*& Layer::WBatch() {
    if(nullptr == m_WBatch) {
        m_WBatch = Matrix<Type>::MatrixFactory::CreateMatrixPtr(m_dim.first, m_dim.second, "GPUPinnedMemoryStrategy");
    }
    return m_WBatch;
}

// Get bias parameters for the layer for batch processing
Matrix<Type>*& Layer::bBatch() {
    if(nullptr == m_bBatch) {
        m_bBatch = Matrix<Type>::MatrixFactory::CreateMatrixPtr(m_dim.first, 1, "GPUPinnedMemoryStrategy");
    }
    return m_bBatch;
}

// Get weight matrix size
Pair<size_t, size_t> Layer::WDim() const {
    return { m_W->getNumRows(), m_W->getNumColumns() };
}

// Get bias matrix size
Pair<size_t, size_t> Layer::bDim() const {
    return { m_b->getNumRows(), m_b->getNumColumns() };
}
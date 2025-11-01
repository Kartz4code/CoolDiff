#include "Layer.hpp"

Layer Layer::LayerFactory::CreateLayer(const size_t N) {
    return CreateLayer(N, N);
}

Layer Layer::LayerFactory::CreateLayer(const size_t N, const size_t M) {
    Layer result; 
    result.m_W = Matrix<Type>::MatrixFactory::CreateMatrix(N, M);
    result.m_b = Matrix<Type>::MatrixFactory::CreateMatrix(N, 1);
    FillRandomWeights(result.m_W);
    FillRandomWeights(result.m_b);
    return result;
}

Matrix<Type>& Layer::W() {
    return m_W;
}

Matrix<Type>& Layer::b() {
return m_b;
}
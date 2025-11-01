#include "NeuralNet.hpp"

class CustomNet : public NeuralNet<CustomNet> {
    private:
        BASE_CLASS_DEF(CustomNet);
        
    public:
        CustomNet() = default;

        auto networkLayers() {
            auto l1 = TanhM(W(0)*X + b(0));
            auto l2 = TanhM(W(1)*l1 + b(1));
            auto l3 = TanhM(W(2)*concat(X,l2) + b(2));
            auto l4 = TanhM(W(3)*l3 + b(3));
            auto l5 = (W(4)*l4 + b(4));

            auto list = std::make_tuple(l1, l2, l3, l4, l5);
            return list;
        }

        auto error() { 
            //auto& Yp = Matrix<Expression>::MatrixFactory::CreateMatrix(GetLayer<4>(networkLayers()));
            auto Yp = GetLayer<4>(networkLayers());
            auto error = (Y - Yp)*100*(Y - Yp);
            return error;
        }
        
        ~CustomNet() = default;  
};

void NN() {
    const size_t N{256};

    CustomNet n({N,1},{1,1});
    auto tuple = n.addLayer(Layer::LayerFactory::CreateLayer(N, N))
                    .addLayer(Layer::LayerFactory::CreateLayer(N, N))
                    .addLayer(Layer::LayerFactory::CreateLayer(N, 2*N))
                    .addLayer(Layer::LayerFactory::CreateLayer(N, N))
                    .addLayer(Layer::LayerFactory::CreateLayer(1, N))
                    .networkLayers();

    Matrix<Type> X(N, 1); Matrix<Type> Y(1,1);
    FillRandomWeights(X); Y(0,0) = 10;
    n.train(X, Y);

    Matrix<Type> Xtest(N, 1);
    FillRandomWeights(Xtest);
    std::cout << CoolDiff::TensorR2::Eval(n.predict<4>(Xtest, tuple)) << "\n";
    std::cout << CoolDiff::TensorR2::Eval(n.predict<4>(X, tuple)) << "\n";
}

int main(int argc, char** argv) {
    #ifndef USE_COMPLEX_MATH
        NN();
    #endif
    return 0;
}
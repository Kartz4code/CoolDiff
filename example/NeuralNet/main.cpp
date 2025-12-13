#include "NeuralNet.hpp"
#include "MNISTData.hpp"
#include "csv.hpp"

// Custom net design 
class CustomNet : public NeuralNet<CustomNet> {
    private:
        BASE_CLASS_DEF(CustomNet);
        
    public:
        CustomNet() = default;

        // CRTP overload - networkLayers
        auto networkLayers(const size_t batch) {
            auto l1 = LeakyReLUM(X*transpose(W(0)) + transpose(broadcast<Axis::COLUMN>(b(0), batch)));
            auto l2 = LeakyReLUM(l1*transpose(W(1)) + transpose(broadcast<Axis::COLUMN>(b(1), batch)));
            auto l3 = SoftMax<Axis::COLUMN>(l2);

            auto list = std::make_tuple(l1, l2, l3);
            return list;
        }

        // CRTP overload - error
        template<typename Z>
        auto error(Z& net) { 
            // Get final layer of the network
            auto Yp = GetFinalLayer(net);

            // Cross entropy objective
            Matrix<Expression> error =  -1 * Sigma(Y^LogM(Yp));

            // Error with L2 regulatization
            for(size_t i{}; i < layerSize(); ++i) {
                error = error + 0.01*(Sigma((W(i)^W(i))) + Sigma((b(i)^b(i))));
            }

            return error;
        }
        
        ~CustomNet() = default;  
};

// Load MNIST data
void LoadData(std::string_view path, Matrix<Type>& X, Matrix<Type>& Y) {
    csv::CSVReader reader(path);
    size_t irows{}; size_t icols{};
    for (csv::CSVRow& row: reader) {
        for (csv::CSVField& field: row) {
            if(0 == icols) {
                Y(irows, field.get<size_t>()) = 1;
            } else {
                X(irows, icols-1) = (field.get<Type>()/(Type)255);
            }
            ++icols; 
        }
        icols = 0; ++irows;
    }
}

// Distribution
template<typename T>
using UniformDistribution = std::uniform_real_distribution<T>;
template<typename T>
using NormalDistribution = std::normal_distribution<T>;


void MNISTPrediction() {
    // Dimension of input, ouput and batch size
    const size_t N{784}, M{10}, K{32};

    // Train MNIST data (60000 x 784)
    Matrix<Type>& Xtrain = Matrix<Type>::MatrixFactory::CreateMatrix(60000, N);
    Matrix<Type>& Ytrain = Matrix<Type>::MatrixFactory::CreateMatrix(60000, M);
    
    // Test MNIST data (10000 x 784)
    Matrix<Type>& Xtest = Matrix<Type>::MatrixFactory::CreateMatrix(10000, N);
    Matrix<Type>& Ytest = Matrix<Type>::MatrixFactory::CreateMatrix(10000, M);

    // Load MNIST train and test data
    LoadData(MNISTData::g_mnist_train_data_path, Xtrain, Ytrain);
    LoadData(MNISTData::g_mnist_test_data_path, Xtest, Ytest);

    CustomNet n({N,M});

    // Generate network layers for prediction on test data
    auto net  =  n.addLayer(Layer::LayerFactory::CreateLayer<UniformDistribution>(5*M, N, -1, 1))
                  .addLayer(Layer::LayerFactory::CreateLayer<UniformDistribution>(M, 5*M, -1, 1))
                  .networkLayers(K);

    // Train data
    TIME_IT_MS(n.train(Xtrain, Ytrain, -0.05, 50));

    // Prediction test
    std::cout << "[Test prediction accuracy]: " << n.accuracy(Xtest, Ytest, 1) << "%\n";
}

int main(int argc, char** argv) {
    #ifndef USE_COMPLEX_MATH
        MNISTPrediction();
    #endif
    return 0;
}
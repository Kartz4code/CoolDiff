#include "NeuralNet.hpp"
#include "MNISTData.hpp"
#include "csv.hpp"

class CustomNet : public NeuralNet<CustomNet> {
    private:
        BASE_CLASS_DEF(CustomNet);
        
    public:
        CustomNet() = default;

        auto networkLayers() {
            auto l1 = LeakyReLUM(W(0)*X + b(0));
            auto l2 = LeakyReLUM(W(1)*l1 + b(1));
            auto l3 = SoftMax(l2);

            auto list = std::make_tuple(l1, l2, l3);
            return list;
        }

        auto error() { 
            auto Yp = GetFinalLayer(networkLayers());
            auto error = -1*transpose(Y)*LogM(Yp);
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

void MNISTPrediction() {
    const size_t N{784}, M{10};

    CustomNet n({N,1},{M,1});

    // Generate network layers
    auto tuple = n.addLayer(Layer::LayerFactory::CreateLayer<std::uniform_real_distribution>(5*M, N, -1, 1))
                    .addLayer(Layer::LayerFactory::CreateLayer<std::uniform_real_distribution>(M, 5*M, -1, 1))
                    .networkLayers();

    // Train MNIST data (60000 x 784)
    Matrix<Type>& Xtrain = Matrix<Type>::MatrixFactory::CreateMatrix(60000, N);
    Matrix<Type>& Ytrain = Matrix<Type>::MatrixFactory::CreateMatrix(60000, M);
    
    // Test MNIST data (10000 x 784)
    Matrix<Type>& Xtest = Matrix<Type>::MatrixFactory::CreateMatrix(10000, N);
    Matrix<Type>& Ytest = Matrix<Type>::MatrixFactory::CreateMatrix(10000, M);

    // Load MNIST train and test data
    LoadData(MNISTData::g_mnist_train_data, Xtrain, Ytrain);
    LoadData(MNISTData::g_mnist_test_data, Xtest, Ytest);

    // Train data
    TIME_IT_MS(n.train(Xtrain, Ytrain, -0.1, 128, 25, false));

    // Prediction test
    std::cout << "Prediction accuracy: " << n.accuracy(tuple, Xtest, Ytest) << "%\n";
}


int main(int argc, char** argv) {
    #ifndef USE_COMPLEX_MATH
        MNISTPrediction();
    #endif
    return 0;
}
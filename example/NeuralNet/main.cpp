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

// Distribution
template<typename T>
using UniformDistribution = std::uniform_real_distribution<T>;
template<typename T>
using NormalDistribution = std::normal_distribution<T>;


void MNISTPrediction() {
    const size_t N{784}, M{10};

    CustomNet n({N,1},{M,1});

    // Generate network layers
    auto tuple = n.addLayer(Layer::LayerFactory::CreateLayer<UniformDistribution>(5*M, N, -1, 1))
                    .addLayer(Layer::LayerFactory::CreateLayer<UniformDistribution>(M, 5*M, -1, 1))
                    .networkLayers();

    // Train MNIST data (60000 x 784)
    Matrix<Type>& Xtrain = Matrix<Type>::MatrixFactory::CreateMatrix(60000, N);
    Matrix<Type>& Ytrain = Matrix<Type>::MatrixFactory::CreateMatrix(60000, M);
    
    // Test MNIST data (10000 x 784)
    Matrix<Type>& Xtest = Matrix<Type>::MatrixFactory::CreateMatrix(10000, N);
    Matrix<Type>& Ytest = Matrix<Type>::MatrixFactory::CreateMatrix(10000, M);

    // Load MNIST train and test data
    LoadData(MNISTData::g_mnist_train_data_path, Xtrain, Ytrain);
    LoadData(MNISTData::g_mnist_test_data_path, Xtest, Ytest);

    // Train data
    TIME_IT_MS(n.train(Xtrain, Ytrain, -0.1, 64, 25, false));

    // Prediction test
    std::cout << "Prediction accuracy: " << n.accuracy(tuple, Xtest, Ytest) << "%\n";
}

int main(int argc, char** argv) {

    Matrix<Variable> X(2,2), W(2,2); 
    X(0, 0) = 1; X(0, 1) = 2;
    X(1, 0) = 3; X(1, 1) = 4;

    for(size_t i{}; i < 2; ++i) {
        for(size_t j{}; j < 2; ++j) {
            W(i,j) = i+j;
        }
    }

    Matrix<Expression> Y = transpose(vec(W*X))*(vec(W*X)); 

    std::cout << CoolDiff::TensorR2::Eval(Y) << "\n";
    std::cout << CoolDiff::TensorR2::DevalF(Y, X) << "\n";

    X(0, 0) = 4; X(0, 1) = 3;
    X(1, 0) = 2; X(1, 1) = 1;
    
    std::cout << CoolDiff::TensorR2::Eval(Y) << "\n";
    std::cout << CoolDiff::TensorR2::DevalF(Y, X) << "\n";


    #ifndef USE_COMPLEX_MATH
        MNISTPrediction();
    #endif
    return 0;
}
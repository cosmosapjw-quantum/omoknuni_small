#include <iostream>
#include <torch/torch.h>
#include "nn/neural_network_factory.h"
#include "nn/resnet_model.h"

int main() {
    try {
        std::cout << "Creating model..." << std::endl;
        auto model = alphazero::nn::NeuralNetworkFactory::createResNet(19, 9, 2, 32, 81);
        
        std::cout << "Initializing weights..." << std::endl;
        for (auto& p : model->parameters()) {
            p.data().normal_(0.0, 0.1);
        }
        
        std::cout << "First param before save: " << model->parameters()[0].data_ptr<float>()[0] << std::endl;
        
        std::cout << "Saving model..." << std::endl;
        model->save("/tmp/test_minimal.pt");
        
        std::cout << "Creating new model..." << std::endl;
        auto model2 = alphazero::nn::NeuralNetworkFactory::createResNet(19, 9, 2, 32, 81);
        
        std::cout << "Loading weights..." << std::endl;
        model2->load("/tmp/test_minimal.pt");
        
        std::cout << "First param after load: " << model2->parameters()[0].data_ptr<float>()[0] << std::endl;
        
        std::cout << "Success!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
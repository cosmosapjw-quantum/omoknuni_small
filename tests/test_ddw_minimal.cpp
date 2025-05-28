#include <iostream>
#include <torch/torch.h>

int main() {
    std::cout << "Minimal DDW test\n" << std::endl;
    
    // Test 1: Basic Torch functionality
    std::cout << "Test 1: Torch availability" << std::endl;
    std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    
    // Test 2: Create simple tensor
    std::cout << "\nTest 2: Tensor creation" << std::endl;
    torch::Tensor x = torch::randn({2, 3}, torch::kCPU);
    std::cout << "Tensor shape: " << x.sizes() << std::endl;
    
    // Test 3: Simple module
    std::cout << "\nTest 3: Simple module creation" << std::endl;
    auto linear = torch::nn::Linear(3, 2);
    auto y = linear(x);
    std::cout << "Output shape: " << y.sizes() << std::endl;
    
    std::cout << "\nBasic tests passed!" << std::endl;
    return 0;
}
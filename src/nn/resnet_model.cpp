// src/nn/resnet_model.cpp
#include "nn/resnet_model.h"
#include <stdexcept>

namespace alphazero {
namespace nn {

ResNetResidualBlock::ResNetResidualBlock(int64_t channels) {
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                             .padding(1).bias(false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels));
    
    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                             .padding(1).bias(false));
    bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels));
    
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
}

torch::Tensor ResNetResidualBlock::forward(torch::Tensor x) {
    torch::Tensor residual = x;
    x = torch::relu(bn1(conv1(x)));
    x = bn2(conv2(x));
    x = torch::relu(x + residual);
    return x;
}

ResNetModel::ResNetModel(int64_t input_channels, int64_t board_size,
                       int64_t num_res_blocks, int64_t num_filters,
                       int64_t policy_size)
    : input_channels_(input_channels),
      board_size_(board_size),
      policy_size_(policy_size) {
    
    // If policy size not specified, default to board_size^2
    if (policy_size_ == 0) {
        policy_size_ = board_size_ * board_size_;
    }
    
    // Input layers
    // Always create input layer with the actual channel count needed (17 for Gomoku)
    input_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels_, num_filters, 3)
                                  .padding(1).bias(false));
    input_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_filters));
    
    register_module("input_conv", input_conv_);
    register_module("input_bn", input_bn_);
    
    // Residual blocks
    res_blocks_ = torch::nn::ModuleList();
    for (int64_t i = 0; i < num_res_blocks; ++i) {
        res_blocks_->push_back(std::make_shared<ResNetResidualBlock>(num_filters));
        register_module("res_block_" + std::to_string(i), res_blocks_[i]);
    }
    
    // Policy head
    policy_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 32, 1).bias(false));
    policy_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
    policy_fc_ = torch::nn::Linear(32 * board_size_ * board_size_, policy_size_);
    
    register_module("policy_conv", policy_conv_);
    register_module("policy_bn", policy_bn_);
    register_module("policy_fc", policy_fc_);
    
    // Value head
    value_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 32, 1).bias(false));
    value_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
    value_fc1_ = torch::nn::Linear(32 * board_size_ * board_size_, 256);
    value_fc2_ = torch::nn::Linear(256, 1);
    
    register_module("value_conv", value_conv_);
    register_module("value_bn", value_bn_);
    register_module("value_fc1", value_fc1_);
    register_module("value_fc2", value_fc2_);
}

std::tuple<torch::Tensor, torch::Tensor> ResNetModel::forward(torch::Tensor x) {
    // Common layers
    x = torch::relu(input_bn_(input_conv_(x)));
    
    // Residual blocks
    for (const auto& block : *res_blocks_) {
        x = block->as<ResNetResidualBlock>()->forward(x);
    }
    
    // Policy head
    torch::Tensor policy = torch::relu(policy_bn_(policy_conv_(x)));
    policy = policy.view({policy.size(0), -1});
    policy = torch::log_softmax(policy_fc_(policy), 1);
    
    // Value head
    torch::Tensor value = torch::relu(value_bn_(value_conv_(x)));
    value = value.view({value.size(0), -1});
    value = torch::relu(value_fc1_(value));
    value = torch::tanh(value_fc2_(value));
    
    return {policy, value};
}

torch::Tensor ResNetModel::prepareInputTensor(const std::vector<std::unique_ptr<core::IGameState>>& states) {
    if (states.empty()) {
        return torch::Tensor();
    }
    
    std::vector<torch::Tensor> batch_tensors;
    batch_tensors.reserve(states.size());
    
    for (const auto& state : states) {
        // Get tensor representation based on input channels
        std::vector<std::vector<std::vector<float>>> tensor;
        
        // Use regular tensor representation for games expecting 3 channels
        if (input_channels_ == 3) {
            tensor = state->getTensorRepresentation();
        } else {
            // Use enhanced tensor for games expecting more channels
            tensor = state->getEnhancedTensorRepresentation();
        }
        
        // Convert to torch tensor
        std::vector<int64_t> dims = {static_cast<int64_t>(tensor.size()),
                                     static_cast<int64_t>(tensor[0].size()),
                                     static_cast<int64_t>(tensor[0][0].size())};
        
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor t = torch::zeros(dims, options);
        
        // Copy data
        for (size_t c = 0; c < tensor.size(); ++c) {
            for (size_t h = 0; h < tensor[c].size(); ++h) {
                for (size_t w = 0; w < tensor[c][h].size(); ++w) {
                    t[c][h][w] = tensor[c][h][w];
                }
            }
        }
        
        batch_tensors.push_back(t);
    }
    
    // Stack tensors into a batch
    return torch::stack(batch_tensors);
}

std::vector<mcts::NetworkOutput> ResNetModel::inference(
    const std::vector<std::unique_ptr<core::IGameState>>& states) {
    
    std::vector<mcts::NetworkOutput> outputs;
    if (states.empty()) {
        return outputs;
    }
    
    // Prepare input tensor
    torch::Tensor input = prepareInputTensor(states);
    
    // Use eval mode
    this->eval();
    
    // Run inference
    torch::NoGradGuard no_grad;
    auto device = this->parameters().begin()->device();
    input = input.to(device);
    
    auto [policy, value] = this->forward(input);
    
    // Extract policy and value
    policy = torch::softmax(policy, 1);
    
    // Move back to CPU if needed
    if (policy.device().is_cuda()) {
        policy = policy.to(torch::kCPU);
        value = value.to(torch::kCPU);
    }
    
    // Convert to NetworkOutput
    auto policy_accessor = policy.accessor<float, 2>();
    auto value_accessor = value.accessor<float, 2>();
    
    outputs.reserve(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
        mcts::NetworkOutput output;
        
        // Extract policy
        output.policy.resize(policy_size_);
        for (int64_t j = 0; j < policy_size_; ++j) {
            output.policy[j] = policy_accessor[i][j];
        }
        
        // Extract value
        output.value = value_accessor[i][0];
        
        outputs.push_back(std::move(output));
    }
    
    return outputs;
}

void ResNetModel::save(const std::string& path) {
    auto self = shared_from_this();
    torch::save(self, path);
}

void ResNetModel::load(const std::string& path) {
    auto self = shared_from_this();
    torch::load(self, path);
}

std::vector<int64_t> ResNetModel::getInputShape() const {
    return {input_channels_, board_size_, board_size_};
}

int64_t ResNetModel::getPolicySize() const {
    return policy_size_;
}

} // namespace nn
} // namespace alphazero
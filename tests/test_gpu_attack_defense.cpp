#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "utils/attack_defense_module.h"
#include "utils/gpu_attack_defense_module.h"

using namespace alphazero;

void benchmark_attack_defense() {
    const int board_size = 15;
    const int batch_size = 32;
    const int num_iterations = 100;
    
    // Create random board positions
    std::vector<std::vector<std::vector<int>>> board_batch(batch_size, 
        std::vector<std::vector<int>>(board_size, std::vector<int>(board_size, 0)));
    
    // Fill with some random stones
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> player_dist(0, 2);
    std::uniform_int_distribution<int> pos_dist(0, board_size - 1);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < 30; ++i) {  // Place ~30 stones per board
            int row = pos_dist(rng);
            int col = pos_dist(rng);
            board_batch[b][row][col] = player_dist(rng);
        }
    }
    
    // Create moves to evaluate
    std::vector<int> chosen_moves(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        // Find an empty position
        for (int pos = 0; pos < board_size * board_size; ++pos) {
            int row = pos / board_size;
            int col = pos % board_size;
            if (board_batch[b][row][col] == 0) {
                chosen_moves[b] = pos;
                break;
            }
        }
    }
    
    std::vector<int> player_batch(batch_size, 1);
    
    // Benchmark CPU version
    auto cpu_module = std::make_unique<GomokuAttackDefenseModule>(board_size);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        auto [attack, defense] = cpu_module->compute_bonuses(board_batch, chosen_moves, player_batch);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    
    std::cout << "CPU Attack/Defense computation:" << std::endl;
    std::cout << "  Total time: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "  Time per batch: " << cpu_duration.count() / (float)num_iterations << " ms" << std::endl;
    std::cout << "  Throughput: " << (batch_size * num_iterations * 1000.0f) / cpu_duration.count() 
              << " boards/sec" << std::endl;
    
    // Benchmark GPU version if available
    if (torch::cuda::is_available()) {
        torch::Device device(torch::kCUDA);
        auto gpu_module = std::make_unique<GomokuGPUAttackDefense>(board_size, device);
        
        // Convert boards to tensor
        auto board_tensor = torch::zeros({batch_size, board_size, board_size}, 
                                        torch::TensorOptions().dtype(torch::kInt32).device(device));
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < board_size; ++i) {
                for (int j = 0; j < board_size; ++j) {
                    board_tensor[b][i][j] = board_batch[b][i][j];
                }
            }
        }
        
        auto moves_tensor = torch::tensor(chosen_moves, 
                                         torch::TensorOptions().dtype(torch::kInt32).device(device));
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            auto [attack, defense] = gpu_module->compute_bonuses_gpu(board_tensor, moves_tensor, 1);
        }
        torch::cuda::synchronize();
        
        auto gpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            auto [attack, defense] = gpu_module->compute_bonuses_gpu(board_tensor, moves_tensor, 1);
        }
        torch::cuda::synchronize();
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
        
        std::cout << "\nGPU Attack/Defense computation:" << std::endl;
        std::cout << "  Total time: " << gpu_duration.count() << " ms" << std::endl;
        std::cout << "  Time per batch: " << gpu_duration.count() / (float)num_iterations << " ms" << std::endl;
        std::cout << "  Throughput: " << (batch_size * num_iterations * 1000.0f) / gpu_duration.count() 
                  << " boards/sec" << std::endl;
        std::cout << "  Speedup: " << (float)cpu_duration.count() / gpu_duration.count() << "x" << std::endl;
    } else {
        std::cout << "\nGPU not available for benchmarking" << std::endl;
    }
}

int main() {
    std::cout << "Attack/Defense Module Benchmark" << std::endl;
    std::cout << "===============================" << std::endl;
    
    benchmark_attack_defense();
    
    return 0;
}
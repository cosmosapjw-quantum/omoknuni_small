#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <iomanip>
#include <thread>
#include <omp.h>
#include "mcts/mcts_engine.h"
#include "nn/neural_network.h"
#include "games/gomoku/gomoku_state.h"
#include "utils/debug_monitor.h"

using namespace alphazero;

// Mock neural network for benchmarking
class MockNeuralNetworkBench : public nn::NeuralNetwork {
public:
    std::vector<alphazero::mcts::NetworkOutput> inference(
        const std::vector<std::unique_ptr<core::IGameState>>& states) override {
        
        std::vector<alphazero::mcts::NetworkOutput> outputs;
        for (size_t i = 0; i < states.size(); i++) {
            alphazero::mcts::NetworkOutput output;
            
            // Create dummy policy
            int num_actions = 361; // 19x19 board
            output.policy.resize(num_actions);
            float sum = 0.0f;
            
            // Simple policy: prefer center moves
            for (int j = 0; j < num_actions; j++) {
                int x = j % 19;
                int y = j / 19;
                float dist = std::sqrt((x - 9) * (x - 9) + (y - 9) * (y - 9));
                output.policy[j] = std::exp(-dist / 5.0f);
                sum += output.policy[j];
            }
            
            // Normalize
            for (int j = 0; j < num_actions; j++) {
                output.policy[j] /= sum;
            }
            
            // Dummy value
            output.value = 0.1f;
            
            outputs.push_back(output);
        }
        
        // Simulate GPU processing time
        std::this_thread::sleep_for(std::chrono::microseconds(10 * states.size()));
        
        return outputs;
    }
    
    // Required virtual methods
    void save(const std::string& path) override {}
    void load(const std::string& path) override {}
    std::vector<int64_t> getInputShape() const override { return {19, 19, 17}; }
    int64_t getPolicySize() const override { return 361; }
};

// Simple benchmark wrapper
class MCTSBenchmark {
public:
    MCTSBenchmark() {
        // Create mock neural network for benchmarking
        neural_net_ = std::make_shared<MockNeuralNetworkBench>();
    }
    
    void runBenchmark(const std::string& test_name, int board_size, int num_sims, int num_threads) {
        // Configure MCTS settings
        alphazero::mcts::MCTSSettings settings;
        settings.num_simulations = num_sims;
        settings.num_threads = num_threads;
        settings.batch_size = 32;
        settings.virtual_loss = 3;
        settings.exploration_constant = 1.4f;
        settings.add_dirichlet_noise = false;  // Disable for consistency
        
        // Set OMP threads
        omp_set_num_threads(num_threads);
        
        // Create MCTS engine
        alphazero::mcts::MCTSEngine engine(neural_net_, settings);
        
        // Create initial game state
        auto game_state = std::make_unique<games::gomoku::GomokuState>(board_size);
        
        // Make a few moves to create more interesting position
        std::vector<int> initial_moves = {180, 181, 162, 163, 144};  // Center moves
        for (int move : initial_moves) {
            if (move < board_size * board_size) {
                game_state->makeMove(move);
            }
        }
        
        std::cout << "\n=== Benchmark: " << test_name << " ===\n";
        std::cout << "Board size: " << board_size << "x" << board_size << "\n";
        std::cout << "Simulations: " << num_sims << "\n";
        std::cout << "Threads: " << num_threads << "\n";
        std::cout << "OMP max threads: " << omp_get_max_threads() << "\n";
        
        // Warm-up run
        std::cout << "Warming up...\n";
        engine.search(*game_state);
        
        // Actual benchmark runs
        const int num_runs = 5;
        std::vector<double> run_times;
        std::vector<double> cpu_usages;
        
        for (int run = 0; run < num_runs; run++) {
            // Start monitoring
            debug::SystemMonitor::instance().start(100);
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Run MCTS search
            auto result = engine.search(*game_state);
            
            auto end = std::chrono::high_resolution_clock::now();
            
            // Stop monitoring
            debug::SystemMonitor::instance().stop();
            
            // Calculate time
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            double time_ms = duration.count();
            run_times.push_back(time_ms);
            
            // Get average CPU usage from the last run
            double cpu_usage = debug::SystemMonitor::instance().getCpuUsage();
            cpu_usages.push_back(cpu_usage);
            
            // Print run results
            std::cout << "Run " << (run + 1) << ": " 
                     << std::fixed << std::setprecision(1) << time_ms << "ms"
                     << " (" << (num_sims * 1000.0 / time_ms) << " sims/sec)"
                     << " CPU: " << cpu_usage << "%\n";
            
            // Print action probabilities for consistency check
            if (run == 0) {
                std::cout << "Top 5 moves: ";
                std::vector<std::pair<int, float>> move_probs;
                for (int i = 0; i < result.probabilities.size(); i++) {
                    if (result.probabilities[i] > 0) {
                        move_probs.push_back({i, result.probabilities[i]});
                    }
                }
                std::sort(move_probs.begin(), move_probs.end(), 
                    [](const auto& a, const auto& b) { return a.second > b.second; });
                
                for (int i = 0; i < std::min(5, (int)move_probs.size()); i++) {
                    std::cout << move_probs[i].first << "(" 
                             << std::setprecision(3) << move_probs[i].second << ") ";
                }
                std::cout << "\n";
            }
        }
        
        // Calculate statistics
        double avg_time = 0;
        double min_time = run_times[0];
        double max_time = run_times[0];
        for (double t : run_times) {
            avg_time += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        avg_time /= num_runs;
        
        double avg_cpu = 0;
        for (double c : cpu_usages) {
            avg_cpu += c;
        }
        avg_cpu /= num_runs;
        
        // Print summary
        std::cout << "\nSummary:\n";
        std::cout << "Average time: " << std::fixed << std::setprecision(1) 
                 << avg_time << "ms\n";
        std::cout << "Min time: " << min_time << "ms\n";
        std::cout << "Max time: " << max_time << "ms\n";
        std::cout << "Average throughput: " << std::setprecision(0) 
                 << (num_sims * 1000.0 / avg_time) << " sims/sec\n";
        std::cout << "Average CPU usage: " << std::setprecision(1) << avg_cpu << "%\n";
        std::cout << "CPU efficiency: " << std::setprecision(1) 
                 << (avg_cpu / (num_threads * 100.0)) * 100 << "%\n";
    }
    
private:
    std::shared_ptr<nn::NeuralNetwork> neural_net_;
};

int main(int /* argc */, char* /* argv */[]) {
    std::cout << "MCTS OpenMP Benchmark\n";
    std::cout << "System threads: " << std::thread::hardware_concurrency() << "\n";
    std::cout << "OpenMP version: " << _OPENMP << "\n\n";
    
    MCTSBenchmark benchmark;
    
    // Test different thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 20, 24};
    
    for (int threads : thread_counts) {
        benchmark.runBenchmark(
            "Gomoku 19x19 - " + std::to_string(threads) + " threads",
            19,      // board size
            800,     // simulations
            threads  // threads
        );
        
        // Brief pause between tests
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    
    // Test scalability with different simulation counts
    std::cout << "\n\n=== Scalability Test (20 threads) ===\n";
    std::vector<int> sim_counts = {100, 200, 400, 800, 1600};
    
    for (int sims : sim_counts) {
        benchmark.runBenchmark(
            "Gomoku 19x19 - " + std::to_string(sims) + " sims",
            19,   // board size
            sims, // simulations
            20    // threads
        );
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    return 0;
}
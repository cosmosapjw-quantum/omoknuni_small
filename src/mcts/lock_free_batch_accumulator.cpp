#include "mcts/lock_free_batch_accumulator.h"
#include "mcts/mcts_engine.h"
#include <algorithm>
#include <chrono>

namespace alphazero {
namespace mcts {

LockFreeBatchAccumulator::LockFreeBatchAccumulator(const LockFreeBatchConfig& config)
    : config_(config),
      shutdown_(false),
      accumulation_thread_(&LockFreeBatchAccumulator::accumulationLoop, this) {
}

LockFreeBatchAccumulator::~LockFreeBatchAccumulator() {
    shutdown();
}

void LockFreeBatchAccumulator::shutdown() {
    if (!shutdown_.exchange(true)) {
        if (accumulation_thread_.joinable()) {
            accumulation_thread_.join();
        }
    }
}

void LockFreeBatchAccumulator::submitRequest(PendingEvaluation&& request) {
    pending_queue_.enqueue(std::move(request));
    pending_count_.fetch_add(1);
}

std::vector<PendingEvaluation> LockFreeBatchAccumulator::collectBatch() {
    std::vector<PendingEvaluation> batch;
    batch.reserve(config_.target_batch_size);
    
    PendingEvaluation eval;
    while (batch.size() < config_.target_batch_size && ready_queue_.try_dequeue(eval)) {
        batch.push_back(std::move(eval));
    }
    
    return batch;
}

void LockFreeBatchAccumulator::accumulationLoop() {
    std::vector<PendingEvaluation> current_batch;
    current_batch.reserve(config_.target_batch_size);
    
    auto last_batch_time = std::chrono::steady_clock::now();
    
    while (!shutdown_.load()) {
        PendingEvaluation eval;
        
        if (pending_queue_.try_dequeue(eval)) {
            current_batch.push_back(std::move(eval));
            pending_count_.fetch_sub(1);
            
            if (current_batch.size() >= config_.target_batch_size) {
                flushBatch(std::move(current_batch));
                current_batch.clear();
                current_batch.reserve(config_.target_batch_size);
                last_batch_time = std::chrono::steady_clock::now();
            }
        } else {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = now - last_batch_time;
            
            if (!current_batch.empty() && elapsed >= config_.max_wait_time) {
                flushBatch(std::move(current_batch));
                current_batch.clear();
                current_batch.reserve(config_.target_batch_size);
                last_batch_time = now;
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    }
    
    if (!current_batch.empty()) {
        flushBatch(std::move(current_batch));
    }
}

void LockFreeBatchAccumulator::flushBatch(std::vector<PendingEvaluation>&& batch) {
    for (auto& eval : batch) {
        ready_queue_.enqueue(std::move(eval));
    }
    ready_count_.fetch_add(batch.size());
}

size_t LockFreeBatchAccumulator::pendingCount() const {
    return pending_count_.load();
}

size_t LockFreeBatchAccumulator::readyCount() const {
    return ready_count_.load();
}

} // namespace mcts
} // namespace alphazero

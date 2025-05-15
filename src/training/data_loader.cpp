#include "training/data_loader.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace alphazero {
namespace training {

DataLoader::DataLoader(
    std::shared_ptr<Dataset> dataset,
    size_t batch_size,
    bool shuffle,
    size_t num_workers,
    bool pin_memory,
    bool drop_last)
    : dataset_(dataset),
      batch_size_(batch_size),
      shuffle_(shuffle),
      num_workers_(num_workers),
      pin_memory_(pin_memory),
      drop_last_(drop_last),
      current_index_(0),
      prepared_(false),
      stop_workers_(false),
      next_batch_index_(0)
{
    if (!dataset_) {
        throw std::invalid_argument("Dataset cannot be null");
    }
    
    if (batch_size_ == 0) {
        throw std::invalid_argument("Batch size must be greater than zero");
    }
    
    // Calculate number of batches
    size_t dataset_size = dataset_->size();
    num_batches_ = dataset_size / batch_size_;
    if (!drop_last_ && dataset_size % batch_size_ != 0) {
        num_batches_++;
    }
    
    // Start worker threads if requested
    if (num_workers_ > 0) {
        start_workers();
    }
    
    std::cout << "DataLoader created with " << dataset_size << " examples, "
              << num_batches_ << " batches, batch size " << batch_size_
              << ", " << num_workers_ << " workers" << std::endl;
}

DataLoader::~DataLoader() {
    // Stop worker threads if they were started
    if (num_workers_ > 0) {
        stop_workers();
    }
}

DataLoader::Iterator DataLoader::begin() {
    // Prepare dataset for iteration
    if (!prepared_) {
        prepare();
    }
    
    // Reset index
    current_index_ = 0;
    
    // Start prefetching batches if using workers
    if (num_workers_ > 0) {
        next_batch_index_.store(0);
    }
    
    return Iterator(this, 0);
}

DataLoader::Iterator DataLoader::end() {
    return Iterator(this, num_batches_);
}

size_t DataLoader::size() const {
    return num_batches_;
}

size_t DataLoader::batch_size() const {
    return batch_size_;
}

void DataLoader::reset() {
    // Reset index
    current_index_ = 0;
    
    // Reset prefetching if using workers
    if (num_workers_ > 0) {
        next_batch_index_.store(0);
    }
    
    // Mark as not prepared
    prepared_ = false;
}

void DataLoader::prepare() {
    // Shuffle dataset if requested
    if (shuffle_) {
        dataset_->shuffle();
    }
    
    prepared_ = true;
}

Batch DataLoader::load_batch(size_t batch_index) {
    size_t dataset_size = dataset_->size();
    
    // Calculate start and end indices
    size_t start_idx = batch_index * batch_size_;
    size_t end_idx = std::min(start_idx + batch_size_, dataset_size);
    size_t actual_batch_size = end_idx - start_idx;
    
    // Create vectors to store tensors
    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> policies;
    std::vector<torch::Tensor> values;
    
    states.reserve(actual_batch_size);
    policies.reserve(actual_batch_size);
    values.reserve(actual_batch_size);
    
    // Load data
    for (size_t i = start_idx; i < end_idx; i++) {
        auto [state, policy, value] = dataset_->get(i);
        states.push_back(state);
        policies.push_back(policy);
        values.push_back(value);
    }
    
    // Stack tensors
    Batch batch;
    batch.states = torch::stack(states);
    batch.policies = torch::stack(policies);
    batch.values = torch::stack(values);
    
    // Pin memory if requested
    if (pin_memory_ && !batch.states.is_pinned()) {
        batch.states = batch.states.pin_memory();
        batch.policies = batch.policies.pin_memory();
        batch.values = batch.values.pin_memory();
    }
    
    return batch;
}

void DataLoader::start_workers() {
    std::cout << "Starting " << num_workers_ << " worker threads" << std::endl;
    
    // Initialize stop flag
    stop_workers_.store(false);
    
    // Clear batch queue
    std::queue<std::future<Batch>> empty_queue;
    std::swap(batch_queue_, empty_queue);
    
    // Start worker threads
    workers_.reserve(num_workers_);
    for (size_t i = 0; i < num_workers_; i++) {
        workers_.emplace_back(&DataLoader::worker_function, this);
    }
}

void DataLoader::stop_workers() {
    std::cout << "Stopping worker threads" << std::endl;
    
    // Set stop flag
    stop_workers_.store(true);
    
    // Notify workers
    queue_cv_.notify_all();
    
    // Join worker threads
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    // Clear worker thread vector
    workers_.clear();
}

void DataLoader::worker_function() {
    while (!stop_workers_.load()) {
        // Get next batch index to process
        size_t batch_index = next_batch_index_.fetch_add(1);
        
        // Check if we've processed all batches
        if (batch_index >= num_batches_) {
            // Sleep a bit and check again
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Create a promise for the batch
        std::promise<Batch> batch_promise;
        auto batch_future = batch_promise.get_future();
        
        // Add future to queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            batch_queue_.push(std::move(batch_future));
        }
        
        // Notify that a batch is ready
        queue_cv_.notify_one();
        
        try {
            // Load batch
            Batch batch = load_batch(batch_index);
            
            // Set promise value
            batch_promise.set_value(std::move(batch));
        } catch (const std::exception& e) {
            // Set promise exception
            batch_promise.set_exception(std::current_exception());
        }
    }
}

// Iterator implementation
DataLoader::Iterator::Iterator(DataLoader* loader, size_t index)
    : loader_(loader), index_(index) {}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    ++index_;
    return *this;
}

Batch DataLoader::Iterator::operator*() {
    if (index_ >= loader_->num_batches_) {
        throw std::out_of_range("Iterator out of range");
    }
    
    if (loader_->num_workers_ > 0) {
        // Multi-threaded loading
        std::unique_lock<std::mutex> lock(loader_->queue_mutex_);
        
        // Wait for a batch to be ready
        while (loader_->batch_queue_.empty()) {
            loader_->queue_cv_.wait(lock);
        }
        
        // Get future from queue
        auto batch_future = std::move(loader_->batch_queue_.front());
        loader_->batch_queue_.pop();
        lock.unlock();
        
        // Wait for batch and return it
        return batch_future.get();
    } else {
        // Single-threaded loading
        return loader_->load_batch(index_);
    }
}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return index_ != other.index_;
}

} // namespace training
} // namespace alphazero
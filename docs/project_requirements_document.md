# Project Requirements Document (PRD)

**Project Name**: Omoknuni\
**Project Type**: C++ AlphaZero-Style Multi-Game AI Engine with Python CLI

## 1. Project Overview

Omoknuni is a high-performance AI engine written in C++ that learns to play Gomoku, Chess, and Go at an expert level without relying on handcrafted heuristics. At its core, it combines Monte Carlo Tree Search (MCTS) with a DDW-RandWire-ResNet deep neural network for position evaluation. By using a modular game abstraction layer, Omoknuni allows new board games to be plugged in with minimal changes to the search or neural network code. A Python package with a command-line interface (CLI) wraps the C++ core, making it easy for game developers to run self-play, train models, evaluate performance, and play interactively.

The engine will be delivered under an MIT open-source license, targeting a single-desktop GPU environment (e.g., NVIDIA RTX 3060 Ti on a Ryzen 5900X, 64 GB RAM) on Linux and Windows (64-bit). Success is measured by achieving stable leaf-parallelized MCTS with batched GPU inference, robust self-play data generation, repeatable ELO improvements, and clear CLI-driven workflows, all without performance regressions or concurrency bugs.

## 2. In-Scope vs. Out-of-Scope

### In-Scope (Version 1.0)

*   **Game Abstraction Layer** for Gomoku, Chess, and Go\
    – C++ interfaces: state representation, move generation, application, outcome detection\
    – Zobrist hashing for fast transposition lookups
*   **Advanced MCTS Engine**\
    – Multi-threaded playouts with virtual loss for collision prevention\
    – Progressive widening to control tree branching factor (C and K parameters)\
    – RAVE (Rapid Action Value Estimation) with configurable weighting\
    – Root parallelization: multiple independent MCTS trees running in parallel\
    – Leaf-parallelization: workers submit leaf states to a centralized batch evaluator\
    – Lock-free concurrent queues (moodycamel) for efficient communication\
    – Transposition table with configurable size and sharding
*   **Neural Network Integration**\
    – DDW-RandWire-ResNet policy/value heads via libtorch (CUDA)\
    – Central batch inference thread with configurable batch size and timeout\
    – Support for external queue integration for shared evaluator
*   **Python CLI & Bindings**\
    – `omoknuni-cli` for self-play, train, eval, play commands via pybind11
*   **Self-Play & Training Pipeline**\
    – Flat-file storage (JSON or binary) for game records, ELO logs, and model checkpoints\
    – Regular text/log output of statistics: ELO progression, GPU/CPU utilization, batch latencies\
    – Temperature scheduling and Dirichlet noise for exploration
*   **Flat-File Logging & Reporting**\
    – spdlog (or similar) for structured text logs; daily summary reports

### Out-of-Scope (Later Phases)

*   Graphical or web-based UIs for replay or real-time visualization
*   Multi-GPU or distributed training/inference
*   Database integrations (SQL/NoSQL) or cloud services
*   User roles or permission systems
*   External monitoring dashboards (beyond text/log files)

## 3. User Flow

A game developer clones the Omoknuni GitHub repository and runs the CMake build system to compile the C++ core library and executables. Next, they install the Python package (which wraps the C++ shared library) via `pip install -e .`. Environment variables—`OMOKNUNI_MODEL_DIR` and `OMOKNUNI_DATA_DIR`—are set to designate where model checkpoints and game records will live. A YAML config file is created to specify the target game (e.g., Gomoku), neural network hyperparameters (batch size, learning rate), and MCTS settings (number of simulations, virtual loss factor).

With configuration in place, the developer runs `omoknuni-cli self-play --config config.yaml`. Worker threads launch MCTS searches, leaf states are queued for batch evaluation by a dedicated NN thread, and completed game records and ELO updates are written to flat files. Once enough data is gathered, `omoknuni-cli train --config config.yaml` ingests these records to update the network weights via stochastic gradient descent, periodically checkpointing to disk and emitting training metrics to logs. The `eval` command lets them pit a new model against the current champion to measure ELO gains, and `play` opens an interactive CLI where the human and AI alternate moves, all backed by the same MCTS + batch inference machinery.

## 4. Core Features

*   **Modular Game Abstraction Layer**\
    • Unified C++ interface for any turn-based, perfect-information board game\
    • Zobrist hashing for efficient transposition table keys
*   **AlphaZero-Style MCTS Engine**\
    • Multi-threaded playouts with virtual losses to avoid thread collisions\
    • Progressive widening with configurable C and K parameters for controlled tree growth\
    • Transposition tables with sharded design for efficient position caching\
    • RAVE (Rapid Action Value Estimation) for improved action value estimates\
    • Root parallelization option to run multiple independent MCTS trees\
    • Lock-free concurrent queues (moodycamel) for producer-consumer patterns
*   **Leaf-Parallelization & Centralized Batch Evaluator**\
    • Worker threads produce leaf evaluation requests to a lock-free queue\
    • Single evaluator thread batches states (up to batch_size or timeout) and runs GPU inference\
    • Results returned via concurrent queues or external shared queues\
    • Adaptive batch collection with configurable timeout strategies
*   **Neural Network: DDW-RandWire-ResNet**\
    • 40-node random DAG backbone, C=144 channels, SE-style edge gating\
    • CoordConv input, policy head (n²+1 logits), value head (tanh output)
*   **Python Bindings & CLI**\
    • pybind11 exposes core commands: self-play, train, eval, play\
    • Configurable via YAML: game choice, MCTS & NN hyperparameters, file paths
*   **Self-Play Manager & ELO Tracking**\
    • Temperature scheduling, checkpoint rotation, flat-file game record storage\
    • ELO rating updates appended to text logs for easy plotting
*   **Logging & Reporting**\
    • spdlog-powered structured logs: timestamps, GPU memory, batch latency, node stats\
    • Daily and per-session summaries in plain text

## 5. Tech Stack & Tools

*   **Languages & Frameworks**\
    • C++17 for core engine and MCTS concurrency\
    • Python 3.x for CLI and orchestrator\
    • pybind11 for C++↔Python bindings
*   **Neural Network**\
    • libtorch (PyTorch C++ API) with CUDA acceleration\
    • (Optional) ONNX Runtime C++ API for model export and inference
*   **Build & Packaging**\
    • CMake for cross-platform build (Linux & Windows x64)\
    • pip / setuptools for Python package
*   **Concurrency & Queues**\
    • std::thread, std::mutex, std::condition_variable, std::atomic\
    • moodycamel::ConcurrentQueue (lock-free producer/consumer)
*   **Logging**\
    • spdlog (fast C++ logging)
*   **IDE & Plugins**\
    • Cursor (AI-powered C++ suggestion in VS Code / CLion)

## 6. Non-Functional Requirements

*   **Performance**\
    • Batch inference latency ≤ 30 ms for 256 states on RTX 3060 Ti\
    • MCTS simulation throughput ≥ 10,000 playouts/sec total with 12 threads\
    • Queue operations < 1 microsecond per enqueue/dequeue
*   **Scalability**\
    • Up to 16 worker threads feeding a single evaluator thread
*   **Reliability & Safety**\
    • Thread-safe tree updates (atomic counters or fine-grained locks)\
    • No deadlocks or data races (verify with sanitizers)
*   **Usability**\
    • Clear CLI UX with descriptive help messages\
    • Human-readable logs and daily summary reports
*   **Portability**\
    • Support Linux and Windows (64-bit) with a single CMake script
*   **Maintainability**\
    • Clean module boundaries: Game layer, MCTS core, NN evaluator, Python CLI\
    • Well-documented public APIs

## 7. Constraints & Assumptions

*   **Hardware**: Single desktop GPU (NVIDIA RTX 3060 Ti), Ryzen 5900X CPU, 64 GB RAM
*   **Software**: Requires libtorch with CUDA support; Python 3.x; C++17 compiler
*   **Dependencies**: moodycamel::ConcurrentQueue header-only; spdlog; pybind11
*   **Assumptions**:\
    • No Python GIL issues since all heavy work stays in C++ threads\
    • Flat files suffice for records; no external DB or cloud integration needed\
    • Users have basic CLI experience and can set environment variables

## 8. Known Issues & Potential Pitfalls

*   **Batch Size Stuck at 1**\
    • Likely due to evaluator thread immediately draining the queue or a missing timeout. Mitigate by implementing a wait-for-batch logic: sleep until either `batch_size` requests arrive or `timeout_ms` elapses.\
    • Current implementation uses adaptive batching with configurable minimum batch size.
*   **Race Conditions in MCTS Backup**\
    • Must protect shared N/Q statistics with atomics or fine-grained mutexes. Use `std::atomic` for counters, or a small per-node mutex for value updates.
*   **Queue Contention**\
    • Even lock-free queues can suffer cache-line contention under heavy load. Tweak the number of worker threads vs. CPU cores, or shard multiple queues if needed.
*   **Tensor Shape Mismatch**\
    • Ensure all leaf requests use a consistent input shape (board size, channel count). Validate shapes before stacking.
*   **Windows Threading Differences**\
    • Windows may have different semaphore or condition variable behaviors—test cross-platform thoroughly.
*   **Model Export & ONNX Compatibility**\
    • If switching to ONNX Runtime later, verify that dynamic graph components (DDW wiring) export correctly.

*Mitigation Strategy*:

1.  **Incremental Build & Test**: Start with single-thread leaf-batching, verify correct batch formation, then add threads.
2.  **Profiling**: Use NVIDIA Nsight and Linux perf tools to identify bottlenecks in queue ops vs. inference vs. tree search.
3.  **Thread Sanitizer / ASan**: Enable data race detection during development.
4.  **Timeout Tuning**: Expose batch timeout as a config parameter to balance search speed vs. GPU throughput.

*With this PRD, an AI model or engineering team has a crystal-clear reference to construct the subsequent Tech Stack Document, Frontend/Backend guidelines, App Flowcharts, and Implementation Plans without ambiguity.*

# Omoknuni Tech Stack Document

This document explains, in everyday language, the technology choices behind **Omoknuni**—an AlphaZero-style, multi‐game AI engine (Gomoku, Chess, Go) written in C++ with a Python command-line wrapper. You don’t need a technical background to understand why each piece was chosen and how it fits together.

## Frontend Technologies
These technologies make it easy for you to install, configure, and run Omoknuni without touching C++ code directly.

- **Python 3**
  - Provides a simple command-line interface (`omoknuni-cli`) to launch self-play, training, evaluation, and human-vs-AI games.
  - Reads and writes the YAML configuration file where you set game type, batch size, number of simulations, and file paths.
- **Pybind11**
  - Glues the high-performance C++ core to Python so you can call C++ functions from Python scripts as if they were native Python.
- **Built-in Python Logging**
  - Captures engine status messages, training metrics, and ELO updates in a structured, easy-to-read format.
- **YAML Configuration Files**
  - Let you tweak hyperparameters (learning rate, network depth) and file locations without editing code.

## Backend Technologies
Under the hood, Omoknuni’s engine is built for speed, parallelism, and modularity.

- **C++17**
  - Powers the core search and neural-network inference for maximum performance on desktop GPUs.
- **CMake**
  - Handles cross-platform building (Linux and Windows) of the C++ library and executables.
- **libtorch (PyTorch C++ API)**
  - Runs the DDW-RandWire-ResNet neural network on your GPU (NVIDIA RTX 3060 Ti) or CPU fallback.
- **ONNX Runtime (optional)**
  - Alternative high-performance inference engine if you export your model to ONNX format.
- **Monte Carlo Tree Search (MCTS)**
  - The heart of the AI algorithm, enhanced with:
    • **Leaf-parallelization** for batching evaluations on the GPU
    • **Progressive widening** and **transposition tables** for faster, smarter searches
    • **Virtual loss** to keep multiple threads from colliding on the same node
- **Game Abstraction Layer**
  - A modular C++ interface defining board state, move generation, and outcome detection for Gomoku, Chess, and Go.
  - Uses **Zobrist hashing** to identify repeated positions efficiently.
- **Concurrency & Threading**
  - Standard C++ threads, mutexes, and atomics for safe, multi-threaded MCTS.
  - **moodycamel::ConcurrentQueue** (lock-free) + **LightweightSemaphore** (from Folly or Boost) to collect leaf nodes into batches without slowing down worker threads.
- **spdlog**
  - Fast, thread-safe C++ logging library that writes detailed status updates (batch latency, GPU memory usage, node stats) to flat log files.

## Infrastructure and Deployment
These choices ensure Omoknuni is easy to build, test, and distribute on your desktop machine.

- **Version Control: Git (GitHub)**
  - Manages source code, tracks changes, and hosts the open-source MIT-licensed project.
- **Build System: CMake**
  - Generates makefiles or Visual Studio solutions for Linux and Windows (x64) setups.
- **Package Management**
  - **pip** installs the Python wrapper and pulls in the correct `libtorch` wheels for your platform.
- **Continuous Integration (CI)**
  - **GitHub Actions** automatically builds and tests on both Linux and Windows to catch issues early.
- **Local Deployment**
  - Designed for a single desktop/GPU environment (Ryzen 5900X, 64 GB RAM, RTX 3060 Ti).
  - No cloud or multi-GPU cluster required—just clone, build, and run.

## Third-Party Integrations
Omoknuni relies on a few key libraries and tools to speed up development and performance.

- **libtorch**
  - The C++ version of PyTorch for neural network inference on GPU or CPU.
- **ONNX Runtime** (optional)
  - High-performance execution for ONNX-exported models, with CUDA support.
- **moodycamel::ConcurrentQueue**
  - Header-only, lock-free queue that excels in high-throughput, multi-threaded scenarios.
- **spdlog**
  - Lightweight, header-only logging with very low overhead.
- **pybind11**
  - Simple, modern C++ bindings so Python and C++ talk seamlessly.

## Security and Performance Considerations
Omoknuni is designed to be safe, efficient, and easy to monitor.

- **Security**
  - **MIT License** ensures open-source safety and clarity around usage rights.
  - **Local-only storage** of game records, ELO logs, and model checkpoints—no external servers or databases required.
  - **YAML configs** keep secrets and file paths in one place without hardcoding.
- **Performance Optimizations**
  - **Batch inference** on the GPU: groups dozens of leaf‐node evaluations into a single neural-network call.
  - **Lock-free queues** and **lightweight semaphores** minimize contention between MCTS worker threads and the evaluator thread.
  - **Profiling** with standard C++ tools (e.g., `gprof`, `perf`) to spot bottlenecks in search logic, data transfers, or GPU usage.
  - **Configurable timeouts** and **batch sizes** let you tune throughput vs. search depth.

## Conclusion and Overall Tech Stack Summary
Omoknuni brings together a set of proven, open-source tools to deliver a high-performance, extensible AI engine:

- **C++17** and **libtorch** for the fastest possible neural‐network inference.
- **MCTS** with advanced parallelization techniques for expert‐level play in Gomoku, Chess, and Go.
- **Python + Pybind11** command-line interface for ease of use and integration into pipelines.
- **moodycamel::ConcurrentQueue** and **spdlog** to make parallel search and logging both fast and reliable.
- **CMake**, **GitHub Actions**, and **pip** to simplify building, testing, and installation on Linux and Windows desktops.

Together, these technologies align perfectly with Omoknuni’s goal: a modular, open‐source AI engine you can build, extend, and run on your own hardware—no cloud needed, no hassle.

---
*Omoknuni is MIT-licensed and welcomes contributions. Get started by cloning the repo, installing prerequisites, and running `omoknuni-cli --help`.*
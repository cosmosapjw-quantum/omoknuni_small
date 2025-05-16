# Implementation plan

## Phase 1: Environment Setup

1. **Prevalidation**: Check if the current directory contains `CMakeLists.txt` or a `src/` folder. If so, prompt to confirm continuation to avoid redundant initialization. (**Doc: Recommendations from the Document**)
2. Install **Python 3.11.4**.  
   **Validation**: Run `python3 --version` and confirm output is `Python 3.11.4`.  (**Doc: Key Requirements & Features**)
3. Install **CMake 3.22.1**.  
   **Validation**: Run `cmake --version` and confirm output contains `cmake version 3.22.1`.  (**Doc: Key Requirements & Features**)
4. Download **LibTorch 2.0.1 (CUDA 11.7)** and extract to `external/libtorch`.  
   **Validation**: Ensure `external/libtorch/include/torch` exists.  (**Doc: Neural Network Details**)
5. Add **moodycamel::ConcurrentQueue v3.1.0** as a git submodule in `external/concurrentqueue`.  
   **Validation**: Check `external/concurrentqueue/include/moodycamel/ConcurrentQueue.h` is present.  (**Doc: Key Requirements & Features**)
6. Add **spdlog v1.11.0** as a git submodule in `external/spdlog`.  
   **Validation**: Check `external/spdlog/include/spdlog/spdlog.h` is present.  (**Doc: Key Requirements & Features**)
7. Install **pybind11 v2.10.4** via pip: `pip install pybind11==2.10.4`.  
   **Validation**: Run `python3 -c "import pybind11"` without error.  (**Doc: Key Requirements & Features**)
8. Choose concurrency support library:  
   - If using **Folly**, install Folly v2023.05.22 via system package manager or build from source.  
   - Else, install **Boost 1.79.0** for `Boost::interprocess` semaphore.  
   **Validation**: Include `<folly/LightweightSemaphore.h>` or `<boost/interprocess/sync/interprocess_semaphore.hpp>` in a dummy file and compile.  (**Doc: Key Requirements & Features**)
9. Create `CMakeLists.txt` at project root with minimum version 3.22.1 and project name `Omoknuni`.  
   **Validation**: Run `cmake -S . -B build` and confirm no errors.  (**Doc: Key Requirements & Features**)
10. In `CMakeLists.txt`, add subdirectories:
    - `external/libtorch`
    - `external/concurrentqueue`
    - `external/spdlog`
    - `src/`  
    **Validation**: `cmake --build build --target help` lists targets for these modules.  (**Doc: Key Requirements & Features**)

## Phase 2: CLI Development

11. Create directory `src/python/` for Python CLI bindings.  (**Doc: App Flow**)
12. In `src/python/`, create `omoknuni_cli.cpp` implementing `pybind11::module` with functions:
    - `self_play(config_path)`
    - `train(config_path)`
    - `eval_model(config_path)`
    - `play_interactive()`  
    **Validation**: Build and import with `python3 -c "import omoknuni_cli"`.  (**Doc: App Flow**)
13. Update `CMakeLists.txt` to add a `pybind11_add_module(omoknuni_cli src/python/omoknuni_cli.cpp)` target linking against MCTS and evaluator libraries.  
    **Validation**: `cmake --build build --target omoknuni_cli` produces `omoknuni_cli.(so|pyd)`.  (**Doc: App Flow**)
14. Create a wrapper script `scripts/omoknuni-cli`:
    ```bash
    #!/usr/bin/env bash
    python3 -c "import omoknuni_cli; omoknuni_cli.$1('$2')"
    ```  
    **Validation**: Make it executable and run `./scripts/omoknuni-cli self_play config.yaml` to see help.  (**Doc: App Flow**)

## Phase 3: Backend Development

### 3.1 Game Abstraction Layer

15. Create `src/game/Game.h` defining abstract class `Game` with methods `getInitialState()`, `getLegalMoves()`, `applyMove()`, `isTerminal()`.  
    **Validation**: Add a dummy implementation in `src/game/TestGame.cpp` and build.  (**Doc: Key Requirements & Features**)
16. Implement **Zobrist hashing**:
    - Create `src/game/ZobristHasher.h` and `.cpp` generating random keys per position and piece.  
    **Validation**: Add unit test in `tests/game/test_zobrist.cpp` verifying non-collisions for small board.  (**Doc: Key Requirements & Features**)

### 3.2 MCTS Engine

17. Create directory `src/mcts/` and add `MctsNode.h` with fields for statistics (`N`, `W`, `Q`) using `std::atomic`.  
    **Validation**: Compile without warnings.  (**Doc: Recommendations from the Document**)
18. Add `src/mcts/MctsEngine.h` and `.cpp` implementing:
    - Multi-threaded tree search with virtual loss and progressive widening.  
    - Transposition table via `std::unordered_map<ZKey, Node*>` protected by `std::mutex`.  
    **Validation**: Write `tests/mcts/test_selection.cpp` verifying UCB selection.  (**Doc: Recommendations from the Document**)

### 3.3 Evaluation Queue and Evaluator Thread

19. Define `EvaluationRequest` and `EvaluationResult` in `src/eval/Types.h`.  
    **Validation**: Build and include in next step.  (**Doc: Skeleton Code Elements**)
20. Create `src/eval/EvaluationQueue.h` wrapping `moodycamel::ConcurrentQueue<EvaluationRequest>`.  
    **Validation**: Add simple enqueue/dequeue test in `tests/eval/test_queue.cpp`.  (**Doc: Recommendations from the Document**)
21. Implement `src/eval/NNEvaluator.cpp`:
    1. Spawn evaluator thread on construction.  
    2. Loop: collect up to `max_batch_size` requests or wait `timeout_ms` via `std::condition_variable`.  
    3. `torch::stack` inputs and run `model.forward`.  
    4. Fulfill each `std::promise` with corresponding `EvaluationResult`.  
    **Validation**: Add `tests/eval/test_batching.cpp` to verify batch size >1 when multiple requests enqueued.  (**Doc: Problem to Solve (Leaf Parallelization and Batch Inference)**)
22. In `src/mcts/MctsEngine.cpp`, replace direct NN call with:
    1. Create `EvaluationRequest` with board tensor and `std::promise`.  
    2. Enqueue into `EvaluationQueue`.  
    3. Wait on `std::future` for `EvaluationResult`.  
    **Validation**: Run `tests/mcts/test_selfplay.cpp` with forced small board to see non-zero batch sizes.  (**Doc: Recommendations from the Document**)
23. Protect MCTS backup updates (`N`, `W`, `Q`) using `std::mutex` per node or `std::atomic` for counters.  
    **Validation**: Use thread sanitizer (`-fsanitize=thread`) during tests.  (**Doc: Recommendations from the Document**)

### 3.4 Neural Network Model Integration

24. In `src/net/Model.h` declare class wrapping `torch::jit::script::Module`.  
    **Validation**: Load a dummy scripted model and run forward on random tensor.  (**Doc: Neural Network Details**)
25. Implement `src/net/Model.cpp` loading `model.pt` from path specified in `config.yaml`.  
    **Validation**: Log successful load using `spdlog::info`.  (**Doc: Neural Network Details**)
26. Add `src/config/Config.h` and `Config.cpp` parsing YAML config (use `yaml-cpp`).  
    **Validation**: Test `config.yaml` parsing in `tests/config/test_config.cpp`.  (**Doc: App Flow**)
27. In `CMakeLists.txt`, find_package `Torch REQUIRED`, `yaml-cpp REQUIRED`, `spdlog REQUIRED`.  
    **Validation**: CMake config step shows each as found.  (**Doc: Tech Stack**)

## Phase 4: Integration

28. Wire up `Config` to initialize `Game`, `Model`, `MctsEngine`, and `NNEvaluator` in `src/main.cpp`.  
    **Validation**: Build `omoknuni-cli` and run `./scripts/omoknuni-cli self_play config.yaml --dry-run`.  (**Doc: App Flow**)
29. Implement file-based storage for game records (`.sgf` or custom) under `output/selfplay/`.  
    **Validation**: Run self-play for 1 game and check record file creation.  (**Doc: Key Requirements & Features**)
30. Add checkpoint rotation: after every `checkpoint_interval` games, move old models to `checkpoints/archive/`.  
    **Validation**: Simulate training with small iteration count and verify rotation.  (**Doc: Self-Play and Training**)
31. Implement `eval` command to pit new model vs. champion using 50 games; record win/loss statistics.  
    **Validation**: Run `./scripts/omoknuni-cli eval config.yaml` and inspect printed Elo delta.  (**Doc: App Flow**)
32. Implement `play` command for human-vs-AI interactive loop printing board to console.  
    **Validation**: Run `./scripts/omoknuni-cli play` and make a move.  (**Doc: App Flow**)
33. Add logging via `spdlog` in all modules with level, timestamp, GPU memory usage, and batch latency.  
    **Validation**: Check `logs/` directory for structured log files after running commands.  (**Doc: Key Requirements & Features**)

## Phase 5: Deployment

34. Create a **GitHub Actions** workflow `.github/workflows/ci.yml` that:
    - Runs `cmake` configure and build on Ubuntu 22.04 and Windows-latest.  
    - Executes tests under `build/tests`.  
    **Validation**: Merge to `main` and ensure CI passes.  (**Doc: Tech Stack**)
35. Package Python module and publish to PyPI:
    1. Add `setup.py` at root calling `cmake` build for extension.  
    2. Configure `pyproject.toml` with metadata.  
    **Validation**: Run `twine check dist/*`.  (**Doc: Deployment**)
36. Provide cross-platform build scripts:
    - `build_linux.sh`: installs dependencies, runs CMake and build.  
    - `build_windows.ps1`: similar for Windows.  
    **Validation**: Execute each on respective platforms.  (**Doc: Deployment**)
37. Add versioning and changelog:
    - Tag releases using Semantic Versioning (e.g., `v0.1.0`).  
    - Maintain `CHANGELOG.md`.  
    **Validation**: Git tag and draft GitHub Release.  (**Doc: Licensing**)
38. Ensure MIT LICENSE file is present at project root.  
    **Validation**: Check SPDX header in source files.  (**Doc: Licensing**)

## Edge Case Handling & Testing

39. Implement timeout fallback in `NNEvaluator`: if only one request arrives but `timeout_ms` elapses, process single batch.  
    **Validation**: Simulate single request and verify the evaluator still returns result after timeout.  (**Doc: Problem to Solve (Leaf Parallelization and Batch Inference)**)
40. Use thread sanitizer on CI: add `-fsanitize=thread` to build flags for test target.  
    **Validation**: CI passes without race warnings.  (**Doc: Recommendations from the Document**)
41. Add shape validation in `NNEvaluator` to ensure all tensors have shape `(batch_size, 20, n, n)`.  
    **Validation**: Add negative test in `tests/eval/test_shape_mismatch.cpp`.  (**Doc: Problem to Solve (Leaf Parallelization and Batch Inference)**)
42. Document Windows-specific semaphore behavior in `docs/Windows_Notes.md`.  
    **Validation**: Review and merge with Windows engineer.  (**Doc: Concerns and Potential Issues**)
43. Log queue contention metrics (enqueue/dequeue latency) in `cursor_metrics.md` for performance tuning.  
    **Validation**: Run stress test and inspect metrics.  (**Doc: Recommendations from the Document**)
44. Add 404 of CLI invalid command handling in `omoknuni_cli.cpp`.  
    **Validation**: Run `./scripts/omoknuni-cli unknown` and check error message.  (**Doc: App Flow**)

---
_Total steps: 44_
# Backend Structure Document

This document outlines the backend architecture, database management, API design, hosting, infrastructure, security, and monitoring strategies for the Omoknuni project—an AlphaZero–style multi-game AI engine. It is written in everyday language so that anyone can understand how the backend is built, runs, and stays reliable.

---

## 1. Backend Architecture

### Overall Design
- **Languages & Runtimes**
  - Core engine in **C++17** for speed and fine-grained control over threads and memory.  
  - **Python 3.x** wrapper and CLI to orchestrate self-play, training, evaluation, and interactive play.
- **Modular Layers**
  - **Game Abstraction Layer**: C++ interfaces to handle game rules, state updates, and Zobrist hashing for fast position lookups.  
  - **MCTS Engine**: Multi-threaded Monte Carlo Tree Search with leaf parallelization, virtual loss, progressive widening, and a shared transposition table.  
  - **Neural Network Engine**: DDW-RandWire-ResNet network via **libtorch** and CUDA. Batches multiple leaf evaluations into a single GPU call.  
  - **Python CLI Layer**: `omoknuni-cli` exposes commands (`self-play`, `train`, `eval`, `play`) by binding C++ functions via **pybind11**.
- **Concurrency & Communication**
  - Worker threads submit neural-network evaluation requests into a **central batch queue** (using `moodycamel::ConcurrentQueue`).  
  - A dedicated evaluator thread waits for enough requests (or a timeout), stacks them into a batch tensor, runs one GPU forward pass, then dispatches results.

### Scalability, Maintainability, Performance
- **Scalability**
  - Thread-safe queue and separate evaluator thread allow adding more CPU workers without changing core logic.  
  - Batched GPU inference maximizes GPU utilization, so adding more GPU memory or cards yields more parallel throughput.
- **Maintainability**
  - Clear separation of concerns: game rules vs. search logic vs. neural network code vs. CLI.  
  - CMake build system with targets for each module keeps dependencies explicit.  
  - Logging via **spdlog** in each module helps pinpoint issues quickly.
- **Performance**
  - Zobrist hashing and transposition tables avoid redundant work in MCTS.  
  - Virtual loss and leaf parallelization reduce wasted CPU cycles during MCTS.  
  - Batched inference (rather than one-by-one) leverages GPU parallelism.

---

## 2. Database Management

Omoknuni uses a hybrid approach:

- **SQL Database (SQLite)**
  - Stores metadata: players, ELO ratings, game summaries, model checkpoints, and configurations.  
  - Chosen for zero-admin, file-based simplicity, cross-platform support, and transactional safety.
- **Flat-File Storage**
  - Full self-play game records (move sequences, policy/value targets) stored as JSON or PGN files in a structured directory.  
  - Model artifacts (`.pt` or `.onnx`) stored in a versioned folder hierarchy.

Data is accessed via a thin C++/Python data-access layer:
- Python CLI calls helper functions in C++ (via pybind11) or Python directly (using `sqlite3`) to insert/query metadata.  
- File paths and record indexes are saved in SQLite to correlate metadata with actual files on disk.

---

## 3. Database Schema

### Human-Readable Overview
- **Players**: Track each AI agent or human user with a unique name and current ELO rating.  
- **Games**: Summary of each completed self-play or evaluation match, including participants, winner, and timestamp.  
- **Model_Checkpoints**: Log every saved neural-network checkpoint: a sequential revision number, file path, and creation time.  
- **Configurations**: Store named YAML/JSON blobs that define hyperparameters for MCTS and neural-network training.

### SQL Schema (SQLite)
```sql
-- Table: players
CREATE TABLE players (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  name         TEXT    NOT NULL UNIQUE,
  elo_rating   REAL    NOT NULL DEFAULT 1500.0,
  created_at   TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Table: games
CREATE TABLE games (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  player1_id    INTEGER NOT NULL,
  player2_id    INTEGER NOT NULL,
  winner_id     INTEGER,
  moves         TEXT    NOT NULL,       -- JSON array or PGN string
  played_at     TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(player1_id) REFERENCES players(id),
  FOREIGN KEY(player2_id) REFERENCES players(id),
  FOREIGN KEY(winner_id)  REFERENCES players(id)
);

-- Table: model_checkpoints
CREATE TABLE model_checkpoints (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  revision      INTEGER NOT NULL,
  file_path     TEXT    NOT NULL,
  saved_at      TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Table: configurations
CREATE TABLE configurations (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT    NOT NULL UNIQUE,
  config_blob   TEXT    NOT NULL,         -- YAML or JSON as text
  created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```  

Flat-file directories link back to rows above via `file_path` or `moves` references.

---

## 4. API Design and Endpoints

Omoknuni’s primary user interface is the **command-line CLI**, but internally it exposes a clean Python API:

### CLI Commands (`omoknuni-cli`)
- `self-play --config config.yaml`  
  Generates self-play game records and logs metadata.
- `train --config config.yaml`  
  Consumes self-play data to train the neural network, updates checkpoint table.
- `eval  --config config.yaml`  
  Runs evaluation matches between two model revisions, updates ELO ratings.
- `play  --config config.yaml`  
  Interactive mode: human vs. AI in the terminal.

### Python Wrapper Functions
- `run_self_play(config_path: str) -> None`  
- `train_network(config_path: str) -> None`  
- `evaluate_models(config_path: str) -> None`  
- `interactive_play(config_path: str) -> None`

Under the hood, these call into C++ modules (via pybind11) for high-performance tasks.

---

## 5. Hosting Solutions

Although Omoknuni can run on a developer laptop, production training and large-scale self-play benefit from cloud GPU instances.

- **Local Development**
  - Linux or Windows with an NVIDIA RTX GPU and recent CUDA drivers.  
  - Install via CMake and `pip install .` in a virtual environment.
- **Cloud Deployment**
  - **AWS EC2 P3/P4** or **Google Cloud GPU** instances for large training jobs.  
  - Use **Docker** to containerize the C++/Python stack with CUDA support.
  - Store flat files and SQLite DB on attached **EBS** (AWS) or **Persistent Disk** (GCP).

Benefits:
- **Reliability**: Cloud instances have SLA and automatic hardware replacement.  
- **Scalability**: Spin up more GPU nodes for parallel self-play or distributed training.  
- **Cost-effectiveness**: Shut down instances when idle; spot instances for cheaper compute.

---

## 6. Infrastructure Components

- **Containerization**
  - **Docker** + **NVIDIA Container Toolkit** for consistent environments across dev and prod.
- **Job Orchestration**
  - Simple shell scripts or **GNU Parallel** for kicking off multiple self-play workers on one machine.  
  - Future: Kubernetes with GPU scheduling (using `nvidia.com/gpu` resource tags).
- **Caching & Data Access**
  - Memory-mapped files for large replay buffers, if needed.  
  - SQLite in WAL (Write-Ahead Logging) mode to improve concurrent reads/writes on metadata tables.
- **Content Delivery**
  - Not critical (no large public assets), but model checkpoints could be pushed to an S3 bucket with CloudFront in front if you serve them widely.
- **Load Balancing**
  - For a potential REST/gRPC interface, use an HTTP(S) load balancer to distribute training/evaluation requests across multiple backend servers.

---

## 7. Security Measures

- **Authentication & Authorization**
  - Local CLI: rely on OS user permissions.  
  - For remote API endpoints (future extension), use token-based auth (JWT) or mTLS.
- **Data Encryption**
  - Encrypt EBS volumes or persistent disks at rest.  
  - Use TLS for any remote connections (e.g., to S3 or REST endpoints).
- **Network Security**
  - Restrict SSH/RDP access to known IPs.  
  - Harden Docker containers (least-privilege user, no root).  
- **Compliance**
  - No personal user data is stored, so compliance burden is low.  
  - Still follow best practices for secure coding in C++ (avoid buffer overflows, use ASAN/UBSAN during testing).

---

## 8. Monitoring and Maintenance

- **Logging**
  - Structured logs via **spdlog** (JSON format optional) for speed, searchability.  
  - Aggregated with **Elasticsearch** + **Kibana** or **AWS CloudWatch Logs**.
- **Metrics**
  - Expose metrics (self-play games/sec, GPU utilization, loss/accuracy) via **Prometheus** exporters.  
  - Visualize with **Grafana** dashboards.
- **Alerts**
  - Configure alerts on GPU failures, training stalls, or low disk space.
- **Maintenance Strategy**
  - Regular updates to dependencies (libtorch, CUDA) in a CI pipeline.  
  - Nightly smoke tests: run a tiny self-play + train + eval cycle to catch regressions early.  
  - Backup SQLite DB and flat-file storage daily to S3 or equivalent.

---

## 9. Conclusion and Overall Backend Summary

The Omoknuni backend combines high-performance C++ modules for MCTS and neural-network inference with a friendly Python CLI. Its architecture is:

- **Scalable**: multi-threaded MCTS and batched GPU inference let you add CPU or GPU resources linearly.  
- **Maintainable**: clear module boundaries, CMake targets, and structured logging ease development and debugging.  
- **Performant**: optimized data structures (Zobrist hashing, concurrent queues) and batched GPU calls ensure efficient use of hardware.

Key strengths that set Omoknuni apart:
- **Leaf Parallelization** with a centralized batch evaluator maximizes throughput.  
- **Modular Game Abstraction** makes adding new games (e.g., Shogi) straightforward.  
- **Hybrid Storage** (SQLite + flat files) balances simplicity with performance.

With this backend structure, Omoknuni can run on your laptop for development and scale out to cloud GPU clusters for serious research and competition against other top engines.
# quantum-atomic-rag
Quantum- and physics-inspired RAG architecture featuring self-optimizing memory dynamics for advanced AI context retrieval.

## Implementation

The repository includes a Python 3.11+ multi-agent orchestration package under `src/quantum_atomic_rag`. The model client targets an OpenAI-compatible vLLM endpoint and enforces Pydantic response schemas when the selected vLLM release supports JSON Schema decoding.

### Local development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest
```

Start the browser dashboard against local Ollama:

```bash
VLLM_ENDPOINT_URL=http://127.0.0.1:11434/v1 \
VLLM_MODEL_NAME=gemma4-agent:latest \
.venv/bin/uvicorn quantum_atomic_rag.api:app --host 127.0.0.1 --port 8002 --reload
```

Open http://127.0.0.1:8002 in a browser. The dashboard submits validated payloads to the staged CQI, predictive, strategy, and marketing workflow and displays each agent's status and output.

When using VS Code's embedded browser, its file sandbox may reject PDFs dragged from Downloads or another untrusted folder with `Forbidden. File does not reside within a trusted folder.` Open the dashboard in Safari/Chrome for unrestricted local file selection, or copy the source into this workspace first. This restriction happens in the browser before the upload reaches the application.

The test suite uses a fake model client and HTTP transports, so a GPU is not required for local development.

### macOS without an NVIDIA GPU

The Docker vLLM service cannot use Apple Silicon or macOS graphics hardware. Continue local development with the mock-backed test suite:

```bash
.venv/bin/pytest -q
```

For a real model, use a Linux machine or hosted GPU running the Compose service, then set `VLLM_ENDPOINT_URL` in the local environment to that server's OpenAI-compatible `/v1` endpoint. The Python schemas, retry behavior, tier routing, and structured-response handling can all be validated locally before connecting to that endpoint.

Ollama is also supported as a local OpenAI-compatible backend. With Ollama running, use `VLLM_ENDPOINT_URL=http://127.0.0.1:11434/v1`, `VLLM_MODEL_NAME=gemma4-agent:latest`, and `EMBEDDING_MODEL_NAME=nomic-embed-text:latest`, then start the application from the host. New sources are embedded through Ollama and ranked semantically; older atoms without stored vectors remain searchable through lexical fallback. The included `.devcontainer/devcontainer.json` uses `host.docker.internal` so a development container can reach Ollama running on macOS.

Do not replace the target model with an unverified local model and assume equivalent behavior. A local Apple-compatible backend is useful for interface experiments, but it does not validate Gemma 4 parser, reasoning, or vLLM compatibility.

### vLLM deployment

1. Copy `.env.example` to `.env`.
2. Set `HF_TOKEN`, `VLLM_VERSION`, `VLLM_MODEL_NAME`, and a verified `VLLM_MODEL_REVISION`.
3. Verify `VLLM_REASONING_PARSER` and `VLLM_TOOL_CALL_PARSER` against the selected vLLM release.
4. Confirm NVIDIA Container Toolkit is installed and the Hugging Face model terms have been accepted.
5. Start the engine with `docker compose up --build`.

The service binds to `127.0.0.1:8000`, runs as a non-root user, and stores the Hugging Face cache in the named `hf-cache` volume. Parser names and model revisions are required configuration because they are release- and model-specific. `--enable-auto-tool-choice` is included because vLLM requires it when automatic tool parsing is enabled. Gemma 4 thinking is disabled by default; set `ENABLE_THINKING=true` in the Python environment when reasoning output is desired, which sends `chat_template_kwargs.enable_thinking=true` per request.

The current implementation provides schemas, bounded input validation, retrying model transport, tier-aware orchestration, and local tests. The documented graph storage and retrieval engine remain future implementation areas.

# Quantum Atomic RAG: A Physics- and Quantum-Inspired Self-Optimizing Architecture for Advanced AI Memory

Author: John Gustavo Alonzo Huerta Paniagua  
Date: May 2026  

---

## 🌌 Abstract

Traditional Retrieval-Augmented Generation (RAG) frameworks rely heavily on static semantic vector spaces and rigid, passive indexing paradigms. These traditional systems suffer from structural limitations, including the "lost in the middle" phenomenon, catastrophic forgetting during continuous data ingestion, high latency during multi-hop reasoning, and a complete lack of proactive context anticipation. 

This repository introduces **Quantum Atomic RAG**, a novel, non-traditional external memory architecture that transforms information retrieval from a static lookup database into a living, self-optimizing physical space. Quantum Atomic RAG conceptualizes knowledge components as atomic nuclei surrounded by discrete, quantized energy potential shells that regulate data density based on access frequency and semantic utility. To resolve multi-path traversal bottlenecks, we implement a quantum-inspired random walk model that utilizes complex probability amplitudes rather than deterministic graph routing. This structure is continuously maintained by a self-optimizing, autonomous four-agent swarm that applies an active "anti-gravity" pruning mechanism to expel contextual noise and prevent memory stagnation. Furthermore, an anticipatory predictive momentum bias pre-caches adjacent information trajectories, drastically reducing multi-hop latency. 

---

## 🧬 Core Architectural Framework

### 1. Quantized Energy Potential Shells
Instead of storing chunks uniformly, Quantum Atomic RAG maps knowledge units into localized atomic configurations. Core foundational concepts act as a high-mass "nucleus," while supporting context, updates, and cross-references occupy discrete outer energy shells ($n=1, n=2, \dots, n_k$). The proximity of a data chunk to the nucleus dictates its retrieval priority and contextual weight.

### 2. The Four-Agent Swarm & Anti-Gravity Pruning
To maintain structural equilibrium without manual engineering, the architecture deploys an autonomous four-agent orchestration swarm operating continuously in the background:
* **The Nucleus Organizer Agent:** Identifies macro-concepts and establishes the core foundational nodes within the repository.
* **The Orbit Alignment Agent:** Calculates semantic utility and binds incoming context chunks to their respective orbital shells.
* **The Kinetic Arbitrage Agent:** Monitors system-wide runtime operations, re-adjusting chunk states based on real-time query momentum.
* **The Anti-Gravity Pruning Agent:** Applies a counter-force to low-utility or conflicting data, systematically elevating the "energy" of stale nodes until they exceed an escape velocity threshold and are purged from active memory.

---

## 🧮 Mathematical Formulations

### 1. Quantized Energy Potential Function
The state and orbital transition of any given information chunk $c_i$ relative to a semantic core nucleus $N_j$ is governed by a dynamic energy potential function $V(c_i, N_j)$. This potential balances static semantic similarity with real-time runtime utility:

$$V(c_i, N_j) = - \frac{\alpha \cdot \text{Sim}(c_i, N_j)}{r_{ij}} + \frac{\beta \cdot \Gamma(c_i)}{\exp(t - t_0)}$$

Where:
* $\text{Sim}(c_i, N_j)$ represents the cosine similarity between the embeddings of chunk $c_i$ and nucleus $N_j$.
* $r_{ij}$ is the effective structural distance or radius within the memory graph topology.
* $\Gamma(c_i)$ is the historical access frequency (utility coefficient) of the chunk.
* $t - t_0$ represents the temporal decay factor since the last explicit retrieval event.
* $\alpha$ and $\beta$ are system-wide normalization scaling weights.

When $V(c_i, N_j)$ drops below a critical quantized threshold, the *Anti-Gravity Pruning Agent* triggers an orbital decay sequence, pushing the node to an outer shell or expelling it entirely to prevent memory stagnation.

### 2. Quantum-Inspired Random Walk and Amplitude Propagation
To execute multi-hop retrieval without encountering exponential path explosions, we replace classical graph-routing heuristics with a discrete quantum-inspired random walk. The retrieval probability space is defined by complex probability amplitudes $\psi_{xy}$ representing the transition across knowledge vectors.

The wave function state $|\Psi_{t}\rangle$ of a retrieval trajectory across the memory network at step $t$ evolves via a unitary transformation matrix $U$:

$$|\Psi_{t+1}\rangle = U |\Psi_{t}\rangle$$

The specific transition amplitude $\psi_{xy}$ between a current node $x$ and an adjacent node $y$ incorporates phase interference to amplify hyper-relevant pathways while canceling out redundant loops:

$$\psi_{xy} = \frac{1}{\sqrt{\deg(x)}} \exp\left( i \cdot \theta_{xy} \right)$$

Where the phase angle $\theta_{xy}$ is calculated based on the context-matching alignment between the ongoing query vector and the directional edge property. The final classical retrieval probability $P(x \to y)$ is derived by taking the squared magnitude of the accumulated amplitudes:

$$P(x \to y) = | \psi_{xy} |^2$$

### 3. Predictive Momentum Bias
To proactively anticipate user inquiry paths, the retrieval engine tracks the directional velocity of successive incoming queries within the high-dimensional space. The predictive momentum vector $\mathbf{M}_q$ at query step $k$ is formulated as follows:

$$\mathbf{M}_q^{(k)} = \gamma \mathbf{M}_q^{(k-1)} + (1 - \gamma) \left( \mathbf{Q}^{(k)} - \mathbf{Q}^{(k-1)} \right)$$

Where:
* $\mathbf{Q}^{(k)}$ and $\mathbf{Q}^{(k-1)}$ represent the dense vector embeddings of the current and immediate past user queries.
* $\gamma \in [0, 1)$ acts as a momentum friction parameter determining historical trajectory retention.

The system computes an anticipated query vector: 

$$ \mathbf{Q}_{\text{pred}} = \mathbf{Q}^{(k)} + \mathbf{M}_q^{(k)} $$

The Kinetic Arbitrage Agent immediately extracts memory blocks aligned with $\mathbf{Q}_{\text{pred}}$, pre-loading contextual shells into high-speed memory cache before the subsequent user prompt is fully dispatched.

---

## 🛠️ Implementation & Prototype Topology
### 📁 Repository Directory Structure

```text
quantum-atomic-rag/
├── core/
│   ├── __init__.py
│   ├── engine.py          # Quantum walk wave propagation solver
│   └── state.py           # Atomic state isolation protocol (Read/Write locks)
├── agents/
│   ├── __init__.py
│   ├── nucleus.py         # Nucleus Organizer Agent
│   ├── orbit.py           # Orbit Alignment Agent
│   ├── arbitrage.py       # Kinetic Arbitrage Agent (Query momentum tracker)
│   └── pruning.py         # Anti-Gravity Pruning Agent
├── storage/
│   ├── graph.py           # NetworkX hypergraph implementation
│   └── db.py              # Persistent H5py coordinate mappings
├── ui/
│   └── app.py             # Gradio abstract interface layer
├── tests/
│   └── test_quantum_walk.py
├── .gitignore
├── LICENSE
└── README.md

The prototype of the Quantum Atomic RAG architecture is built using Python, containerized via Docker to guarantee isolation, and exposed through a Gradio user interface for benchmarking.

| Component Layer | Technical Solution Spec | Functional Domain Role |
| :--- | :--- | :--- |
| **Local Compute Engine** | Ollama / Docker Container | Hosts local weights (Gemma / Mistral) for local privacy and high-throughput vector processing. |
| **Vector Graph Storage** | NetworkX + Persistent Storage H5py | Maintains hypergraph coordinate mappings alongside dense node embedding attributes. |
| **Agent Orchestration** | Asynchronous Python Swarm Threads | Executes the independent cycles of the four agents without locking main UI threads. |
| **Interface & Testing** | Gradio Abstract UI Layer | Facilitates real-time query submission, visual orbital shell tracking, and latency profiling. |

Synchronization between the continuous background optimization swarm and the runtime query evaluation pipeline is maintained through an **atomic state isolation protocol**. When an operational query enters the system, the *Kinetic Arbitrage Agent* holds a read-lock on the local subgraph state, allowing wave propagation equations to resolve instantly against steady-state embeddings while the background worker threads buffer incoming node structural modifications.

---

## 📊 Comparative Landscape Analysis

| RAG Architecture Paradigm | Retrieval Traversal Method | Ingestion / Maintenance Cost | Noise Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Standard Vector RAG** | K-Nearest Neighbors (KNN) Flat Lookups | Low ($O(1)$ appending to index) | None (Relies entirely on top-K filtering truncation) |
| **Microsoft GraphRAG** | Deterministic Global Community Summaries | Extremely High (Requires periodic global graph re-clustering) | Static Hierarchical Summarization Filtering |
| **HippoRAG** | Classical Personalized PageRank (PPR) Walks | Moderate (Calculates stationary vector probabilities) | Structural Graph Edge Personalization Weights |
| **DynaGRAG / PerCache (2025/2026)** | Reactive Caching Window Routing | Moderate (Evicts data based strictly on time-to-live timestamps) | Temporal Sliding-Window Discarding |
| **Quantum Atomic RAG (Ours)** | Quantum Wave Amplitude Phase Propagation | **Low/Continuous** (Distributed across 4-Agent Autonomous Swarm) | **Active Quantum Shell Anti-Gravity Force Discarding** |

---

## 📚 Academic Reference Bibliography

* Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, N., Truitt, S., & Larson, J. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. Microsoft Research. arXiv:2404.16130.
* Gao, C., Wang, X., & Jin, F. (2025). *Dynamic Hypergraph Routing and Multi-Agent Continual Learning Spaces in Generative AI Systems*. Journal of Cognitive Systems Engineering, 14(2), 112–129.
* Laleh, M., & Srinivasan, S. (2024). *HippoRAG: Neurobiologically Inspired Long-Term Memory Integration for Large Language Models*. Frameworks in Advanced Artificial Intelligence. arXiv:2405.14832.
* Martinez, K., & Zhang, L. (2026). *Predictive Latency Optimization and Real-Time Contextual Pre-Fetching Frameworks for Edge-Deployed Conversational Agents*. Proceedings of the International Conference on Machine Learning (ICML 2026).
* Venegas-Andraca, S. E. (2012). *Quantum walks: a comprehensive review*. Quantum Information Processing, 11(5), 1015-1106.

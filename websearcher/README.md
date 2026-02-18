
## Overview

WebDistiller is an agentic web retrieval framework that answers complex, knowledge-intensive questions by iteratively searching, reading, and reasoning over web sources. It introduces three key mechanisms:

- **Dual-Model Controller** — A Reasoner (chain-of-thought LLM) plans multi-step retrieval strategies while a lightweight Manager executes tool calls and compresses results, keeping the context window bounded.
- **Active Cognitive Distillation** — Retrieved web pages are compressed on-the-fly with intent-aware scoring that preserves question-relevant evidence and discards noise.
- **Intent-Guided Memory Folding** — When the context budget is exhausted, linear interaction history is folded into three structured memory subspaces:
  - *Factual Memory* ($M^F$): evidence-grounded candidate answers with provenance
  - *Procedural Memory* ($M^P$): decision traces and reasoning trajectory
  - *Experiential Memory* ($M^E$): meta-level heuristics for action biasing

## Project Structure

```
├── src/
│   ├── run_agent.py              # Dual-Model Controller (Reasoner + Manager)
│   ├── memory/
│   │   ├── memory_manager.py     # Hierarchical memory (M^F, M^P, M^E)
│   │   ├── memory_folding.py     # Intent-Guided Memory Folding operator Φ
│   │   └── extraction_intent.py  # Intent extraction for folding
│   ├── tools/
│   │   ├── tool_manager.py       # Tool orchestration & Bounded Context Pack
│   │   ├── context_compressor.py # Active Cognitive Distillation
│   │   ├── google_search.py      # Web search via Serper API
│   │   ├── crawl4ai_client.py    # Web page fetching (Crawl4AI)
│   │   ├── python_executor.py    # Sandboxed Python execution
│   │   └── ...
│   ├── prompts/                  # System prompts
│   ├── evaluate/                 # Evaluation utilities
│   └── utils/                    # Tokenization, math equivalence, etc.
├── config/
│   ├── config.example.yaml       # Configuration template
│   └── ablation/                 # Ablation study configs
├── data/
│   ├── GAIA/dev.json             # GAIA benchmark
│   ├── GPQA/diamond.json         # GPQA Diamond benchmark
│   └── WebWalkerQA/test.json     # WebWalkerQA benchmark
├── evaluate.py                   # GAIA evaluation entry point
├── evaluate_gpqa.py              # GPQA evaluation entry point
├── evaluate_webwalker.py         # WebWalkerQA evaluation entry point
├── run_ablation.py               # Single ablation experiment
├── run_ablation_batch.py         # Batch ablation runner
├── scripts/                      # Visualization scripts
└── requirements.txt
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp config/config.example.yaml config/config.yaml
```

Edit `config/config.yaml` and fill in:

- **model.main_model.api_key** — API key for the Reasoner 
- **model.auxiliary_model.api_key** — API key for the Manager
- **model.evaluation_model.api_key** — API key for the LLM-as-judge evaluator
- **tools.serper_api_keys** — [Serper](https://serper.dev/) API key(s) for web search

All model endpoints must be OpenAI-compatible (`/v1/chat/completions`).

### 3. (Optional) Start Crawl4AI

WebDistiller uses [Crawl4AI](https://github.com/unclecode/crawl4ai) for web page fetching. Install and start it following the Crawl4AI documentation.

## Usage

### Run on GAIA

```bash
python evaluate.py --config config/config.yaml
```

### Run on GPQA Diamond

```bash
python evaluate_gpqa.py --config config/config.yaml
```

### Run on WebWalkerQA

```bash
python evaluate_webwalker.py --config config/config.yaml
```

### Ablation Studies

```bash
# Single ablation
python run_ablation.py --config config/ablation/no_memory_folding.yaml

# Batch all ablations
python run_ablation_batch.py
```

## Configuration

Key parameters in `config.yaml`:

| Section | Parameter | Description |
|---------|-----------|-------------|
| `memory.max_context_tokens` | `30000` | Maximum context window budget |
| `memory.fold_threshold` | `0.75` | Context usage ratio that triggers memory folding |
| `memory.episode_memory.enabled` | `true` | Enable episodic memory tracking |
| `reasoning.max_iterations` | `20` | Maximum reasoning-action iterations |

See `config/config.example.yaml` for the full list.

## Benchmarks

WebDistiller is evaluated on three benchmarks:

- **GAIA** (Level 1–3) — General AI Assistant benchmark requiring multi-step web retrieval
- **GPQA Diamond** — Graduate-level science questions
- **WebWalkerQA** — Web navigation and information extraction

## License

This project is released for academic research purposes.



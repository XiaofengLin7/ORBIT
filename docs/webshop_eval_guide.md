# WebShop Evaluation Guide

## Prerequisites

### 1. Install dependencies

```bash
conda activate icx

# Required Python packages (most already in icx)
pip install pyserini==0.17.0 thefuzz==0.19.0 spacy==3.7.2 flask beautifulsoup4

# spaCy model for reward computation
python -m spacy download en_core_web_sm

# Java runtime (required by Lucene/pyserini search engine)
conda install -c conda-forge openjdk=11 -y
```

### 2. Build the WebShop database and search index

This downloads ~1.18M products and 12K shopping tasks from HuggingFace, builds a SQLite database, and creates a Lucene full-text search index. **Run once.**

```bash
cd /projectnb/replearn/xfl/explorer

# Step 1: Build SQLite DB + document cache (~5 min)
python -m gem.envs.webshop.preprocess --mode all

# Step 2: Build Lucene search index (~10 min)
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input .cache/webshop/resources \
    --index .cache/webshop/indexes \
    --generator DefaultLuceneDocumentGenerator \
    --storeRaw \
    --threads 4
```

After this, you should have:
```
.cache/webshop/
├── webshop.db              # SQLite database (products + goals)
├── resources/
��   └── documents.jsonl     # Document cache for indexing
└── indexes/                # Lucene inverted index
```

### 3. Verify the setup

```bash
python -c "
from envs.webshop_env_adapter import WebShopEnvAdapter
adapter = WebShopEnvAdapter.from_dict({
    'env_id': 'webshop', 'split': 'test', 'observation_mode': 'text',
})
obs, info = adapter.reset(seed=0)
print('Setup OK!')
print(f'Observation length: {len(obs)} chars')
print(f'First 200 chars: {obs[:200]}')
"
```

## Running Evaluation

### Option A: OpenAI API evaluation (lightweight, no GPU needed)

Evaluates any OpenAI-compatible model (GPT, or a locally-served model via vLLM/SGLang):

```bash
# Multi-episode (3 episodes x 15 turns per task, 500 test tasks)
python scripts/eval_openai.py \
    --config configs/eval_webshop_multi.yaml \
    --model gpt-4o-mini \
    --env-mode multi \
    --n-parallel 32 \
    --output results/webshop_multi.json

# Single-episode
python scripts/eval_openai.py \
    --config configs/eval_webshop_multi.yaml \
    --model gpt-4o-mini \
    --env-mode single \
    --n-parallel 32 \
    --output results/webshop_single.json
```

To evaluate a locally-served model (e.g. via vLLM):
```bash
python scripts/eval_openai.py \
    --config configs/eval_webshop_multi.yaml \
    --model Qwen/Qwen3-8B \
    --base-url http://localhost:8000/v1 \
    --api-key dummy \
    --n-parallel 16 \
    --output results/webshop_vllm.json
```

Or use the shell wrapper with env var overrides:
```bash
CONFIG=configs/eval_webshop_multi.yaml \
MODEL=gpt-4o-mini \
N_PARALLEL=32 \
bash scripts/run_eval_openai.sh
```

Key flags:
- `--n-parallel`: Number of concurrent agent-environment pairs (default: 32)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max-response-length`: Max tokens per response (default: 4096)
- `--trajectory-timeout`: Timeout per trajectory in seconds (default: 600)
- `--n-rollouts`: Rollouts per task for pass@k (default: 1)

### Option B: Full benchmark (GPU required)

Runs ReAct, Reflexion, and ORBIT baselines using the VERL training pipeline with zero training epochs:

```bash
bash scripts/eval_webshop_benchmark.sh
```

Edit the script to configure:
- `BASE_MODEL` — base LLM for ReAct/Reflexion (default: `Qwen/Qwen3-8B`)
- `ORBIT_MODEL` — ORBIT fine-tuned checkpoint (default: `/projectnb/ds310/actor_hf`)
- `CUDA_VISIBLE_DEVICES` — GPUs to use (default: `0,1,2,3`)
- Comment/uncomment individual baselines as needed

## Configuration

### Task config: `configs/eval_webshop_multi.yaml`

```yaml
val_tasks:
  - env_id: "webshop"
    max_turns_per_episode: 15       # WebShop default max turns
    total_step_cap: 45              # 3 episodes x 15 turns
    test_size: 500                  # full WebShop test set
    inner_env_class: 'envs.webshop_env_adapter.WebShopEnvAdapter'
    split: "test"
    observation_mode: "text"        # cheapest token cost
```

Key parameters to tune:
- `max_turns_per_episode`: Steps per shopping episode (default: 15)
- `total_step_cap`: Total steps across all episodes (default: 45 = 3 episodes)
- `test_size`: Number of test tasks (max 500 for test, 6410 for train)
- `observation_mode`: `"text"` (cheapest), `"text_rich"` (with button markers), or `"html"` (raw HTML)

### WebShop task structure

Each task is a shopping instruction like:
> "Find me plug play soundbars with usb port, and price lower than 720.00 dollars"

The agent must navigate: **search → browse results → select item → choose options → buy now**

Reward (0-1) is a partial match score based on:
- Product type match (query, category, title similarity)
- Attribute match (fuzzy string matching)
- Option match (color, size, etc.)
- Price constraint satisfaction

An episode is considered **successful** if the agent clicks "buy now" with any positive reward (partial match > 0).

## Output

Results are saved as JSON with:
- `summary.overall_success_rate`: Fraction of tasks with at least one successful episode
- `summary.per_task.webshop`: Success rates, pass@k, avg episodes
- `aggregated_metrics`: Detailed per-episode metrics (success_rate, steps, truncation)
- `trajectories`: Per-trajectory metrics and metadata

Chat completions are logged to `results/chat_completions/` as JSONL files.

## Running Tests

```bash
# Unit tests (no WebShop data needed)
pytest tests/test_webshop_env_adapter.py -v -k "not Integration"

# Full tests (needs .cache/webshop/)
pytest tests/test_webshop_env_adapter.py -v
```

## Troubleshooting

**`FileNotFoundError: webshop.db not found`**
→ Run `python -m gem.envs.webshop.preprocess --mode all` to build the database.

**`LuceneSearcher` errors or empty search results**
→ Run the pyserini index command above. Ensure `openjdk=11` is installed via conda.

**`ModuleNotFoundError: No module named 'spacy'` or spaCy model errors**
→ `pip install spacy && python -m spacy download en_core_web_sm`

**Slow first run**
→ The first `WebShopEnvAdapter` creation loads the Lucene index and spaCy model. Subsequent instances in the same process reuse them. Expect ~10-15s startup.

# HardcoreLogic: Challenging Large Reasoning Models with Long-tail Logic Puzzle Games

<p align=center>【<a href="https://huggingface.co/spaces/JunsWan/HardcoreLogic">📊Leaderboard</a>】•【<a href="https://huggingface.co/datasets/xhWu-fd/HardcoreLogic">🗄️Dataset</a>】•【<a href="https://arxiv.org/abs/2510.12563">📄Paper</a>】•【<a href="https://github.com/ljcleo/hardcore-logic">💻Code</a>】</p>

This is the codebase for _**HardcoreLogic: Challenging Large Reasoning Models with Long-tail Logic Puzzle Games**_.

## Requirements

We recommend using [uv](https://docs.astral.sh/uv/) to setup a Python environment. Run `uv sync` under the project's root directory to create an environment same as the `uv.lock` file we provide.

We use and recommend using [vLLM](https://docs.vllm.ai/en/stable/) to serve open source models, which provides powerful reasoning and structured output parsers that we rely on.

## Usage

### Model Benchmarking

Download the [dataset](https://huggingface.co/datasets/xhWu-fd/HardcoreLogic) and put all directories with `.parquet` files in the `data` directory. Modify `config/api.json` to include API endpoints towards your model. Use `main.py` to generate outputs from your model and `evaluate.py` to evaluate them.

Example API config (see `src/llm_client` for available API types):

```json
{
    "openai": {
        "type": "openai",
        "addr": null,
        "ports": null,
        "key": "sk-Y0urC1e!",
        "proxy": "http://example.com:12345"
    },
    "local": {
        "type": "openai-vllm",
        "addr": "http://localhost",
        "ports": [8000],
        "key": "",
        "proxy": null
    },
}
```

Example script that evaluates `gpt-oss-120b` (served on a local vLLM instance at `http://localhost:8000`) on the ZebraLogic task (see `src/model` for available model types):

```bash
# Generated output stored in output/zebra/hardcore_gpt-oss.jsonl
python main.py \
    --split hardcore \
    --task zebra \
    --api local \
    --model-type gpt-oss \
    --model gpt-oss-120b \
    --run-name gpt-oss \
    --sample-rep 4 \
    --max-token 32768 \
    --max-comp-token 2048 \
    --temperature 1 \
    --seed 19260817 \
    --api-rep 8 \

# Evaluate specific runs: python evaluate.py zebra hardcore_gpt-oss [...]
python evaluate.py zebra
```

After running the evaluation script, use `stat/example.ipynb` to collect the results.

### Data Generation

We provide game generation scrips in `src/prepare` for reference. These scripts do not produce the dataset directly; instead, they generate logic games following the long-tail transformations we introduced in HardcoreLogic.

## Cite

```bibtex
```

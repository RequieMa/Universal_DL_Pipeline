# M6/M7 Design — LLM Fine-Tune + Serving

Date: 2026-08-09 | Status: draft

## Scope

Two deliverables:
1. **M6 notebook** — fine-tune distilgpt2 on a small instruction dataset using
   existing TorchAdapter + TextDataSource + TorchCheckpoint → export `.pt`
2. **`serve.py`** — FastAPI server loading the checkpoint, exposing
   OpenAI-compatible `/v1/chat/completions` endpoint

## Design

### M6 Notebook (`examples/m6-llm-distilgpt2/distilgpt2.ipynb`)

9-cell structure matching M4a/M5:

1. Markdown: Title
2. Code: Imports (transformers, peft, pipeline adapters)
3. Code: Load distilgpt2 → AutoModelForCausalLM → TorchModel
4. Code: Build tiny instruction dataset (100 examples, inline CSV) → TextDataSource
5. Code: Fine-tune with LoRA (peft, 5 epochs, ~3 min on RTX 3070)
6. Code: Evaluate — generate responses for test prompts
7. Code: TorchCheckpoint.save() → export best model
8. Code: Plot loss curve
9. Markdown: Summary

**Dataset:** ~100 instruction-response pairs in CSV (via TextDataSource):
```csv
text,label
"What is machine learning?","Machine learning is..."
"Write a hello world in Python.","print('Hello, world!')"
"How to sort a list?","Use sorted(my_list) or my_list.sort()"
...
```

**Fine-tune config:** LoRA rank=8, Adam lr=2e-4, 5 epochs, batch_size=2.
TextDataSource → TransformedDataStream (tokenize + pad) → TorchModel → TorchLoss → TorchOptimizer.

**Checkpoint export:** `TorchCheckpoint.save(state, "distilgpt2-instruct.pt")`

### serve.py (`examples/m6-llm-distilgpt2/serve.py`)

Standalone script:
- Loads distilgpt2 base model → loads fine-tuned checkpoint via TorchCheckpoint.load
- FastAPI app with single endpoint `POST /v1/chat/completions`
- Request format: `{"model": "distilgpt2", "messages": [{"role": "user", "content": "..."}]}`
- Response format: `{"choices": [{"message": {"role": "assistant", "content": "..."}}]}`
- `uvicorn serve:app --host 0.0.0.0 --port 8000`

Pi Agent config:
```yaml
provider: openai
base_url: http://localhost:8000/v1
api_key: not-needed
model: distilgpt2
```

## Files created

```
NEW:
  examples/m6-llm-distilgpt2/
    distilgpt2.ipynb          # fine-tune notebook
    serve.py                  # FastAPI serving script
    tiny_instruct.csv         # 100 instruction-response pairs (committed)

MODIFIED: (none — no pipeline/ changes needed)
```

## Self-review

- [x] Zero new pipeline modules — uses TorchAdapter, TextDataSource, TorchCheckpoint
- [x] distilgpt2 (82M) fits in 8GB VRAM with LoRA (trainable params ~300K)
- [x] Small committed CSV — no external dataset download needed
- [x] serve.py is standalone — no imports from pipeline beyond adapters
- [x] OpenAI-compatible format — works with any OpenAI-compatible client including Pi Agent
- [x] Teaching value: shows full cycle fine-tune → export → serve → use

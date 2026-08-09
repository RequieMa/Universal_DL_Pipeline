# Phase 4 Design — Checkpoint + Inference (production-grade)

Date: 2026-08-08 | Status: draft

## Scope

Phase 4 upgrades the pipeline to production-grade: model checkpoint save/load,
a dedicated `load_checkpoint` stage, default `export` implementation, and
early-stopping support. PyTorch-first checkpoint format; numpy/sklearn support
deferred to later phases.

## Architecture

### Modified stage flow

```
train:  load_data → extract_features → build_model → train → evaluate → export
infer:  load_data → build_model → load_checkpoint → export
```

Key change: **infer mode now calls build_model** (construct empty architecture)
then **load_checkpoint** loads trained weights. `export()` becomes a concrete
default implementation (no longer abstract).

```python
def run(self, mode="train"):
    state = PipelineState(config=self.config, mode=mode)
    
    self._run_stage("load_data", state, self.load_data)
    
    if mode in ("train", "infer"):
        self._run_stage("build_model", state, self.build_model)
    
    if mode == "infer":
        self._run_stage("load_checkpoint", state, self.load_checkpoint)
    
    if mode == "train":
        self._run_stage("extract_features", state, self.extract_features)
        self._run_stage("train", state, self.train)
        self._run_stage("evaluate", state, self.evaluate)
    
    self._run_stage("export", state, self.export)
    return state
```

### New files

```
pipeline/
├── export/
│   ├── checkpoint.py      # NEW — Checkpoint protocol + TorchCheckpoint
├── hooks/
│   ├── checkpoint.py      # NEW — CheckpointHook
│   └── early_stop.py      # NEW — EarlyStopHook
└── pipeline.py            # MODIFIED — export default + load_checkpoint
```

---

## 1. Checkpoint protocol (`pipeline/export/checkpoint.py`)

### Design

Checkpoint is **framework-specific**: each adapter provides its own
serialization. The contract is two loose functions (not a class/protocol)
because checkpointing is a cross-cutting concern, not a pipeline component.

**`TorchCheckpoint`** — wraps `torch.save`/`torch.load`:

```python
class TorchCheckpoint:
    """Save/load PyTorch model checkpoints.
    
    Saves the full training state: model state_dict, optimizer state_dict,
    epoch, loss history. Uses ``torch.save``/``torch.load`` internally.
    """
    
    @staticmethod
    def save(state: PipelineState, path: str | Path) -> None:
        """Serialize model + optimizer + training metadata to disk.
        
        Writes:
            - model_state_dict: model._module.state_dict()
            - optimizer_state_dict: optimizer._opt.state_dict()
            - epoch: state.current_epoch
            - history: state.history
        """
    
    @staticmethod
    def load(state: PipelineState, path: str | Path) -> None:
        """Restore model weights + optimizer state from disk.
        
        Loads state_dict into state.model._module and
        state.optimizer._opt (if present). Sets state.current_epoch.
        """
```

**Config fields used:**
- `config.checkpoint_dir` — where checkpoints are written (default: `output_dir/checkpoints`)

**Backward compatibility:** `TorchCheckpoint` only works with `TorchModel`
+ `TorchOptimizer`. If `state.model` is not a `TorchModel`, save raises
`TypeError` with a clear message.

---

## 2. CheckpointHook (`pipeline/hooks/checkpoint.py`)

### Design

Saves checkpoints at configurable intervals from `on_epoch_end`:

```python
class CheckpointHook(BaseHook):
    """Save model checkpoints during training.
    
    Tracks best loss and saves both "best" and "latest" checkpoints.
    Only fires in train mode.
    
    Usage::
    
        pipeline.add_hook(CheckpointHook(save_best_only=False))
    """
    
    def __init__(
        self,
        save_best_only: bool = False,
        monitor: str = "loss",
        mode: str = "min",
    ):
        """Configure checkpointing strategy.
        
        Args:
            save_best_only: If True, only save when monitored metric improves.
            monitor: Metric key in state.history to track.
            mode: "min" (lower is better) or "max" (higher is better).
        """
    
    def on_epoch_end(self, epoch: int, state: PipelineState) -> None:
        # Skip if not training or model/optimizer not set
        # Compute current monitored value from state.history
        # Save "latest" checkpoint
        # If improved, save "best" checkpoint
```

**File naming:**
- `{checkpoint_dir}/latest.pt` — always overwritten each epoch
- `{checkpoint_dir}/best.pt` — only when metric improves

**Edge cases:**
- `state.model` is None → skip (no-op)
- `state.mode != "train"` → skip
- First epoch: always save (no prior best to compare)
- Non-torch model: `TorchCheckpoint.save()` raises `TypeError` → hook catches and logs warning

---

## 3. EarlyStopHook (`pipeline/hooks/early_stop.py`)

### Design

Monitors a metric and sets `state.should_stop = True` when no improvement
for N consecutive epochs.

```python
class EarlyStopHook(BaseHook):
    """Stop training when a metric stops improving.
    
    Usage::
    
        pipeline.add_hook(EarlyStopHook(patience=5, monitor="loss"))
    """
    
    def __init__(
        self,
        patience: int = 10,
        monitor: str = "loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        """Configure early stopping.
        
        Args:
            patience: Epochs without improvement before stopping.
            monitor: Metric key in state.history to watch.
            mode: "min" or "max".
            min_delta: Minimum absolute change to count as improvement.
        """
    
    def on_epoch_end(self, epoch: int, state: PipelineState) -> None:
        # Compute current monitored value
        # Compare to best; if improved, reset counter and update best
        # If not improved for `patience` epochs, set state.should_stop = True
```

**Edge cases:**
- `state.history` is None or missing `monitor` key → skip (no-op)
- Train loop already checks `state.should_stop` before each epoch

---

## 4. Default export implementation (`pipeline/pipeline.py`)

### Design

`export()` changes from `@abstractmethod` to concrete:

```python
def export(self, state: PipelineState) -> None:
    """Stage 6: Write predictions on validation data to CSV.
    
    Default implementation: iterate over validation data stream,
    run model.forward() on each batch, concatenate predictions,
    write to ``{output_dir}/predictions.csv``.
    
    Override for custom export logic (ONNX, multi-file, etc.).
    """
    if state.model is None or state.val_data_stream is None:
        return
    
    state.model.eval_mode()
    all_preds = []
    for batch in state.val_data_stream:
        preds = state.model.forward(batch.inputs)
        all_preds.append(np.asarray(preds))
    
    state.predictions = np.concatenate(all_preds, axis=0)
    
    output_dir = Path(state.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    to_csv(state.predictions, output_dir / "predictions.csv")
```

This eliminates the repetitive prediction-collection code in every subclass.
Subclasses that need custom behavior (ONNX export, checkpoint-only export)
override as before.

---

## 5. load_checkpoint stage (`pipeline/pipeline.py`)

### Design

New method on `BasePipeline`, default no-op:

```python
def load_checkpoint(self, state: PipelineState) -> None:
    """Stage: Load model weights from a checkpoint.
    
    Default is no-op. Override to load weights after build_model
    in infer mode, or for resuming training.
    
    Called after build_model in infer mode. The subclass is
    responsible for choosing the checkpoint path (typically
    from ``state.config.checkpoint_dir``).
    """
```

**Usage in a pipeline subclass:**
```python
def load_checkpoint(self, state):
    path = Path(state.config.checkpoint_dir) / "best.pt"
    TorchCheckpoint.load(state, path)
```

---

## 6. Edge cases and error handling

| Scenario | Behavior |
|----------|----------|
| Infer mode, no checkpoint file | `TorchCheckpoint.load()` raises `FileNotFoundError` with clear message |
| load_checkpoint on sklearn model | No-op if not overridden (default does nothing) |
| CheckpointHook with no model | `on_epoch_end` logs warning, skips |
| EarlyStopHook with no history | `on_epoch_end` skips silently |
| EarlyStop patience exhausted | `state.should_stop = True`, train loop exits cleanly |
| TorchCheckpoint.load with stale architecture | `load_state_dict` raises `RuntimeError` with key mismatch info |

---

## 7. Test strategy

| File | Tests | Key scenarios |
|------|-------|---------------|
| `tests/unit/test_checkpoint.py` | 12+ | save/load round-trip, missing dir auto-create, load into fresh model, wrong model type raises, nonexistent file raises, state dict mismatch, optimizer state restore, epoch restore |
| `tests/unit/test_checkpoint_hook.py` | 10+ | save on epoch end, best-only mode, metric improvement detection, first-epoch always saves, non-train mode skips, missing model skips, file existence after save, best file only updated on improvement |
| `tests/unit/test_early_stop.py` | 10+ | stops after patience exhausted, resets on improvement, no-op when no history, min_delta effect, mode="max" correct, patience=0 edge, negative epoch edge |
| `tests/contract/test_checkpoint_contract.py` | 3+ | TorchCheckpoint round-trip preserves forward output, parameters after load match original, optimizer state restores learning rate |
| `tests/integration/test_checkpoint_e2e.py` | 3+ | full train→save→infer→load pipeline, resume mid-training, early stop triggers checkpoint save |

---

## 8. Files changed/created

```
NEW:
  pipeline/export/checkpoint.py       # TorchCheckpoint
  pipeline/hooks/checkpoint.py        # CheckpointHook
  pipeline/hooks/early_stop.py        # EarlyStopHook
  tests/unit/test_checkpoint.py
  tests/unit/test_checkpoint_hook.py
  tests/unit/test_early_stop.py
  tests/contract/test_checkpoint_contract.py
  tests/integration/test_checkpoint_e2e.py

MODIFIED:
  pipeline/pipeline.py                # export default, load_checkpoint, run() flow
  pipeline/export/__init__.py         # add TorchCheckpoint export
  pipeline/hooks/__init__.py          # add CheckpointHook, EarlyStopHook
  tests/contract/test_model_contract.py  # update infer-mode test (build_model now called)
```

---

## Self-review

- [x] No placeholders — all interfaces fully specified
- [x] No framework import at module level — torch imports lazy inside Checkpoint methods
- [x] Default export is minimal and overridable
- [x] CheckpointHook and EarlyStopHook are independent, compose naturally
- [x] Backward compatible — existing subclasses that override `export()` and don't touch `load_checkpoint` continue to work
- [x] Existing infer-mode tests need update: build_model is now called in infer mode
- [x] load_checkpoint has ~/.pt default no-op, doesn't force new code on existing pipelines
- [x] Phase 5 (HPO + Ensemble) can build on CheckpointHook's "best.pt" tracking

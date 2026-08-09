# Review Remediation Plan — 2026-08-09

Fixes from the codebase + examples review. Scope: **verifiable-now correctness + safe edits**.
Deferred (needs torch install / network / notebook re-run with outputs): notebook pedagogy
uplift (EDA-with-decisions, diagnostics, held-out splits, HPO-in-build_model for m4/m5).

## Constraints (READ FIRST)

- **DO NOT git commit.** This repo tracks progress in markdown, not commits. Implementers
  stop after code + tests pass. The human commits.
- **Offline.** Use `uv run --no-sync <cmd>` (plain `uv run` tries to network-sync and fails).
- **Not installed:** torch, torchvision, transformers, peft. Installed: numpy, pandas,
  sklearn, optuna, matplotlib, seaborn, pyyaml, tqdm, pytest, ruff, mypy.
- Verify each task: `uv run --no-sync pytest <files> -q`, `uv run --no-sync ruff check .`,
  `uv run --no-sync mypy` (mypy may warn on stubs — pre-existing).
- Baseline before work: 415 passed, 10 skipped, 1 flaky-fail (`test_best_only_mode`, an mtime
  tick collision — Task 5 owns making it robust).
- No mocking frameworks. Only fakes/stubs. Teaching code: readable, no metaclasses, no over-abstraction.

## Tasks

### Task 1 — Adam per-parameter step counter (Critical C1) [VERIFIABLE]
`pipeline/training/optimizers.py`. Bug: `self._t` is a single int incremented once per
`update(param)` call, and `NumpyOptimizer.step()` calls `update` once per parameter. So `t`
advances by (n_params) per optimizer step, AND within one step each parameter sees a different
`t` → a different `1-beta1**t` bias-correction factor. This is mathematically wrong Adam.
**Fix:** make `t` per-parameter, keyed by `id(param)` like `_m`/`_v`
(`self._t: dict[int,int] = {}`; in `update`: `self._t[key] = self._t.get(key,0)+1; t = self._t[key]`).
Because each param is updated exactly once per `step()`, per-param `t` equals the global step
count → correct, identical bias correction across params within a step. Update the docstring
(currently line ~69-70 says "The `t` counter is shared across all parameters" — that documents
the bug; correct it).
**Test:** add to `tests/unit/test_optimizers.py` — build a 2-parameter case, run N steps,
assert both parameters used bias-correction factor `1-beta1**k` at step k (e.g. after 1 step
each param's effective `t==1`; after 3 steps `t==3`). A regression test proving t is not
inflated by param count.

### Task 2 — Metrics `average` validation (Important I5) [VERIFIABLE]
`pipeline/evaluation/metrics.py`. Bug: `precision`/`recall`/`f1_score` do
`if average == "binary" and len(classes) <= 2: ... else: <macro>`. So `average="micro"`/
`"weighted"`/typos silently compute macro, and `average="binary"` on 3+ classes silently
downgrades to macro — silent wrong numbers.
**Fix:** at the top of each of the three functions, `if average not in {"binary","macro"}: raise ValueError(...)`.
When `average=="binary"` but `len(classes) > 2`, raise `ValueError` (don't downgrade). Keep the
existing binary/macro math otherwise. Update docstrings to state the raise behavior.
**Test:** extend `tests/unit/test_metrics.py` — assert `ValueError` on `average="micro"`,
and on `average="binary"` with a 3-class input. Keep all existing metric tests green.

### Task 3 — Ensemble robustness (Important I6 + I9) [VERIFIABLE]
`pipeline/hpo/ensemble.py`.
- **I9 VotingEnsemble:** `predict` hard/soft assume every model returns 2D probabilities
  (`np.argmax(p, axis=-1)`). Docstring claims labels are supported; a 1D label array collapses
  to a scalar. **Fix:** normalize each model output — if `p.ndim == 1` treat as labels as-is,
  else `argmax(axis=-1)`; for soft mode require 2D (raise a clear `ValueError` if a model
  returns 1D in soft mode, since you can't average labels). Assert consistent per-model shapes.
- **I6 StackingEnsemble:** `fit` trains the meta-model only inside
  `if hasattr(meta_model,"_estimator") and hasattr(meta_model._estimator,"fit")`, else sets
  `_fitted=True` and trains nothing → silent untrained model; also couples to `SklearnModel`
  private `_estimator`. **Fix:** if the meta-model isn't trainable via that path, raise a clear
  `TypeError`/`ValueError` explaining stacking needs a fittable meta-model (e.g. `SklearnModel`),
  instead of silently succeeding. Keep the sklearn path working.
**Test:** `tests/unit/test_ensemble.py` (create if absent) — Voting with fake models returning
1D labels votes correctly; Stacking with a non-fittable fake meta-model raises. Use existing
fakes from `tests/unit/conftest.py` where possible.

### Task 4 — OptunaSearch int ranges (Important I7) [VERIFIABLE]
`pipeline/hpo/search.py`. Bug: a `(low, high)` tuple always routes to `trial.suggest_float`,
so `{"batch_size": (8, 128)}` yields `63.4` into an int-expecting `Config.batch_size`. No way
to express an int range.
**Fix:** in the Optuna objective search-space dispatch, detect `tuple[int,int]` → `suggest_int`,
`tuple[float,float]` (any float endpoint) → `suggest_float`, `list` → `suggest_categorical`.
Update the docstring example. optuna import stays lazy (inside the function).
**Test:** `tests/unit/test_hpo_search.py` (or existing optuna test file) — an int-range space
yields integer params across trials; a float-range yields floats; categorical unchanged.
Mark optuna tests to skip cleanly if optuna missing (it's installed here, so they run).

### Task 5 — Hook epoch aggregation + missing-key warning (Important I3 + I4) [VERIFIABLE]
`pipeline/hooks/early_stop.py`, `pipeline/hooks/checkpoint.py`.
- **I4:** both hooks read `state.history[monitor][-1]` at `on_epoch_end`. But `history["loss"]`
  is one float **per batch** — `[-1]` is the last noisy batch of the epoch, not an epoch
  aggregate. **Fix:** compute the epoch's loss as the mean of that epoch's batch slice. The
  clean approach: have the hook track how many loss entries it has already consumed and average
  only the new entries appended during the just-finished epoch (per-epoch mean), rather than the
  whole history or the last element. Keep it simple and documented.
- **I3:** default `monitor` in docstrings/examples is `"val_loss"` but only `"loss"` is ever
  populated by `TrainLoop`, so the hook is a permanent silent no-op. **Fix:** when `monitor`
  is absent from `state.history` at `on_epoch_end`, log a `warning` **once** (guard with a flag)
  naming the missing key and the available keys — then continue (don't crash; hooks never crash).
- This also fixes the flaky `tests/unit/test_checkpoint_hook.py::test_best_only_mode` (two saves
  in one mtime tick). Make the CheckpointHook "best only" decision robust: it should save on
  strict improvement of the epoch-mean metric; the test should assert *content/decision* (saved
  vs not-saved) rather than mtime ordering, OR the hook must genuinely skip the write when not
  improved. Prefer fixing the hook so no write happens when not improved, then the test can
  assert the file is unchanged by comparing bytes/mtime reliably. If the test asserts mtime and
  that's inherently flaky, adjust the test to assert on save-count / file content instead.
**Test:** existing `test_early_stop*.py` and `test_checkpoint_hook.py` stay green; add a test
that a monitored-but-absent key logs a warning and does not crash; add a test that epoch-mean
(not last-batch) drives the decision (feed a history where last-batch vs epoch-mean differ).

### Task 6 — Correctness minors bundle [VERIFIABLE]
Small independent fixes:
- `pipeline/data/csv_source.py` and `pipeline/data/text_source.py`: `__len__` uses
  `max(1, ceil(n/batch))`, reporting 1 batch for an empty source while `__iter__` yields 0.
  **Fix:** drop the `max(1, ...)` so empty → 0 (`ceil(0/32)==0`). Verify non-empty unchanged.
- `pipeline/data/image_folder.py` (~line 160): globs both `*ext` and `*EXT`, double-counting on
  case-insensitive filesystems (WSL2 DrvFs). **Fix:** collect into a `set()` of resolved paths
  and sort, so each file counts once. (No torch needed — this is pure pathlib/PIL-free listing;
  if listing needs PIL, guard the test to skip without PIL.)
- `pipeline/training/optimizers.py`: `SGD.update`/`Adam.update` use `assert param.grad is not None`
  as a runtime guard — stripped under `python -O`. **Fix:** replace with
  `if param.grad is None: raise ValueError(f"Parameter {param.name} has no gradient")`.
**Test:** add/extend unit tests — empty CsvDataSource has `len==0`; image folder with mixed-case
duplicate-name files counts each once (or a targeted test of the dedupe helper); SGD/Adam raise
`ValueError` (not AssertionError) on missing grad.

### Task 7 — Torch adapter + checkpoint fixes (Important I2 + I8 + dead code) [UNVERIFIED OFFLINE]
`pipeline/adapters/torch_adapter.py`, `pipeline/export/checkpoint.py`. **torch not installed** —
implement carefully, add tests marked to skip without torch, and REPORT this as
DONE_WITH_CONCERNS (unverified by execution here).
- **I2 device bridging:** `TorchModel.forward` does `torch.as_tensor(inputs)` (CPU) and never
  moves to the module's device; same in `TorchLoss._backward`. **Fix:** capture the module
  device (`next(self._module.parameters()).device`, fall back to CPU if no params) and
  `x = x.to(device)` in both forward and the backward re-run.
- **I8 checkpoint load symmetry:** `save` raises `TypeError` if `state.model` isn't a
  `TorchModel`, but `load` silently does nothing (guarded `if ... isinstance`). **Fix:** make
  `load` raise the same `TypeError` when `state.model` is None or not a `TorchModel`, so a
  forgotten build_model doesn't silently load an untrained model.
- **Dead code:** `TorchOptimizer.zero_grad` has a second loop over `self._params` that can never
  fire (stale `None` snapshots captured at construction); `self._opt.zero_grad(set_to_none=False)`
  already zeros the live tensors. **Fix:** delete the dead loop and the docstring paragraph that
  claims it zeros Parameter grad fields.
**Test:** add torch tests guarded by `pytest.importorskip("torch")` — device move (CPU is fine
to assert the tensor lands on the module's device), load raises on wrong model type. These will
`skip` in this offline env; that's expected.

### Task 8 — forward() contract docs + declare transformers/peft (I1 + deps) [VERIFIABLE]
- **I1 docs:** `pipeline/protocols.py` `ModelProtocol.forward` only says "return predictions".
  Adapters disagree: sklearn→probabilities `(N,C)`, numpy/torch→logits. **Fix (docs only, no
  behavior change):** document the convention explicitly in `ModelProtocol.forward` docstring —
  classifiers return class scores shape `(N, C)` and callers argmax over axis -1; note that
  sklearn returns normalized probabilities while numpy/torch return raw logits (argmax-equivalent).
  Add a one-line note to each adapter's `forward` docstring stating what it returns. This is a
  Google-style docstring edit.
- **deps:** `pyproject.toml` — add a new optional extra `llm = ["transformers>=4.40.0", "peft>=0.10.0"]`
  (the m6 example + serve.py need them and they are NOT in `dev`). Do not add to core or dev.
  Note in the extra's context that it's for the m6 LLM example.
**Verify:** `uv run --no-sync ruff check .`; `uv run --no-sync python -c "import pipeline"`
still imports without torch/transformers (lazy-import invariant holds). No new runtime deps in core.

### Task 9 — Notebook bug fixes: m2/m3 plot + m6 honesty [UNVERIFIED — source edit only]
Cannot re-run (needs torch/HF/downloads). Fix SOURCE and CLEAR the outputs of only the cells you
change (set outputs `[]`, execution_count `null`) so no stale/misleading output remains next to
corrected source. Leave untouched cells' outputs intact.
- `examples/m2-m3-numpy/gradient_to_mlp.ipynb`: cell 6 reassigns `losses` to a `y=2x` MSE demo;
  cell 7 then plots `losses` titled "Binary Cross-Entropy Loss" — wrong data under wrong label.
  **Fix:** rename the Act-1 BCE training loss variable to `bce_losses` in its source cell (~cell 5)
  and plot that in cell 7 with the correct title; either remove cell 6's `losses` reassignment or
  rename its variable so it no longer shadows. Clear outputs of the cells you edit.
- `examples/m6-llm-distilgpt2/distilgpt2.ipynb`: (a) remove the dead `SimpleTokenizer` block
  (fits + prints a vocab that's never used; the real path uses AutoTokenizer) — or replace with a
  one-line honest markdown note explaining HF tokenizer is required to match pretrained embeddings.
  (b) Fix the false "held-out prompts" claim (3 of 5 test prompts are verbatim in
  `tiny_instruct.csv`): reword the markdown + wrap-up to "prompts from the training distribution"
  (or swap in genuinely unseen prompts). Clear outputs of edited cells.
**Verify:** `uv run --no-sync python -c "import json,glob; [json.load(open(f)) for f in glob.glob('examples/**/*.ipynb',recursive=True)]"` (notebooks remain valid JSON). Report UNVERIFIED (not executed).

### Task 10 — Notebook reproducibility + state-bus [UNVERIFIED — source edit only]
Cannot re-run. Fix source; clear outputs of edited cells.
- `examples/m4a-torch-mnist/mnist.ipynb`, `examples/m4b-torch-cifar10/cifar10.ipynb`,
  `examples/m5-text-imdb/imdb.ipynb`:
  - Call `set_seed(config.seed, deterministic=True)` (from `pipeline.utils.seed`) at the start of
    `build_model` (before constructing the torch module), and construct the model **inside**
    `build_model` rather than as a module-level global cell. This fixes both the reproducibility
    hole and the state-bus violation in one move.
  - In `evaluate`/`export`, reference `state.model` (not the module-global `model`).
**Verify:** notebooks remain valid JSON (as Task 9). Report UNVERIFIED (not executed).

## Out of scope (deferred — needs re-run/torch/network)
- EDA-with-Read/Decision, confusion/ROC/per-class/gen-gap diagnostics, held-out third splits,
  HPO-inside-build_model for m4a/m4b/m5.
- m6 conversion to a `BasePipeline` subclass with a custom `train()` (larger design change).
- TorchLoss double-forward refactor (over-engineering item; leave as-is, it works).

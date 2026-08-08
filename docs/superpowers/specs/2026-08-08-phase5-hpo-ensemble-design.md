# Phase 5 Design — HPO + Ensemble

Date: 2026-08-08 | Status: draft

## Scope

Add hyperparameter optimization (grid search, random search) and ensemble
methods (voting, stacking) to the pipeline. Zero new framework dependencies
— operates on existing Config, BasePipeline, Metrics, and registry infrastructure.

## Architecture

```
pipeline/
├── hpo/
│   ├── __init__.py       # exports
│   ├── search.py         # GridSearch, RandomSearch
│   └── ensemble.py       # VotingEnsemble, StackingEnsemble
```

No changes to existing pipeline/ modules. HPO is a consumer of the pipeline,
not integrated into it. The pipeline trains one model; HPO trains many.

---

## 1. HPO Search (`pipeline/hpo/search.py`)

### Design principle

HPO is NOT a pipeline stage. It's a standalone orchestrator that creates
multiple pipeline instances with different Configs. Each trial is a full
`pipeline.run("train")` call.

### Search space

A search space is a dict mapping Config field names to lists of values:

```python
param_grid = {
    "learning_rate": [0.1, 0.01, 0.001],
    "batch_size": [16, 32, 64],
    "num_epochs": [5, 10],
}
```

### Base class

```python
class BaseSearch(ABC):
    """Abstract interface for hyperparameter search.
    
    Subclasses implement _generate_configs() to produce a sequence
    of (trial_id, Config) pairs. run() iterates through them,
    trains a pipeline for each, and records results.
    """
    
    def __init__(self, pipeline_cls: type, base_config: Config, scoring: str = "accuracy"):
        """pipeline_cls: a BasePipeline subclass (not instance)."""
    
    @abstractmethod
    def _generate_configs(self) -> Iterator[tuple[int, Config]]: ...
    
    def run(self) -> SearchResult:
        """Train one model per config, return sorted results."""
```

### GridSearch

```python
class GridSearch(BaseSearch):
    """Exhaustive search over the Cartesian product of a param grid.
    
    Usage::
    
        search = GridSearch(MyPipeline, base_config, param_grid)
        result = search.run()
        print(result.best_config, result.best_score)
    """
    
    def __init__(self, pipeline_cls, base_config, param_grid: dict[str, list], scoring="accuracy"): ...
```

Takes the Cartesian product of all values in `param_grid`, merges each combination into a copy of `base_config`, yields them.

### RandomSearch

```python
class RandomSearch(BaseSearch):
    """Sample N random configs from a param grid.
    
    Usage::
    
        search = RandomSearch(MyPipeline, base_config, param_grid, n_trials=10, seed=42)
    """
    
    def __init__(self, pipeline_cls, base_config, param_grid, n_trials=10, scoring="accuracy", seed=42): ...
```

Randomly samples `n_trials` combinations (with replacement? no — sampling without replacement, min(n_trials, total_combinations)).

### SearchResult

```python
@dataclass
class TrialResult:
    trial_id: int
    config: Config
    metrics: dict[str, float]
    
@dataclass  
class SearchResult:
    trials: list[TrialResult]
    best_trial: TrialResult
    best_config: Config
    best_score: float
    
    def to_dataframe(self) -> "pd.DataFrame": ...  # for analysis
```

---

## 2. Ensemble (`pipeline/hpo/ensemble.py`)

### Design principle

Ensemble takes already-trained models (from `PipelineState` instances after `.run("train")`) and combines their predictions. HPO search can feed into ensemble: the top-K trials become the base models.

### VotingEnsemble

```python
class VotingEnsemble:
    """Combine predictions by majority vote (classification) or mean (regression).
    
    Usage::
    
        ensemble = VotingEnsemble(models=[state1.model, state2.model], mode="hard")
        predictions = ensemble.predict(inputs)
    """
    
    def __init__(self, models: list[ModelProtocol], mode: str = "hard"):
        """mode: "hard" (majority vote) or "soft" (average probabilities)."""
    
    def predict(self, inputs: ArrayLike) -> np.ndarray:
        """Run all models, aggregate predictions."""
    
    def evaluate(self, data_stream: DataStream, metrics: Metrics) -> Metrics:
        """Evaluate ensemble on a data stream."""
```

### StackingEnsemble

```python
class StackingEnsemble:
    """Train a meta-model on base model predictions.
    
    Base models produce predictions on training data. A meta-model
    (e.g., sklearn LogisticRegression or another pipeline model)
    learns to combine them.
    
    Usage::
    
        stacking = StackingEnsemble(
            base_models=[state1.model, state2.model],
            meta_model=SklearnModel(LogisticRegression()),
        )
        stacking.fit(train_data_stream)
        predictions = stacking.predict(test_inputs)
    """
    
    def __init__(self, base_models: list[ModelProtocol], meta_model: ModelProtocol): ...
    
    def fit(self, data_stream: DataStream) -> "StackingEnsemble":
        """Train meta-model on base model outputs."""
    
    def predict(self, inputs: ArrayLike) -> np.ndarray: ...
```

Key: `fit()` iterates the data stream, runs each base model, stacks their outputs as features, fits the meta-model on those features.

---

## 3. Integration patterns

### HPO → Best model → export

```python
search = GridSearch(MyPipeline, base_config, param_grid)
result = search.run()
best_pipeline = MyPipeline(result.best_config)
best_pipeline.run("train")
best_pipeline.run("infer")
```

### HPO → Top-K models → ensemble

```python
search = RandomSearch(MyPipeline, base_config, param_grid, n_trials=20)
result = search.run()
top_models = [trial_to_model(t) for t in result.top_k(3)]
ensemble = VotingEnsemble(top_models, mode="soft")
ensemble.evaluate(val_stream, metrics)
```

---

## 4. Edge cases

| Scenario | Behavior |
|----------|----------|
| Empty param_grid | GridSearch: ValueError. RandomSearch: ValueError. |
| n_trials > total combinations | RandomSearch: clamp to total (warn) |
| Pipeline subclass raises during trial | Catch, record as failed trial (score=0), continue |
| Single model in ensemble | Voting: returns same predictions. Stacking: meta-model receives one feature. |
| Different model types in ensemble | Works as long as all satisfy ModelProtocol |
| Classification vs regression | Voting mode="hard" requires class labels (argmax). mode="soft" averages probabilities. |
| No validation data | Scoring uses train loss from state.history |

---

## 5. Test strategy

| File | Tests | Key scenarios |
|------|-------|---------------|
| `tests/unit/test_grid_search.py` | 8+ | Cartesian product, config merge, run completes, best trial selection, empty grid raises, single trial, scoring from metrics |
| `tests/unit/test_random_search.py` | 6+ | n_trials count, reproducibility (seed), clamp n_trials, run completes, best trial, different configs per trial |
| `tests/unit/test_voting_ensemble.py` | 8+ | hard voting (majority), soft voting (avg prob), single model fallback, different model types together, evaluate on stream, predict shape |
| `tests/unit/test_stacking_ensemble.py` | 6+ | fit + predict, meta-model training, base model outputs as features, predict shape, cross-validation warning |
| `tests/integration/test_hpo_e2e.py` | 3+ | GridSearch on fake pipeline, RandomSearch on fake pipeline, VotingEnsemble from search results |

---

## 6. Files created

```
NEW:
  pipeline/hpo/__init__.py
  pipeline/hpo/search.py          # BaseSearch, GridSearch, RandomSearch, SearchResult, TrialResult
  pipeline/hpo/ensemble.py        # VotingEnsemble, StackingEnsemble
  tests/unit/test_search.py       # GridSearch + RandomSearch tests
  tests/unit/test_ensemble.py     # VotingEnsemble + StackingEnsemble tests
  tests/integration/test_hpo_e2e.py

MODIFIED: (none required — HPO is standalone consumer of existing API)
```

---

## Self-review

- [x] Zero new framework dependencies — pure Python + numpy
- [x] Follows existing patterns: registry-friendly, @dataclass results, Google docstrings
- [x] HPO search is NOT a pipeline stage — it orchestrates pipeline instances externally
- [x] Ensemble works with any ModelProtocol — framework-agnostic by construction
- [x] SearchResult.to_dataframe() uses lazy pandas import
- [x] Trial failures don't crash the entire search
- [x] RandomSearch sampling without replacement avoids duplicate trials
- [x] VotingEnsemble hard/soft modes cover classification (argmax) and probability averaging
- [x] StackingEnsemble fit() pattern mirrors sklearn stacking

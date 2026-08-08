# HPO

## Search

::: pipeline.hpo.search
    options:
      members:
        - BaseSearch
        - GridSearch
        - RandomSearch
        - SearchResult
        - TrialResult

## Ensemble

::: pipeline.hpo.ensemble
    options:
      members:
        - VotingEnsemble
        - StackingEnsemble

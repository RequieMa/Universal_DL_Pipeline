"""CLI entry point for the Universal DL Pipeline.

Usage::

    python run.py --config config.yaml --mode train
    python run.py --config config.yaml --mode infer
    python run.py --help
"""
from __future__ import annotations

import argparse
import sys

from pipeline.config import Config
from pipeline.evaluation import Metrics, accuracy
from pipeline.pipeline import BasePipeline


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the pipeline.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description="Universal DL Pipeline -- train and infer"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        default="train",
        help="Pipeline mode (default: train)",
    )
    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config)

    # Phase 1: use a minimal pipeline with default stages.
    # Students subclass BasePipeline for custom behavior.
    class DefaultPipeline(BasePipeline):
        """Phase 1 default pipeline using TrainLoop + CsvDataSource."""

        def load_data(self, state):
            from pipeline.data.csv_source import CsvDataSource
            from pipeline.data.split import train_test_split

            full_stream = CsvDataSource(
                f"{config.data_dir}/train.csv",
                batch_size=config.batch_size,
            )
            train_stream, val_stream = train_test_split(
                full_stream,
                train_ratio=config.train_ratio,
            )
            state.data_stream = train_stream
            state.val_data_stream = val_stream

        def build_model(self, state):
            # Phase 1: fake model for framework validation.
            # Real adapters arrive in Phase 2.
            import numpy as np

            class _FakeModel:
                def forward(self, inputs):
                    return np.asarray(inputs)[:, :1]

                def parameters(self):
                    return []

                def train_mode(self):
                    pass

                def eval_mode(self):
                    pass

            class _FakeLoss:
                def forward(self, p, t):
                    from pipeline.protocols import Loss

                    return Loss(
                        float(np.mean((np.asarray(p) - np.asarray(t)) ** 2))
                    )

                def __call__(self, p, t):
                    return self.forward(p, t)

            class _FakeOptimizer:
                def step(self):
                    pass

                def zero_grad(self):
                    pass

            state.model = _FakeModel()
            state.loss_fn = _FakeLoss()
            state.optimizer = _FakeOptimizer()
            state.metrics = Metrics(accuracy=accuracy)

        def evaluate(self, state):
            import numpy as np

            state.model.eval_mode()
            all_preds, all_targets = [], []
            for batch in state.val_data_stream:
                preds = np.asarray(state.model.forward(batch.inputs))
                all_preds.append(preds.reshape(-1))
                all_targets.append(np.asarray(batch.targets))
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            state.metrics.compute(y_true, y_pred)

        def export(self, state):
            import numpy as np

            from pipeline.export.to_csv import to_csv

            if state.model is not None and state.val_data_stream is not None:
                preds = []
                for batch in state.val_data_stream:
                    p = np.asarray(state.model.forward(batch.inputs))
                    preds.append(p.reshape(-1))
                state.predictions = np.concatenate(preds)
                # to_csv reshapes 1D arrays as rows; reshape to (N, 1)
                # so column names match. Pattern from E2E test.
                col_array = state.predictions.reshape(-1, 1)
            else:
                # Infer mode: no model available; write placeholder output.
                state.predictions = np.array([0.0])
                col_array = state.predictions.reshape(-1, 1)
            to_csv(
                col_array,
                f"{config.output_dir}/predictions.csv",
                columns=["prediction"],
            )

    pipeline = DefaultPipeline(config)
    state = pipeline.run(args.mode)
    print(f"Pipeline completed. Mode: {args.mode}")


if __name__ == "__main__":
    main()

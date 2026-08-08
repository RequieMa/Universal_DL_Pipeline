"""Integration tests: full pipeline with TorchAdapter.

E2E tests covering:
    1. The full six-stage pipeline in train mode using a TorchModel on tabular CSV data.
    2. Infer mode, which skips training stages (build_model / train / evaluate).
    3. An image-folder E2E: ImageFolderDataSource -> torch transforms -> TorchModel,
       training one epoch on synthetic image batches.

All torch imports are lazy (inside functions) so this file imports cleanly without
PyTorch installed. Tests are gated behind ``requires_torch`` (skipif) and marked
``slow`` per project conventions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


@requires_torch
@pytest.mark.slow
class TestTorchPipelineE2E:
    """Full pipeline with TorchAdapter on tabular data."""

    def test_full_pipeline_train_mode(self, tmp_path: Path) -> None:
        """Train mode: all 6 stages complete with TorchAdapter on CSV data."""
        import pandas as pd
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchLoss, TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.data.csv_source import CsvDataSource
        from pipeline.data.split import train_test_split
        from pipeline.evaluation.metrics import Metrics, accuracy
        from pipeline.pipeline import BasePipeline, PipelineState

        # Build synthetic CSV
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame(
            {
                "a": np.random.randn(20),
                "b": np.random.randn(20),
                "c": np.random.randn(20),
                "label": np.random.randint(0, 2, 20),
            }
        )
        df.to_csv(csv_path, index=False)

        config = Config(
            batch_size=4,
            num_epochs=2,
            train_ratio=0.75,
            learning_rate=0.1,
            seed=42,
            output_dir=str(tmp_path / "output"),
        )

        class TestPipeline(BasePipeline):
            def load_data(self, state: PipelineState) -> None:
                source = CsvDataSource(
                    str(csv_path), batch_size=config.batch_size, shuffle=False
                )
                train, val = train_test_split(source, train_ratio=config.train_ratio)
                state.data_stream = train
                state.val_data_stream = val

            def build_model(self, state: PipelineState) -> None:
                state.model = TorchModel(nn.Linear(3, 2))
                # WHY model=<TorchModel>: CsvDataSource yields numpy arrays, so
                # TorchModel.forward returns a *detached* numpy output. Passing the
                # model rebinds loss.backward() to re-run the module on the cached
                # input, rebuilding the autograd graph so gradients reach parameters.
                state.loss_fn = TorchLoss("CrossEntropyLoss", model=state.model)
                state.optimizer = TorchOptimizer(
                    state.model.parameters(), "SGD", lr=config.learning_rate
                )
                state.metrics = Metrics(accuracy=accuracy)

            def evaluate(self, state: PipelineState) -> None:
                state.model.eval_mode()
                all_preds, all_targets = [], []
                for batch in state.val_data_stream:
                    logits = np.asarray(state.model.forward(batch.inputs))
                    all_preds.append(np.argmax(logits, axis=1))
                    all_targets.append(np.asarray(batch.targets))
                y_pred = np.concatenate(all_preds)
                y_true = np.concatenate(all_targets)
                state.metrics.compute(y_true, y_pred)

            def export(self, state: PipelineState) -> None:
                state.predictions = np.array([0])

        pipeline = TestPipeline(config)
        state = pipeline.run("train")

        assert state.history is not None
        assert len(state.history["loss"]) > 0
        assert "loss" in state.history
        assert "accuracy" in state.metrics
        assert state.predictions is not None

    def test_infer_mode_skips_training(self, tmp_path: Path) -> None:
        """Infer mode only runs load_data + export."""
        import pandas as pd
        import torch.nn as nn

        from pipeline.adapters.torch_adapter import TorchLoss, TorchModel, TorchOptimizer
        from pipeline.config import Config
        from pipeline.data.csv_source import CsvDataSource
        from pipeline.evaluation.metrics import Metrics, accuracy
        from pipeline.pipeline import BasePipeline, PipelineState

        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "label": [0, 1, 0, 1],
            }
        )
        df.to_csv(csv_path, index=False)

        config = Config(
            batch_size=2,
            num_epochs=5,
            train_ratio=0.75,
            learning_rate=0.1,
            seed=42,
            output_dir=str(tmp_path / "output"),
        )

        class InferPipeline(BasePipeline):
            def load_data(self, state: PipelineState) -> None:
                state.data_stream = CsvDataSource(
                    str(csv_path), batch_size=config.batch_size, shuffle=False
                )

            def build_model(self, state: PipelineState) -> None:
                state.model = TorchModel(nn.Linear(1, 2))
                state.loss_fn = TorchLoss("CrossEntropyLoss")
                state.optimizer = TorchOptimizer(
                    state.model.parameters(), "SGD", lr=config.learning_rate
                )
                state.metrics = Metrics(accuracy=accuracy)

            def evaluate(self, state: PipelineState) -> None:
                pass

            def export(self, state: PipelineState) -> None:
                state.predictions = np.array([0, 0])

        pipeline = InferPipeline(config)
        state = pipeline.run("infer")

        # Infer mode: model is None (build_model not called), history is None
        assert state.model is None
        assert state.history is None
        assert state.predictions is not None


@requires_torch
@pytest.mark.slow
class TestImageFolderToTorchE2E:
    """ImageFolderDataSource -> TorchAdapter end-to-end."""

    def test_image_folder_to_torch_one_epoch(self, tmp_path: Path) -> None:
        """Train one epoch: ImageFolderDataSource -> torch transforms -> TorchModel.

        NOTE: images are created inline in tmp_path rather than via the
        ``tiny_image_folder`` conftest fixture (that lives in tests/unit/ and is
        not visible from tests/integration/).
        """
        import torch.nn as nn
        from PIL import Image

        from pipeline.adapters.torch_adapter import TorchLoss, TorchModel, TorchOptimizer
        from pipeline.data.image_folder import ImageFolderDataSource
        from pipeline.data.transforms import TransformedDataStream
        from pipeline.protocols import Batch

        # Create 2 classes x 3 images of 4x4 RGB in tmp_path
        for class_name in ["cat", "dog"]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(3):
                img_data = bytes([i * 40] * (4 * 4 * 3))
                img = Image.frombytes("RGB", (4, 4), img_data)
                img.save(class_dir / f"{i:02d}.png")

        # Build data source (6 images, batch_size=2 -> 3 batches)
        source = ImageFolderDataSource(tmp_path, batch_size=2, shuffle=False)

        # Build torch transform: PIL images -> normalized float tensors
        def to_tensor(batch: Batch) -> Batch:
            import torch
            import torchvision.transforms as T  # noqa: N812

            to_tensor_fn = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            imgs = torch.stack([to_tensor_fn(img) for img in batch.inputs])
            return Batch(inputs=imgs, targets=torch.as_tensor(batch.targets))

        stream = TransformedDataStream(source, to_tensor)

        # Build model: conv -> pool -> linear for 2 classes
        conv = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 2),
        )
        model = TorchModel(conv)
        opt = TorchOptimizer(model.parameters(), "SGD", lr=0.01)
        # Tensor inputs keep the autograd graph intact, so no model= rebinding needed.
        loss_fn = TorchLoss("CrossEntropyLoss")

        # Train one epoch
        model.train_mode()
        losses = []
        for batch in stream:
            preds = model.forward(batch.inputs)
            loss = loss_fn(preds, batch.targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))

        assert len(losses) > 0  # at least one batch processed
        assert all(isinstance(loss_val, float) for loss_val in losses)

import abc
import functools
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from typing_extensions import Literal

from tfh_train.utils.metrics_collection_wrapper import MetricsCollectionWrapper


class BaseLightningModelModule(L.LightningModule):
    """Base lightning model module that wraps boilerplate code."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: functools.partial,
        scheduler: Optional[functools.partial],
        train_metric: Union[torchmetrics.Metric, torchmetrics.MetricCollection, MetricsCollectionWrapper],
        test_metric: Union[torchmetrics.Metric, torchmetrics.MetricCollection, MetricsCollectionWrapper],
        compile_mode: Optional[str] = None,
    ) -> None:
        """Assign parameters.

        Args:
            model (nn.Module): pytorch model.
            criterion (nn.Module): Loss function.
            optimizer (functools.partial): Training init method optimizer.
            scheduler (Optional[functools.partial]): Training scheduler.
            train_metric (Union[torchmetrics.Metric, torchmetrics.MetricCollection, MetricsCollectionWrapper]): Train metric or metrics collection.
            test_metric (Union[torchmetrics.Metric, torchmetrics.MetricCollection, MetricsCollectionWrapper]): Test/Validation metrics or metrics collection.
            compile_mode (Optional[str], optional): Should apply PyTorch 2.0 `compile` mechanism and in which mode. If None mechanism isn't applied. Defaults to None.

        References:
            [1] https://pytorch.org/get-started/pytorch-2.0/#user-experience
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion", "metric", "train_metric", "test_metric"], logger=False)

        self.neural_net = model
        if compile_mode is not None:
            self.neural_net = torch.compile(self.neural_net, mode=compile_mode)
        self.criterion = criterion

        self.training_step_buffer: List[Dict[str, torch.Tensor]] = []
        self.validation_step_buffer: List[Dict[str, torch.Tensor]] = []
        self.testing_step_buffer: List[Dict[str, torch.Tensor]] = []

        self.train_metric = deepcopy(train_metric)
        self.validation_metric = deepcopy(test_metric)
        self.test_metric = deepcopy(test_metric)

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Optional[L.pytorch.utilities.types.STEP_OUTPUT]:
        """Single training step.

        Args:
            batch (torch.Tensor): Input batch.
            batch_idx (int): Batch index.

        Returns:
            Optional[L.pytorch.utilities.types.STEP_OUTPUT]: Optional output or outputs dictionary.
        """
        step_results = self._step(batch, stage_metric=self.train_metric)
        self.training_step_buffer.append(step_results)

        self.log("train_step_loss", step_results["loss"].item(), on_step=True)

        return step_results

    def on_train_epoch_end(self) -> None:
        """On train epoch ends hook."""
        self._summarize_epoch(stage="train", outputs=self.training_step_buffer, stage_metric=self.train_metric)
        self.training_step_buffer.clear()

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Optional[L.pytorch.utilities.types.STEP_OUTPUT]:
        """Single validation step.

        Args:
            batch (List[torch.Tensor]): Input batch.
            batch_idx (int): Batch index.

        Returns:
            Optional[L.pytorch.utilities.types.STEP_OUTPUT]: Optional output or outputs dictionary.
        """
        step_results = self._step(batch, stage_metric=self.validation_metric)
        self.validation_step_buffer.append(step_results)

        return step_results

    def on_validation_epoch_end(self) -> None:
        """On validation epoch ends hook."""
        self._summarize_epoch(
            stage="validation", outputs=self.validation_step_buffer, stage_metric=self.validation_metric
        )
        self.validation_step_buffer.clear()

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Optional[L.pytorch.utilities.types.STEP_OUTPUT]:
        """Single test step.

        Args:
            batch (List[torch.Tensor]): Input batch.
            batch_idx (int): Batch index.

        Returns:
            Optional[L.pytorch.utilities.types.STEP_OUTPUT]: Optional output or outputs dictionary.
        """
        step_results = self._step(batch, stage_metric=self.test_metric)
        self.testing_step_buffer.append(step_results)

        return step_results

    def on_test_epoch_end(self) -> None:
        """On test epoch ends hook."""
        self._summarize_epoch(stage="test", outputs=self.testing_step_buffer, stage_metric=self.test_metric)
        self.testing_step_buffer.clear()

    def on_train_start(self) -> None:
        """On train start hook. JIC clean validation mechanisms."""
        self.validation_step_buffer.clear()
        self.validation_metric.reset()

    @abc.abstractmethod
    def configure_optimizers(
        self,
    ) -> Any:
        """Configure optimizer.

        Returns:
            Any: Optimizer and optional lr scheduler objects wraps into output processable by PyTorch Lightning.
        """
        pass

    @abc.abstractmethod
    def _step(
        self,
        batch: Any,
        stage_metric: Union[torchmetrics.Metric, torchmetrics.MetricCollection, MetricsCollectionWrapper],
    ) -> L.pytorch.utilities.types.STEP_OUTPUT:
        """Single step.

        Args:
            batch (Any): Input batch.
            stage_metric (Union[torchmetrics.Metric, torchmetrics.MetricCollection, MetricsCollectionWrapper]): Stage metric.

        Returns:
            L.pytorch.utilities.types.STEP_OUTPUT: Output dictionary.
        """
        pass

    @abc.abstractmethod
    def _summarize_epoch(
        self,
        stage: Literal["train", "validation", "test"],
        outputs: List[Dict[str, torch.Tensor]],
        stage_metric: Union[torchmetrics.Metric, torchmetrics.MetricCollection, MetricsCollectionWrapper],
    ) -> None:
        """Summarize epoch computations.

        Args:
            stage (Literal["train", "validation", "test"]): Log prefix.
            outputs (List[Dict[str, torch.Tensor]]): Steps outputs.
            stage_metric (Union[torchmetrics.Metric, torchmetrics.MetricCollection, MetricsCollectionWrapper]): Stage metric.
        """
        pass

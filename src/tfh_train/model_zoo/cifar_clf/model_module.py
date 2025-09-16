import functools
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from jaxtyping import Num
from typing_extensions import Literal

from tfh_train.model_zoo.common.base_lightning_model_module import BaseLightningModelModule
from tfh_train.utils.metrics_collection_wrapper import MetricsCollectionWrapper

class CifarClassifierTraining(BaseLightningModelModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: functools.partial,
    ) -> None:
        """Assign parameters.

        Args:
            model (nn.Module): pytorch model.
            criterion (nn.Module): Loss function.
            optimizer (functools.partial): Training init method optimizer.
            """
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            train_metric=torchmetrics.Accuracy(task="multiclass", num_classes=10),
            test_metric=torchmetrics.Accuracy(task="multiclass", num_classes=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        return self.neural_net(x)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        """Configure optimizer.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]: Optimizer and lr scheduler objects.
        """
        optimizer = self.hparams.optimizer(params=self.neural_net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer)
            return [optimizer], [scheduler]

        return [optimizer], []

    def _step(
        self,
        batch: List[Num[torch.Tensor, "batch *c h w"]],
        stage_metric: Union[torchmetrics.Metric, torchmetrics.MetricCollection, MetricsCollectionWrapper],
    ) -> L.pytorch.utilities.types.STEP_OUTPUT:
        """Single step.

        Args:
            batch (List[Num[torch.Tensor, "batch *c h w"): Input batch.
            stage_metric (Union[torchmetrics.Metric, torchmetrics.MetricCollection, MetricsCollectionWrapper]): Stage metric.

        Returns:
            L.pytorch.utilities.types.STEP_OUTPUT: Output dictionary.
        """
        image, true_labels = batch

        pred_logits = self.neural_net(image)

        loss = self.criterion(pred_logits, true_labels)

        stage_metric(pred_logits, true_labels)

        return {"loss": loss}

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
        mean_loss = torch.tensor([out["loss"] for out in outputs]).mean()
        self.log(f"{stage}_loss", mean_loss, on_epoch=True)

        metrics = stage_metric.compute()

        if isinstance(stage_metric, (torchmetrics.MetricCollection, MetricsCollectionWrapper)):
            for metric_name, metric_value in metrics.items():
                metric_value = metric_value.detach().item()
                self.log(f"{stage}_{metric_name}", metric_value, on_epoch=True)
        elif isinstance(stage_metric, torchmetrics.Metric):
            metric_value = metrics.detach().item()
            metric_name = stage_metric.__class__.__name__
            self.log(f"{stage}_{metric_name}", metric_value, on_epoch=True)

        stage_metric.reset()

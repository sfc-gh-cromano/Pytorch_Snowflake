from dataclasses import dataclass
from typing import Iterable

import torchmetrics


@dataclass
class MetricObjectWrapper:
    """Wrapper for torchmetrics.Metric object with corresponding name."""

    name: str
    metric_object: torchmetrics.Metric


class MetricsCollectionWrapper(torchmetrics.MetricCollection):
    """torchmetrics.MetricCollection that provides different way of instantiating torchmetrics.MetricCollection object which suits more hydra config parser."""

    def __init__(self, metrics: Iterable[MetricObjectWrapper]) -> None:
        """Assign paramters.

        Args:
            metrics (Iterable[MetricObjectWrapper]): Iterable objects with wrapped torchmetrics.Metric objects in MetricObjectWrapper objects.
        """
        metrics_dictionary = {metric.name: metric.metric_object for metric in metrics}

        super().__init__(metrics=metrics_dictionary)

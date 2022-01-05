from typing import Any, Callable, Optional

from torch import Tensor

from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.functional import auc, precision_recall_curve


class AUPR(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Computes AUPR based on inputs passed in to ``update`` previously."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return auc(*precision_recall_curve(preds, target)[1::-1])

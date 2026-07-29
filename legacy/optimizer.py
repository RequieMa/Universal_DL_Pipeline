from typing import Annotated
from msgspec import Struct, Meta

class AdamW(
    Struct, tag_field="name",
    tag="adamw",
    forbid_unknown_fields=True
):
    lr: Annotated[float, Meta(gt=0)]
    weight_decay: Annotated[float, Meta(ge=0)] = 0.0
    beta1: Annotated[float, Meta(ge=0)] = 0.0
    beta2: Annotated[float, Meta(ge=0)] = 0.0

class SGD(
    Struct, tag_field="name",
    tag="sgd",
    forbid_unknown_fields=True
):
    lr: Annotated[float, Meta(gt=0)]
    momentum: Annotated[float, Meta(ge=0, le=1)] = 0.0
    nesterov: bool = False

Optimizer = AdamW | SGD  # <- single alias
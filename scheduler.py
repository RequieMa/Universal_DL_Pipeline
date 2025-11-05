from typing import Annotated
from msgspec import Struct, Meta

class OneCycleLR(
    Struct, tag_field="name",
    tag="oneCycleLR",
    forbid_unknown_fields=True
):
    lr: Annotated[float, Meta(gt=0)]
    weight_decay: Annotated[float, Meta(ge=0)] = 0.0
    beta1: Annotated[float, Meta(ge=0)] = 0.0
    beta2: Annotated[float, Meta(ge=0)] = 0.0

class StepLR(
    Struct, tag_field="name",
    tag="stepLR",
    forbid_unknown_fields=True
):
    step_size: Annotated[int, Meta(gt=0)]
    gamma: Annotated[float, Meta(ge=0, le=1)] = 0.0

Scheduler = OneCycleLR | SGD  # <- single alias
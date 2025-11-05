from typing import Annotated
from msgspec import Struct, Meta


class AdamW(
    msgspec.Struct,
    tag="adamw",
    tag_field="name",
    forbid_unknown_fields=True
):
    lr: Annotated[float, Meta(gt=0)]
    weight_decay: Annotated[float, Meta(ge=0)] = 0.0

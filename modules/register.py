"""Register custom modules into the ultralytics namespace (no source patching).

``ultralytics.nn.tasks.parse_model`` resolves layer names via ``globals()[m]``
inside ``ultralytics/nn/tasks.py``; unknown layers fall through to the generic
branch (``c2 = ch[f]``, args passed through as written in the yaml). So all we
need for ``DCTAttention`` / ``SEAttention`` to be usable in a model yaml is to
inject the names into that module's globals (and into ``ultralytics.nn.modules``
for good measure). Call :func:`register_yolo_modules` before building a model.
"""

from __future__ import annotations

import ultralytics.nn.modules as unn_modules
import ultralytics.nn.tasks as unn_tasks

from .dct_attention import DCTAttention, SEAttention

__all__ = ["DCTAttention", "SEAttention", "register_yolo_modules"]


def register_yolo_modules() -> None:
    for ns in (unn_modules, unn_tasks):
        ns.DCTAttention = DCTAttention
        ns.SEAttention = SEAttention

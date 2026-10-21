"""Optional external VIS scorer injection for unmodified lmms-eval/GAZE code.

Python imports ``sitecustomize`` during startup. When EXTERNAL_VIS_SCORER is
set, this module wraps the selector model's ``from_pretrained`` and swaps only
``visual.importance_scorer`` after the original checkpoint has loaded. This
keeps the external GAZE and lmms-eval repositories read-only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


scorer_path = os.environ.get("EXTERNAL_VIS_SCORER", "").strip()
if scorer_path:
    evaluation_root = Path("/home/ruimeng/gaze/qwen-evaluation1")
    sys.path.insert(0, str(evaluation_root))
    import torch
    from safetensors.torch import load_file
    from token_compression.selector_model import Qwen2_5_VLForConditionalGeneration_Selector

    original_from_pretrained = Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained

    @classmethod
    def from_pretrained_with_external_scorer(cls, *args, **kwargs):
        model = original_from_pretrained(*args, **kwargs)
        target = model.visual.importance_scorer
        parameter = next(target.parameters())
        state = load_file(scorer_path, device="cpu")
        target.load_state_dict(
            {key: value.to(device=parameter.device, dtype=parameter.dtype) for key, value in state.items()},
            strict=True,
        )
        print(f"[external VIS scorer injected] {Path(scorer_path).resolve()}", flush=True)
        return model

    Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained = from_pretrained_with_external_scorer

# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Apply CFG prompt variants at the same point in the state transform pipeline."""


def transform_with_guidance(
    sample, transforms, tokenizer, positive_prompt, negative_prompt
):
    """Keep normalized, unpadded Pi05 state identical across all prompt variants."""
    guidance = {}
    for transform in transforms:
        if transform is tokenizer:
            for name, prompt in [
                ("positive", positive_prompt),
                ("negative", negative_prompt),
            ]:
                tokens = tokenizer({**sample, "prompt": prompt})
                guidance[f"tokenized_{name}_guidance_prompt"] = tokens[
                    "tokenized_prompt"
                ]
                guidance[f"tokenized_{name}_guidance_prompt_mask"] = tokens[
                    "tokenized_prompt_mask"
                ]
        sample = transform(sample)
    if not guidance:
        raise ValueError("Guidance tokenizer is missing from input transforms")
    return {**sample, **guidance}

#!/usr/bin/env python3
"""Generate prompts from GSM8K dataset and save to prompts.txt"""

from datasets import load_dataset

print("Generating prompts from GSM8K dataset...")
ds = load_dataset("openai/gsm8k", "main", split="test")
prompts = [ex["question"] for ex in ds.select(range(25))]

with open("prompts.txt", "w") as f:
    f.write("\n\n---\n\n".join(prompts))

print(f"✓ Generated prompts.txt with {len(prompts)} prompts")

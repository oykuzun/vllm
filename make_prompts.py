from datasets import load_dataset

print("Generating prompts from GSM8K dataset...")
ds = load_dataset("openai/gsm8k", "main", split="test")
prompts = [ex["question"] for ex in ds.select(range(25))] # it is mentioned to only use the first 25 prompts in the problem statement

with open("prompts.txt", "w") as f:
    f.write("\n\n---\n\n".join(prompts))
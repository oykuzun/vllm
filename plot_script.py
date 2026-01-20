#although we dont have moe_routes.jsonl, we can write a parser to generate a histogram based on how we have setup the json in the logger.

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import math

def calculate_metrics(counts, num_experts):
    # Total number of expert selections (tokens * top_k)
    total_selections = sum(counts.values())
    
    # p_i: probability of each expert being selected
    # Ensure we include experts that were NEVER selected (p_i = 0)
    probabilities = [counts.get(i, 0) / total_selections for i in range(num_experts)]
    
    # 1. Entropy: H = -sum(p_i * log2(p_i))
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    # 2. Coefficient of Variation (CV): std / mean
    expert_counts = [counts.get(i, 0) for i in range(num_experts)]
    mean_count = np.mean(expert_counts)
    std_count = np.std(expert_counts)
    cv = std_count / mean_count if mean_count > 0 else 0
    
    # 3. Utilization (%)
    utilization = (sum(1 for c in expert_counts if c > 0) / num_experts) * 100
    
    return entropy, cv, utilization

def generate_plot(log_path):
    expert_ids = []
    num_experts = 0
    model_id = "Unknown"

    with open(log_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data["type"] == "meta":
                model_id = data["model_id"]
                # Qwen1.5-MoE-A2.7B has 64 experts total (usually)
                # We can infer or hardcode based on the model_id
                num_experts = 64 
            elif data["type"] == "route":
                expert_ids.extend(data["topk_ids"])

    counts = Counter(expert_ids)
    entropy, cv, util = calculate_metrics(counts, num_experts)

    # Plotting
    plt.figure(figsize=(12, 6))
    x = list(range(num_experts))
    y = [counts.get(i, 0) for i in x]
    
    plt.bar(x, y, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title(f"Expert Usage Histogram: {model_id}\n(Entropy: {entropy:.2f}, CV: {cv:.2f}, Util: {util:.1f}%)")
    plt.xlabel("Expert ID")
    plt.ylabel("Selection Count")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.savefig("expert_hist.png")
    print(f"Plot saved to expert_hist.png")
    print(f"Metrics -> Entropy: {entropy:.4f}, CV: {cv:.4f}, Utilization: {util:.2f}%")

def get_normalized_distribution(expert_ids, num_experts, top_k):
    total_tokens = len(expert_ids) / top_k
    total_slots = len(expert_ids)
    
    # Calculate counts
    counts = Counter(expert_ids)
    
    # Normalize: Each value represents the % of total selections
    # e.g., 0.05 means this expert was chosen 5% of the time
    normalized = {i: counts.get(i, 0) / total_slots for i in range(num_experts)}
    
    # Get Top-3 for the README
    top_3 = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return normalized, top_3

# Example output for README:
if __name__ == "__main__":
    generate_plot("moe_routes.jsonl")
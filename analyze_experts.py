#!/usr/bin/env python3
"""
Analyze moe_routes.jsonl to find the top experts by usage frequency.
"""

import json
from collections import Counter, defaultdict

def analyze_experts(jsonl_file="moe_routes.jsonl"):
    """
    Analyze expert usage from moe_routes.jsonl file.
    
    Returns:
        - Top experts by count (how many times each expert was selected)
        - Top experts by total weight (sum of routing weights)
        - Expert utilization statistics
    """
    expert_counts = Counter()  # Count how many times each expert is selected
    expert_weights = defaultdict(float)  # Sum of weights for each expert
    total_selections = 0
    
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                # Skip metadata lines
                if record.get("type") != "route":
                    continue
                
                # Extract expert IDs and weights
                topk_ids = record.get("topk_ids", [])
                topk_weights = record.get("topk_weights", [])
                
                if not topk_ids or not topk_weights:
                    continue
                
                # Count each expert selection and accumulate weights
                for expert_id, weight in zip(topk_ids, topk_weights):
                    expert_counts[expert_id] += 1
                    expert_weights[expert_id] += weight
                    total_selections += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    # Get top experts by count
    top_by_count = expert_counts.most_common()
    
    # Get top experts by total weight
    top_by_weight = sorted(expert_weights.items(), key=lambda x: x[1], reverse=True)
    
    print("=" * 60)
    print("EXPERT USAGE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal expert selections: {total_selections}")
    print(f"Unique experts used: {len(expert_counts)}")
    print(f"Total experts available: 64 (assuming Qwen MoE model)")
    
    print("\n" + "-" * 60)
    print("TOP 10 EXPERTS BY SELECTION COUNT:")
    print("-" * 60)
    for rank, (expert_id, count) in enumerate(top_by_count[:10], 1):
        percentage = (count / total_selections) * 100 if total_selections > 0 else 0
        print(f"  {rank:2d}. Expert {expert_id:2d}: {count:6d} selections ({percentage:5.2f}%)")
    
    print("\n" + "-" * 60)
    print("TOP 10 EXPERTS BY TOTAL ROUTING WEIGHT:")
    print("-" * 60)
    for rank, (expert_id, total_weight) in enumerate(top_by_weight[:10], 1):
        percentage = (total_weight / sum(expert_weights.values())) * 100 if expert_weights else 0
        print(f"  {rank:2d}. Expert {expert_id:2d}: {total_weight:10.4f} total weight ({percentage:5.2f}%)")
    
    print("\n" + "-" * 60)
    print("TOP 3 EXPERTS (by selection count):")
    print("-" * 60)
    for rank, (expert_id, count) in enumerate(top_by_count[:3], 1):
        percentage = (count / total_selections) * 100 if total_selections > 0 else 0
        avg_weight = expert_weights[expert_id] / count if count > 0 else 0
        print(f"  {rank}. Expert {expert_id}: {count} selections ({percentage:.2f}%), avg weight: {avg_weight:.4f}")
    
    # Calculate utilization
    utilization = (len(expert_counts) / 64) * 100 if len(expert_counts) <= 64 else 100
    print(f"\nExpert Utilization: {len(expert_counts)}/64 = {utilization:.2f}%")
    
    # Calculate entropy
    import math
    probabilities = [count / total_selections for count in expert_counts.values() if total_selections > 0]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    print(f"Entropy: {entropy:.4f} (max for 64 experts: {math.log2(64):.4f})")
    
    # Calculate normalized distribution
    print("\n" + "-" * 60)
    print("NORMALIZED DISTRIBUTION (Top 20 Experts):")
    print("-" * 60)
    print("Expert ID | Count    | Percentage | Cumulative %")
    print("-" * 60)
    cumulative = 0.0
    for rank, (expert_id, count) in enumerate(top_by_count[:20], 1):
        percentage = (count / total_selections) * 100 if total_selections > 0 else 0
        cumulative += percentage
        print(f"  {expert_id:3d}    | {count:6d}  | {percentage:6.2f}%   | {cumulative:6.2f}%")
    
    # Calculate distribution statistics
    all_percentages = [(count / total_selections) * 100 for count in expert_counts.values() if total_selections > 0]
    top_4_percentage = sum([(count / total_selections) * 100 for _, count in top_by_count[:4]]) if total_selections > 0 else 0
    top_10_percentage = sum([(count / total_selections) * 100 for _, count in top_by_count[:10]]) if total_selections > 0 else 0
    remaining_percentage = 100 - top_10_percentage
    
    print("\n" + "-" * 60)
    print("DISTRIBUTION SUMMARY:")
    print("-" * 60)
    print(f"Top 4 experts:  {top_4_percentage:.2f}% of all selections")
    print(f"Top 10 experts: {top_10_percentage:.2f}% of all selections")
    print(f"Remaining {len(expert_counts) - 10} experts: {remaining_percentage:.2f}% of all selections")
    print(f"Average per expert: {(100.0 / len(expert_counts)):.2f}% (if perfectly uniform)")
    print(f"Std deviation: {math.sqrt(sum((p - (100.0/len(expert_counts)))**2 for p in all_percentages) / len(all_percentages)):.2f}%")
    
    return {
        "top_by_count": top_by_count[:3],
        "top_by_weight": top_by_weight[:3],
        "expert_counts": dict(expert_counts),
        "expert_weights": dict(expert_weights),
        "total_selections": total_selections,
        "utilization": utilization,
        "entropy": entropy,
        "normalized_distribution": {expert_id: (count / total_selections) * 100 for expert_id, count in expert_counts.items()},
        "top_4_percentage": top_4_percentage,
        "top_10_percentage": top_10_percentage
    }

if __name__ == "__main__":
    results = analyze_experts()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"Top 3 experts by count: {[f'Expert {eid}' for eid, _ in results['top_by_count']]}")
    print(f"Top 3 experts by weight: {[f'Expert {eid}' for eid, _ in results['top_by_weight']]}")

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

import os, json, time, random, torch

# Fixed generation parameters given in the problem statement
SEED = 1234
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0

random.seed(SEED) # random seed for reproducibility, and fairness for the comparison between logged and non-logged version

def run_without_logging():
    
    prompts = open("prompts.txt").read().split("\n\n---\n\n")

    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)
    llm = LLM(
        model="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        quantization="gptq", #quantize to fit into memory
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.6 #from 0.8
    )
    
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    t1 = time.time()
    
    timing_data = {
        "no_log": {
            "time_taken": t1 - t0,
            "tokens_generated": sum(len(o.outputs[0].token_ids) for o in outs)
        }
    }
    
    # Save initial timing data
    with open("timing.json", "w") as f:
        json.dump(timing_data, f, indent=2)
    
    return timing_data


def run_with_logging():

    # set variables for logging
    os.environ["VLLM_LOG_MOE"] = "moe_routes.jsonl" #file name to save MoE logs
    os.environ["VLLM_LOG_MOE_LAYER"] = "0" #Log only one MoE layer(configurable)
    
    prompts = open("prompts.txt").read().split("\n\n---\n\n")

    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)
    llm = LLM(
        model="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        quantization="gptq", #quantize to fit into memory - TODO: not working!!
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.6 #from 0.8
    )
    
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    t1 = time.time()
    
    timing_data_with_log = {
        "log": { 
            "time_taken": t1 - t0,
            "tokens_generated": sum(len(o.outputs[0].token_ids) for o in outs)
        }
    }
    
    try:
        with open("timing.json", "r") as f:
            timing_data = json.load(f) #read the old timing data with logging disabled
    except FileNotFoundError:
        timing_data = {}
    
    # Merge the log timing data with the old timing data
    timing_data.update(timing_data_with_log)
    
    # Save updated timing data
    with open("timing.json", "w") as f:
        json.dump(timing_data, f, indent=2)
    
    # observe the diff - should be the same
    if "no_log" in timing_data and "log" in timing_data:
        no_log_time = timing_data["no_log"]["time_taken"]
        log_time = timing_data["log"]["time_taken"]
        print(f"   No logging: {no_log_time:.2f}s")
        print(f"   With logging: {log_time:.2f}s")
    
    return timing_data


def main():
    # Run without logging first
    timing_data = run_without_logging()
    
    # Run with logging (appends to timing.json)
    timing_data = run_with_logging()



if __name__ == "__main__":
    main()
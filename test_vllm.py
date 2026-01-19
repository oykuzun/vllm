from vllm import LLM, SamplingParams #import to run with vLLM offline python api

# Use a tiny model for the first test to save time/memory
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Initialize the engine
# We use max_model_len=512 to keep memory usage low
llm = LLM(model=model_name, max_model_len=512, gpu_memory_utilization=0.6, enforce_eager=True)

# Set sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

# Generate text
prompts = ["The capital of Canada is", "The University of Waterloo is famous for"]
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}\n")
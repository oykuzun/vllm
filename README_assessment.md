**README (≤1 page)**: where you hooked, commands to run, and a 5–8 line note with (a) top-3 experts, (b) normalized distribution, (c) one metric (e.g., entropy) + a one-sentence interpretation.

-> Where the Logger is Hooked:

The MoE routing logger is implemented in `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`.
As the problem statment also suggests, we need it right after router’s topk_ids/topk_weights are computed.
The log_moe() function is called during the forward pass of all MoE layers in both quantized and unquantized data paths. 
It captures per-token routing decisions (expert IDs and weights) for the specified layer.
One alternative considered for where to add the logger was in router/fused_topk_router.py, directly in select_experts()
    -> this would have required passing layer_id through router interface, adding complexity

-> Commands:
generate prompts: python3 make_prompts.py
run with no logging: python3 run_generate.py
run with logging: export VLLM_LOG_MOE="moe_routes.jsonl"
                  export VLLM_LOG_MOE_LAYER="0"
                  python3 run_generate.py


-> Results Analysis - tried running various other smaller MoE models along with Qwen/Qwen1.5-MoE-A2.7B-Chat to get some sort of an output but was unable to run due to memory constraints...
                    - I tried running on a NVIDIA GeForce RTX 2080 Ti GPU through UWaterloo servers as well as the Intel(R) Xeon(R) Silver 4114 CPU
                      but when loading weights of the model, the program kept throwing: RuntimeError: Engine core initialization failed.
                      I tried different quantization methods, running on Google Colab, on my local machine, setting 
(a) Top-3 Experts: n/a 
    - Typically, in MoE models, a few experts become "specialists" for common tokens, we would have been able to detect these from the histogram

(b) Normalized Distribution: n/a

(c) Metrics: 
    - Entropy: Entropy measures the uncertainty/diversity in expert selection. Higher entropy = more diverse routing (experts used more evenly), lower entropy = more concentrated routing (few experts dominate - expert collapse). 
        Formula: H = -sum(p_i * log_2(p_i)) where p_i is the probability of expert i being selected.
    - Coefficient of variation (std/mean): can be used to measure load balancing across various experts - 0: perfect load balancing, high CV : biased router
    - Load Balancing Loss = alpha * N * sum(f_i * p_i) - loss used during training to encourage uniform routing: can be used to reason further, how well the training loss held up
    - Expert utilization: The percentage of total available experts that were called at least once during the 25-prompt GSM8K run. If utilization is low, model looks like it is wasting parameters


->AI usage log: tools used and how you verified output.
- Gemini was used for prompt generation logic/loading from the data set - verified by observing the output file
- Gemini was used to explore the large repository and to get recommendations on where to hook the requested logic. Verified by looking at the options given and observing the context around them personally.
- ChatGPT was used to come up with the sample script (test_vllm.py) to check environment setup
- Gemini was used to troubleshoot runtime errors when I was trying to troubleshoot the vLLM build/model loading tweaks to optimize for the platform being used: The AI helped identify that device="cpu" is deprecated in the newest EngineArgs and suggested environment variables (VLLM_USE_V1=0) to bypass the broken experimental core
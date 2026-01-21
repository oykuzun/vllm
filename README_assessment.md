-> Where the Logger is Hooked:

The MoE routing logger is implemented in `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`.
As the problem statment also suggests, we need it right after router’s topk_ids/topk_weights are computed.
The log_moe() function is called during the forward pass of all MoE layers in both quantized and unquantized data paths. 
It captures per-token routing decisions (expert IDs and weights) for the specified layer.
One alternative considered for where to add the logger was in router/fused_topk_router.py, directly in select_experts()
    -> this would have required passing layer_id through router interface, adding complexity

-> Commands:
generate prompts:               python3 make_prompts.py
run with no logging:            python3 run_generate.py no_log
         yes logging:           export VLLM_LOG_MOE_LAYER="0" #configure, default = 0
                                python3 run_generate.py log
histogram generation:           python3 plot_script.py
analyze results:                python3 analyze_experts.py # need moe_routes.jsonl in the path - run $python3 run_generate.py log first


-> Results Analysis - running the modified vLLM on two NVIDIA RTX A4500 GPUs (20 GB VRAM each) gave these results:
(a) Top-3 Experts: When we set enforce_eager = True, we see top-3 to be experts 43, 7, and 58 (similar to when enforce_eager = False, but now the distribution is way more balanced)
    - Typically, in MoE models, a few experts become "specialists" for common tokens, and these dominating experts show a consistent routing pattern

(b) Normalized Distribution: top 4 experts account for ~9% of the selection. mean per expert in uniform dist: 100/60 = 1.67% (vs. 3.17% for top expert, indicates good balancing), std dev: 0.44% (low, good balance), entropy: 5.86 (max = 6, good)

(c) Metrics (on real data) -> Entropy: 5.8600, CV: 0.3765, Utilization: 93.75%
    - Entropy: Entropy measures the uncertainty/diversity in expert selection. Higher entropy = more diverse routing (experts used more evenly), lower entropy = more concentrated routing (few experts dominate - expert collapse). 
        Formula: H = -sum(p_i * log_2(p_i)) where p_i is the probability of expert i being selected.
        *perfect uniform dist is log2(64) = 6, we have 5.86 which indicates decent routing
    - Coefficient of variation (std/mean): can be used to measure load balancing across various experts - 0: perfect load balancing, high CV -> biased router
        *perfectly balanced MoE has CV = 0, we have 0.3765 - balanced load distribution
    - Expert utilization: The percentage of total available experts that were called at least once during the 25-prompt GSM8K run. If utilization is low, model looks like it is wasting parameters
        *93.75% util means model is using most of its capacity rather than relying on a few experts (60 out of 64 experts were used at least once during inference)

**CRITICAL FINDING** - enforce_eager Impact on Expert Distribution:
- When *enforce_eager=False* (optimized CUDA kernels): Top 4 experts dominate, showing expert collapse
  - Metrics -> Entropy: 2.0791, CV: 3.8410, Utilization: 28.12% (poor load balancing)
  - This suggests a potential numerical precision issue in the optimized CUDA kernels
- When *enforce_eager=True* (eager PyTorch mode): Experts are well balanced across all 64 experts:)
  - Metrics -> Entropy: 5.8600, CV: 0.3765, Utilization: 93.75%
- Both configurations now use enforce_eager=True for fair comparison and consistent behavior - this fixed the difference in timing issue I was having initially!!
- This finding suggests the optimized kernels may have routing precision issues that cause expert collapse

--->  AI usage log: tools used and how you verified output.
- Gemini was used for prompt generation logic/loading from the data set - verified by observing the output file
- Gemini was used to explore the large repository and to get recommendations on where to hook the requested logic. Verified by looking at the options given and observing the context around them personally.
- ChatGPT was used to come up with the sample script (test_vllm.py) to check environment setup
- Gemini was used to troubleshoot runtime errors when I was trying to troubleshoot the vLLM build/model loading tweaks to optimize for the platform being used
- ChatGPT and Gemini were used to assist with reasoning about vLLM configuration parameters and GPU memory trade-offs. Suggested parameters were manually evaluated and refined through empirical testing. Output correctness was verified by comparing token counts and generated outputs across runs, and performance was validated via timed measurements with fixed seeds and generation settings.
- Gemini was used to come up with analyze_experts.py script to parse the moe_routes.jsonl file and draw conclusions on top experts and get the distribution across them, verified by observing the output & cross checking with the histogram

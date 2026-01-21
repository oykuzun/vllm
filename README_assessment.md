**README (≤1 page)**: where you hooked, commands to run, and a 5–8 line note with (a) top-3 experts, (b) normalized distribution, (c) one metric (e.g., entropy) + a one-sentence interpretation.

-> Where the Logger is Hooked:

The MoE routing logger is implemented in `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`.
As the problem statment also suggests, we need it right after router’s topk_ids/topk_weights are computed.
The log_moe() function is called during the forward pass of all MoE layers in both quantized and unquantized data paths. 
It captures per-token routing decisions (expert IDs and weights) for the specified layer.
One alternative considered for where to add the logger was in router/fused_topk_router.py, directly in select_experts()
    -> this would have required passing layer_id through router interface, adding complexity

-> Commands:
generate prompts:               python3 make_prompts.py
run with no logging & logging: 
                                export VLLM_LOG_MOE_LAYER="0" #run this only if you want to configure the layer, default is layer 0 
                                python3 run_generate.py

histogram generation:           python plot_script.py


-> Results Analysis - running the modified vLLM on two NVIDIA RTX A4500 GPUs (20 GB VRAM each) gave these results:
(a) Top-3 Experts: there is a tie between Expert 43, 5, 7, 58 for the first 25 primpts of the GSM8K dataset.
    - Typically, in MoE models, a few experts become "specialists" for common tokens, and these dominating experts show a consistent routing pattern

(b) Normalized Distribution: top 4 experts account for ~25% of all expert selections, while the rest of 60 share only ~0.8%

(c) Metrics (on real data) -> Entropy: 2.0791, CV: 3.8410, Expert Utilization: 28.12%
    - Entropy: Entropy measures the uncertainty/diversity in expert selection. Higher entropy = more diverse routing (experts used more evenly), lower entropy = more concentrated routing (few experts dominate - expert collapse). 
        Formula: H = -sum(p_i * log_2(p_i)) where p_i is the probability of expert i being selected.
        *perfect uniform dist is log2(64) = 6, we have 2.08 -> biased router! It is acting like it has 2^2.08 !~ 4.22 active experts only!
    - Coefficient of variation (std/mean): can be used to measure load balancing across various experts - 0: perfect load balancing, high CV -> biased router
        *perfectly balanced MoE has CV = 0, we have 3.84 - imbalanced load distribution - can cause computatinal bottlenecks, hand in hand with entropy
    - Expert utilization: The percentage of total available experts that were called at least once during the 25-prompt GSM8K run. If utilization is low, model looks like it is wasting parameters
        *28.12% util : almost 70% of the model's knowledge has been consulted - could be because prompts have similar structure.
    - Load Balancing Loss = alpha * N * sum(f_i * p_i) - loss used during training to encourage uniform routing: can be used to reason further, how well the training loss held up

TODO:
- timing for logging and no logging are not the same, debug further.
-printing the metadata in moe_routed.jsonl more than once, to be fixed.

--->  AI usage log: tools used and how you verified output.
- Gemini was used for prompt generation logic/loading from the data set - verified by observing the output file
- Gemini was used to explore the large repository and to get recommendations on where to hook the requested logic. Verified by looking at the options given and observing the context around them personally.
- ChatGPT was used to come up with the sample script (test_vllm.py) to check environment setup
- Gemini was used to troubleshoot runtime errors when I was trying to troubleshoot the vLLM build/model loading tweaks to optimize for the platform being used
- ChatGPT and Gemini were used to assist with reasoning about vLLM configuration parameters and GPU memory trade-offs. Suggested parameters were manually evaluated and refined through empirical testing. Output correctness was verified by comparing token counts and generated outputs across runs, and performance was validated via timed measurements with fixed seeds and generation settings.

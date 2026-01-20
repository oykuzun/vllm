# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from abc import abstractmethod

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

logger = init_logger(__name__)


# Initialize MoE logging from environment variables
# VLLM_LOG_MOE: path to log file - moe_routes.jsonl
# VLLM_LOG_MOE_LAYER: layer ID to log (default is 0)
log_file = None
if os.getenv("VLLM_LOG_MOE"):
    try:
        log_file = open(os.getenv("VLLM_LOG_MOE"), "a")
    except Exception:
        pass

try:
    target_layer = int(os.getenv("VLLM_LOG_MOE_LAYER", "0"))
except ValueError:
    target_layer = 0

def log_moe(
    layer_id: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_tokens: int,
) -> None:
    """
    Minimal flag-gated logger for MoE routing.
    """
    global log_file, target_layer
    
    # Early return if logging disabled or wrong layer
    # for now, we call the logger for every layer but it only logs for the target layer
    # each transformer layer with a MoE block calls log_moe() during forward pass
    if log_file is None or layer_id != target_layer:
        return
    
    # top_k: the number of experts selected per token, we can get this from either weights or ids
    top_k = topk_ids.shape[1]
    
    try:
        # get meta data mentioned in the email
        #TODO: think of what other metadata might be relevant here
        from vllm import __version__ as vllm_version
        model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat" # model given in the problem statement
        torch_version = torch.__version__
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        header = {
            "type": "meta",
            "model_id": model_id,
            "vllm_version": vllm_version,
            "torch_version": torch_version,
            "device": device,
            "seed": 1024, #seed value for metadata - given in the problem statement
            "layers_logged": [layer_id],
            "top_k": top_k,
        }
        log_file.write(json.dumps(header) + "\n")
    except Exception:
        pass
    
    # Convert to CPU and log per token & to iterate over them for JSON logging
    topk_weights_cpu = topk_weights.cpu().tolist()
    topk_ids_cpu = topk_ids.cpu().tolist()
    
    for token_idx in range(num_tokens): #Per-token record (one line per token per logged layer)
        record = {
            "type": "route",
            "req_id": "r1", #hard coded for now -> TODO: check this? i dont think we have request id at this level?
            "token_idx": token_idx, #represents position within the current batch of tokens
            "layer": layer_id,
            "topk_ids": topk_ids_cpu[token_idx],
            "topk_weights": topk_weights_cpu[token_idx],
        }
        log_file.write(json.dumps(record) + "\n")
    
    log_file.flush()

class FusedMoEMethodBase(QuantizeMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe: FusedMoEConfig = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def uses_weight_scale_2_pattern(self) -> bool:
        """
        Returns True if this quantization method uses 'weight_scale_2' pattern
        for per-tensor weight scales (e.g., FP4 variants), False otherwise.

        This method should be overridden by subclasses that use the
        'weight_scale_2' pattern instead of the standard 'weight_scale' pattern.
        """
        return False

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalize | None:
        from .all2all_utils import maybe_make_prepare_finalize

        return maybe_make_prepare_finalize(
            self.moe, self.moe_quant_config, routing_tables
        )

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        # based on the all2all implementation, select the appropriate
        # gemm implementation
        raise NotImplementedError(
            f"{self.__class__.__name__} must select appropriate gemm "
            "implementation based on the prepare_finalize"
        )

    def prepare_dp_allgather_tensor(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Hook to prepare tensors and extra tensors for DP allgather + EP dispatch."""
        raise NotImplementedError(
            "Method 'prepare_dp_allgather_tensor' is not implemented in "
            f"{self.__class__.__name__}."
        )

    @abstractmethod
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        raise NotImplementedError

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    @property
    def supports_eplb(self) -> bool:
        return False

    @property
    def allow_inplace(self) -> bool:
        return False

    @property
    def method_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def apply(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

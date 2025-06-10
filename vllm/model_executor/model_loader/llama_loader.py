# SPDX-License-Identifier: Apache-2.0
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from vllm.config import LoadConfig, ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)


def load_chkpt_params(
    ckpt_dir: Optional[Union[str, Path]] = None,
    params_path: Optional[Union[str, Path]] = None,
    overwrite_model_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if params_path is None:
        assert ckpt_dir is not None
        logger.info(f"Loading from unified ckpt in {ckpt_dir}")
        params_path = Path(ckpt_dir) / "sharded_params.json"
    else:
        params_path = Path(
            os.path.dirname(params_path)) / "sharded_params.json"

    with open(params_path) as f:
        params = json.loads(f.read())

    # Use MP size and PP size from consolidate_params.json if it exists
    consolidate_params_path = Path(params_path) / "consolidate_params.json"
    if consolidate_params_path.exists():
        consolidate_params = json.load(open(consolidate_params_path))
        old_mp_size = params.get("model_parallel_size", 1)
        old_pp_size = params.get("pipeline_parallel_size", 1)
        new_mp_size = consolidate_params.get("model_parallel_size", 1)
        new_pp_size = consolidate_params.get("pipeline_parallel_size", 1)

        if new_mp_size != old_mp_size:
            params["model_parallel_size"] = new_mp_size
        if new_pp_size != old_pp_size:
            params["pipeline_parallel_size"] = new_pp_size

    # Add only necessary metap params
    metap_params = params["model"].get("metap", {})
    use_metap = metap_params.get("use_metap")
    if use_metap:
        params["model"]["use_metap"] = use_metap
        params["model"]["base_width"] = metap_params.get("base_width")
        params["model"]["metap_mode"] = metap_params.get("metap_mode")
        params["model"]["m_emb"] = metap_params.get("m_emb")

    # Overwite model arguments if necessary
    if overwrite_model_args is not None:

        def deep_update(source, overrides):
            for key, value in overrides.items():
                if isinstance(value, dict) and value:
                    returned = deep_update(source.get(key, {}), value)
                    source[key] = returned
                else:
                    source[key] = overrides[key]
            return source

        if isinstance(overwrite_model_args, str):
            from ast import literal_eval

            overwrite_model_args = literal_eval(overwrite_model_args)

        deep_update(params["model"], overwrite_model_args)

    return params


def permute(name, w, n_heads, dim1=2048, dim2=2048):
    logger.info(
        f"\t\t\t permute {name=}, {w.shape=}, {n_heads=}, {dim1=}, {dim2=}")
    return (w.view(n_heads, dim1 // n_heads // 2, 2,
                   dim2).transpose(1, 2).reshape(dim1, dim2))


class LlamaUnifiedLoader(BaseModelLoader):
    """
    Unified checkpoint weights:
        layers.0.feed_forward.w1.weight
        layers.0.feed_forward.w3.weight
        layers.0.feed_forward.w2.weight
        layers.0.ffn_norm.weight
        layers.0.attention.wo.weight
        layers.0.attention.wq.weight
        layers.0.attention_norm.weight
        layers.0.attention.wk.weight
        layers.0.attention.wv.weight
        tok_embeddings.weight
        norm.weight
        output.weight

    Expected weights (original) <> checkpoint weights mapping:
        model.layers.0.input_layernorm.weight <> attention_norm
        model.layers.0.mlp.down_proj.weight <> w2
        model.layers.0.mlp.gate_up_proj.weight <> w1:gate_proj, w3:up_proj
        model.layers.0.post_attention_layernorm.weight <> ffn_norm
        model.layers.0.self_attn.o_proj.weight <> wo
        model.layers.0.self_attn.qkv_proj.weight <> wq, wk, wv
        model.embed_tokens.weight <> tok_embeddings
        model.norm.weight <> norm
        lm_head.weight <> output
    
    Expected weights (transformed) <> checkpoint weights mapping:
        layers.21.mlp.gate_proj.weight <> w1
        layers.21.mlp.up_proj.weight <> w3
        layers.21.mlp.down_proj.weight <> w2
        layers.21.post_attention_layernorm.weight <>ffn_norm
        layers.21.self_attn.o_proj.weight <> wo
        layers.21.self_attn.q_proj.weight <> wq
        layers.21.self_attn.k_proj.weight <> wk
        layers.21.self_attn.v_proj.weight <> wv
        layers.21.input_layernorm.weight <> attention_norm
        embed_tokens.weight
        norm.weight
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        logger.info("[qqzz] init LlamaUnifiedLoader")
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        ckpt_path = model_config.model

        # params = load_chkpt_params(params_path=ckpt_path)
        # ec_args = params["model"].get("experts_choice_moe", {})
        # is_moe = ec_args.get("is_enabled", False)

        patterns = ["/layer_*/*", "/embeddings/*", "/outputs/*"]
        weight_files = []
        for patten in patterns:
            weight_files.extend(glob.glob(ckpt_path + patten))
        checkpoint_weights = {}
        for weight_file in weight_files:
            checkpoint_weights.update(
                torch.load(weight_file, map_location="cpu"))

        weight_names_to_load = {name for name, _ in model.named_parameters()}
        logger.info(f"[qqzz] Need to load {weight_names_to_load=}")
        logger.info(f"[qqzz] Loading {checkpoint_weights.keys()=}")
        n_layers = 22
        n_heads = 16
        dim = 2048
        weights_to_load = {}
        for i in range(n_layers):
            weights_to_load[
                f"model.layers.{i}.self_attn.q_proj.weight"] = checkpoint_weights[
                    f"layers.{i}.attention.wq.weight"]
            # permute(
            #     name=f"layers.{i}.attention.wq.weight",
            #     checkpoint_weights[f"layers.{i}.attention.wq.weight"],
            #     n_heads=n_heads,
            # )
            weights_to_load[
                f"model.layers.{i}.self_attn.k_proj.weight"] = checkpoint_weights[
                    f"layers.{i}.attention.wk.weight"]
            # permute(
            #     name=f"layers.{i}.attention.wk.weight",
            #     checkpoint_weights[f"layers.{i}.attention.wk.weight"],
            #     n_heads=n_heads,
            #     dim1=dim // n_heads,
            # )
            weights_to_load[
                f"model.layers.{i}.self_attn.v_proj.weight"] = checkpoint_weights[
                    f"layers.{i}.attention.wv.weight"]
            weights_to_load[
                f"model.layers.{i}.self_attn.o_proj.weight"] = checkpoint_weights[
                    f"layers.{i}.attention.wo.weight"]
            weights_to_load[
                f"model.layers.{i}.mlp.gate_up_proj.weight"] = checkpoint_weights[
                    f"layers.{i}.feed_forward.w1.weight"]
            weights_to_load[
                f"model.layers.{i}.mlp.down_proj.weight"] = checkpoint_weights[
                    f"layers.{i}.feed_forward.w2.weight"]
            weights_to_load[
                f"model.layers.{i}.mlp.up_proj.weight"] = checkpoint_weights[
                    f"layers.{i}.feed_forward.w3.weight"]
            weights_to_load[
                f"model.layers.{i}.input_layernorm.weight"] = checkpoint_weights[
                    f"layers.{i}.attention_norm.weight"]
            weights_to_load[
                f"model.layers.{i}.post_attention_layernorm.weight"] = checkpoint_weights[
                    f"layers.{i}.ffn_norm.weight"]
        weights_to_load["model.embed_tokens.weight"] = checkpoint_weights[
            "tok_embeddings.weight"]
        weights_to_load["model.norm.weight"] = checkpoint_weights[
            "norm.weight"]
        weights_to_load["lm_head.weight"] = checkpoint_weights["output.weight"]
        missing_weights = weight_names_to_load - set(weights_to_load.keys())
        extra_weights = set(weights_to_load.keys()) - weight_names_to_load
        if missing_weights:
            logger.warning(f"[qqzz] Missing weights: {missing_weights}")
            logger.warning(f"[qqzz] Extra weights: {extra_weights}")
        print(f"[qqzz] {model=}")
        model.load_weights(checkpoint_weights.items())

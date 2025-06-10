# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

from transformers import PretrainedConfig

from vllm.logger import init_logger

logger = init_logger(__name__)


def file_to_dict(file_name: str, model: Union[str, Path]):
    file_path = os.path.join(model, file_name)
    with open(file_path) as file:
        return json.load(file)


def compute_intermediate_size(n,
                              ffn_dim_multiplier: int = 1,
                              multiple_of: int = 256):
    return multiple_of * (
        (int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) //
        multiple_of)


def load_llama_unified_config(model: Union[str, Path], revision: Optional[str],
                              **kwargs) -> PretrainedConfig:
    # This function loads a params.json config which
    # should be used when loading models in mistral format

    config_file_name = "sharded_params.json"

    config_dict = file_to_dict(config_file_name, model)
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"Failed to load unified '{config_file_name}' config for model "
            f"{model}. Please check if the model is a Llama-unified-format "
            f"model and if the config file exists.")

    config_mapping = {
        "dim": "hidden_size",
        "norm_eps": "rms_norm_eps",
        "n_kv_heads": "num_key_value_heads",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "eos_id": "eos_token_id",
        "attention_chunk_size": "batchify_local_attention_len",
        "attention_sliding_window_size": "local_attention_window_len",
        "rope_scaling_factor": "rope_scale_factor",
        "rope_high_freq_factor": "high_freq_factor",
        "nope_layer_interval": "every_n_layers_nope",
        "floor_scale": "attn_temperature_tuning_floor_scale",
        "attn_scale": "attn_temperature_tuning_q_scale_constant",
    }

    def recurse_elems(elem: Any):
        if isinstance(elem, dict):
            config_dict = {}
            for key, value in elem.items():
                key = config_mapping.get(key, key)
                config_dict[key] = recurse_elems(value)

            return config_dict
        else:
            return elem

    model_config_dict = config_dict["model"]
    model_config_dict["hidden_act"] = model_config_dict.get(
        "activation", "silu")
    model_config_dict["tie_word_embeddings"] = model_config_dict.get(
        "tie_embeddings", False)
    model_config_dict["max_position_embeddings"] = model_config_dict.get(
        "max_position_embeddings", 16384)

    if model_config_dict.get("quantization") is not None:
        quantization = model_config_dict.get("quantization", {})
        if quantization.get("qformat_weight") == "fp8_e4m3":
            # This maps to the FP8 static per-tensor quantization scheme
            quantization_config = {
                "quant_method": "fp8",
                "activation_scheme": "static"
            }
        elif quantization.get("quant_method") == "compressed-tensors":
            # Pass through the quantization config to compressed-tensors
            quantization_config = quantization
        else:
            raise ValueError(
                f"Found unknown quantization='{quantization}' in config")

        model_config_dict["quantization_config"] = quantization_config

    config_type: Literal["text",
                         "multimodal"] = "multimodal" if model_config_dict.get(
                             "modalities").get("use_image") else "text"
    model_config_dict["architectures"] = ["RawLlamaForCausalLM"]

    if config_type == "multimodal":
        raise NotImplementedError

    model_config_dict.update(kwargs)
    multiple_of = (model_config_dict["multiple_of"]
                   if "multiple_of" in model_config_dict else 256)
    ffn_dim_multiplier = (model_config_dict["ffn_dim_multiplier"]
                          if "ffn_dim_multiplier" in model_config_dict else 1)
    intermediate_size = compute_intermediate_size(model_config_dict["dim"],
                                                  ffn_dim_multiplier,
                                                  multiple_of)
    model_config_dict["intermediate_size"] = intermediate_size

    model_config_dict = recurse_elems(model_config_dict)
    if model_config_dict.get("num_key_value_heads") is None:
        model_config_dict["num_key_value_heads"] = model_config_dict[
            "num_attention_heads"]

    # transform to HF config format
    if config_type == "multimodal":
        model_config_dict["text_config"] = PretrainedConfig(
            **model_config_dict["text_config"])
        model_config_dict["vision_config"] = PretrainedConfig(
            **model_config_dict["vision_config"])

    return PretrainedConfig(**model_config_dict)

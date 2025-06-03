# SPDX-License-Identifier: Apache-2.0
import json

global_same_key_list = [
    "dim",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "vocab_size",
    "multiple_of",
    "ffn_dim_multiplier",
    "ffn_exp",
    "norm_eps",
    "attention_chunk_size",
    "rope_theta",
    "use_scaled_rope",
    "use_qk_norm",
]
global_key_mapping = {
    **{
        key: key
        for key in global_same_key_list
    },
    "attention_chunk_size": "batchify_local_attention_len",
    "attention_sliding_window_size": "local_attention_window_len",
    "rope_scaling_factor": "rope_scale_factor",
    "rope_high_freq_factor": "high_freq_factor",
    "nope_layer_interval": "every_n_layers_nope",
    "floor_scale": "attn_temperature_tuning_floor_scale",
    "attn_scale": "attn_temperature_tuning_q_scale_constant",
}
yoco_args_mapping = {
    "kv_layers": "kv_layers",
    "attention_window_schema": "global_attn_cfg",
}
moe_mapping = {
    "num_experts": "num_experts",
    "capacity_factor": "capacity_factor",
    "auto_scale_F": "auto_scale_F",
    "top_k": "top_k",
    "interleave_moe_layer_step": "interleave_moe_layer_step",
}
default_vision_config = {
    "image_size": {
        "height": 336,
        "width": 336
    },
    "patch_size": {
        "height": 14,
        "width": 14
    },
    "dim": 1408,
    "n_layers": 34,
    "n_heads": 16,
    "mlp_ratio": 4.0,
    "output_dim": 4096,
    "pixel_shuffle_ratio": 0.5,
}


def convert_attention_window_schema(global_attn_cfg: str, layer: int,
                                    window_size: int) -> str:
    if global_attn_cfg == "all":
        return str([0] * layer)
    global_attn_cfg_list = global_attn_cfg[1:-1].split(",")
    attention_window_schema_list = [
        int(w) // window_size for w in global_attn_cfg_list
    ]
    assert len(attention_window_schema_list) == layer
    attention_window_schema = ",".join(
        [str(w) for w in attention_window_schema_list])
    return f"[{attention_window_schema}]"


def convert_batchify_attention_schema(batchify_global_attn_cfg: str,
                                      layer: int,
                                      local_attention_len: int) -> str:
    batchify_global_attn_cfg_list = batchify_global_attn_cfg[1:-1].split(",")
    attention_window_schema_list = [
        int(w) // local_attention_len for w in batchify_global_attn_cfg_list
    ]
    assert len(attention_window_schema_list) == layer
    attention_window_schema = ",".join(
        [str(w) for w in attention_window_schema_list])
    return f"[{attention_window_schema}]"


def get_yoco_kv_layers(yoco_args_raw) -> list[int]:
    if "kv_layers" not in yoco_args_raw:
        return []
    try:
        yoco_kv_layer = [
            int(layer) for layer in yoco_args_raw["kv_layers"].split(",")
        ]
        return yoco_kv_layer
    except Exception as e:
        print(f"Error parsing kv_layers: {e}")
        return []


def process_json(in_path, out_path, vision=False):
    # Read the input JSON file
    with open(in_path) as f:
        raw_config = json.load(f)
        print(f"Loaded raw config from {in_path}")
    model_config = raw_config["model"]
    output_json = {}
    for key in global_key_mapping:
        if global_key_mapping[key] in model_config:
            print(f"Found key {key} in model config")
            output_json[key] = model_config[global_key_mapping[key]]
    if "yoco_args" in model_config:
        yoco_args_raw = model_config["yoco_args"]
        if ("batchify_global_attn_cfg" in model_config
                and "every_n_layers_nope" in model_config):
            # irope
            assert "attention_chunk_size" in output_json
            local_attention_len = output_json["attention_chunk_size"]
            attention_window_schema = convert_batchify_attention_schema(
                model_config["batchify_global_attn_cfg"],
                output_json["n_layers"],
                local_attention_len,
            )
        else:
            # sliding window
            assert "attention_sliding_window_size" in output_json
            assert "global_attn_cfg" in model_config
            window_size = output_json["attention_sliding_window_size"]
            attention_window_schema = convert_attention_window_schema(
                model_config["global_attn_cfg"], output_json["n_layers"],
                window_size)
        yoco_args = {
            "kv_layers": get_yoco_kv_layers(yoco_args_raw),
            "attention_window_schema": attention_window_schema,
        }
        print(f"Generated yoco args: {yoco_args}")
        output_json["yoco_args"] = yoco_args
    if vision:
        output_json["vision_args"] = default_vision_config
    if "modalities" in model_config and "image" in model_config[
            "modalities"] and model_config["modalities"].get("use_image"):
        image_config = model_config["modalities"]["image"]
        output_json["vision_args"]["image_size"]["height"] = image_config[
            "image_height"]
        output_json["vision_args"]["image_size"]["width"] = image_config[
            "image_width"]
        output_json["vision_args"]["patch_size"]["height"] = image_config[
            "patch_height"]
        output_json["vision_args"]["patch_size"]["width"] = image_config[
            "patch_width"]
        output_json["vision_args"]["pixel_shuffle_ratio"] = image_config[
            "ps_ratio"]
    if ("experts_choice_moe" in model_config
            and model_config["experts_choice_moe"]["is_enabled"]):
        moe_args_raw = model_config["experts_choice_moe"]
        moe_args = {}
        for key in moe_mapping:
            if moe_mapping[key] in moe_args_raw:
                moe_args[key] = moe_args_raw[moe_mapping[key]]
        output_json["moe_args"] = moe_args
    # Raw Llama model config
    output_json["model_type"] = "rawllama"
    with open(out_path, "w") as f:
        json.dump(output_json, f, indent=4)
    print(f"Processed config saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        "Converting raw params.json to processed open-source params.json")
    parser.add_argument("--raw-config",
                        help="Path to the raw params JSON file",
                        required=True)
    parser.add_argument("--out-config",
                        help="Path to the output params JSON file",
                        required=True)
    parser.add_argument("--vision",
                        action="store_true",
                        help="add default vision args")
    args = parser.parse_args()
    process_json(args.raw_config, args.out_config, args.vision)

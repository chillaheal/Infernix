"""Qwen-Image (diffusers pipeline) adapter: DiT + Qwen3-VL text tower to logical parameters.

Qwen-Image is not a causal LM. A pipeline root holds subfolders transformer/,
text_text/, vae/ (model_index.json _class_name QwenImage21Pipeline). This
adapter opens the transformer and text-tower roots directly and maps every
source tensor to a logical Parameter. It does NOT reuse the text build_model
(no tokens/frontend/RoPE path). VAE maps every source tensor as a direct BF16
parameter with its conv/norm rank preserved as-is from the source header.

FP8 contract: rank-two DiT/text linears quantize BF16->E4M3FN row-scaled on CPU
(fp8_row_maxabs forces host numpy); adaLN modulation, norms, QK-norm, timestep
and img/proj in/out stay direct BF16 (default cast_direct + A16Only).

The input names below (e.g. "dit/layers/0/norm1_output") are the logical
operand contract the C++ qwen_image Program and bind_dit/bind_text_encoder must
emit for each GEMM activation. They are chosen here as the architecture's
naming authority; the loader binds to these exact strings.
"""

from __future__ import annotations

from math import prod
from pathlib import Path

from .model import Model, Parameter
from .recipe import Recipe
from .sources.safetensors import SafetensorsSource, tensor_source

FP8 = "fp8_e4m3fn_row_bf16"
FP8_METHOD = "fp8_row_maxabs"


def is_qwen_image(base: SafetensorsSource) -> bool:
    """Detect a Qwen-Image pipeline root by its model_index.json."""
    root = base.path if base.path.is_dir() else base.path.parent
    index = root / "model_index.json"
    if not index.is_file():
        return False
    import json

    try:
        return json.loads(index.read_text()).get("_class_name") == "QwenImage21Pipeline"
    except (ValueError, OSError):
        return False


def _load_config(path: Path) -> dict:
    import json

    return json.loads(path.read_text())


def _add(model, name, store, source_name, shape, *, inputs=(), direct="bf16"):
    shape = tuple(shape)
    real = tuple(store.describe(source_name).shape)
    if real != shape:
        raise ValueError(f"{name}: declared {shape}, source {source_name} is {real}")
    source = tensor_source(store, source_name, shape)
    model.add(
        Parameter(
            name,
            shape,
            source,
            None,
            tuple(inputs),
            direct,
            residency=name.split("/", 1)[0],
        )
    )


def _build_dit(model, store, cfg):
    heads = cfg["num_attention_heads"]
    head_dim = cfg["attention_head_dim"]
    hidden = heads * head_dim
    intermediate = hidden * cfg["mlp_ratio"]
    image_dim = cfg["in_channels"] * cfg["patch_size"] * cfg["patch_size"]
    out_dim = cfg["out_channels"] * cfg["patch_size"] * cfg["patch_size"]

    _add(model, "dit/img_in", store, "img_in.weight", (hidden, image_dim),
         inputs=("dit/patch_input",))
    _add(model, "dit/proj_out", store, "proj_out.weight", (out_dim, hidden),
         inputs=("dit/final_hidden",))
    _add(model, "dit/modulation", store, "modulation.1.weight",
         (4 * hidden, hidden))  # global adaLN for all blocks; stays A16Only
    _add(model, "dit/norm_out", store, "norm_out.linear.weight", (hidden, hidden))
    _add(model, "dit/txt_in/in", store, "txt_in.in_layer.weight", (hidden, cfg["context_in_dim"]),
         inputs=("dit/context_input",))
    _add(model, "dit/txt_in/out", store, "txt_in.out_layer.weight", (hidden, hidden),
         inputs=("dit/context_activation",))
    _add(model, "dit/txt_in/norm", store, "txt_in.text_norm.weight", (cfg["context_in_dim"],))
    _add(model, "dit/time_embed/linear_1", store,
         "time_text_embed.timestep_embedder.linear_1.weight", (hidden, 256),
         inputs=("dit/timestep_sinu",))
    _add(model, "dit/time_embed/linear_2", store,
         "time_text_embed.timestep_embedder.linear_2.weight", (hidden, hidden),
         inputs=("dit/time_embed/activation",))

    for i in range(cfg["num_layers"]):
        p = f"dit/layers/{i}/"
        sp = f"transformer_blocks.{i}."
        mixer_in = p + "norm1_output"
        _add(model, p + "attention/query", store, sp + "attn.to_q.weight", (hidden, hidden),
             inputs=(mixer_in,))
        _add(model, p + "attention/key", store, sp + "attn.to_k.weight", (hidden, hidden),
             inputs=(mixer_in,))
        _add(model, p + "attention/value", store, sp + "attn.to_v.weight", (hidden, hidden),
             inputs=(mixer_in,))
        _add(model, p + "attention/query_norm", store, sp + "attn.norm_q.weight", (head_dim,))
        _add(model, p + "attention/key_norm", store, sp + "attn.norm_k.weight", (head_dim,))
        _add(model, p + "attention/output", store, sp + "attn.to_out.0.weight",
             (hidden, hidden), inputs=(p + "attention_output",))
        mlp_in = p + "norm2_output"
        _add(model, p + "mlp/gate", store, sp + "img_mlp.gate_layer.weight",
             (intermediate, hidden), inputs=(mlp_in,))
        _add(model, p + "mlp/up", store, sp + "img_mlp.proj.weight",
             (intermediate, hidden), inputs=(mlp_in,))
        _add(model, p + "mlp/down", store, sp + "img_mlp.out.weight",
             (hidden, intermediate), inputs=(p + "mlp_product",))


def _build_text_encoder(model, store, cfg):
    text = cfg["text_config"]
    hidden = text["hidden_size"]
    head_dim = text["head_dim"]
    q = text["num_attention_heads"] * head_dim
    k = text["num_key_value_heads"] * head_dim
    intermediate = text["intermediate_size"]

    _add(model, "text/embed_tokens", store, "model.language_model.embed_tokens.weight",
         (text["vocab_size"], hidden))
    _add(model, "text/final_norm", store, "model.language_model.norm.weight", (hidden,))
    for i in range(text["num_hidden_layers"]):
        p = f"text/layers/{i}/"
        sp = f"model.language_model.layers.{i}."
        mixer_in = p + "norm1_output"
        for role, field in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")):
            width = q if role == "query" else k
            _add(model, p + "attention/" + role, store, sp + "self_attn." + field + ".weight",
                 (width, hidden), inputs=(mixer_in,))
        for role, field in (("query_norm", "q_norm"), ("key_norm", "k_norm")):
            _add(model, p + "attention/" + role, store, sp + "self_attn." + field + ".weight",
                 (head_dim,))
        _add(model, p + "attention/output", store, sp + "self_attn.o_proj.weight",
             (hidden, q), inputs=(p + "attention_output",))
        mlp_in = p + "norm2_output"
        for role, field in (("gate", "gate_proj"), ("up", "up_proj")):
            _add(model, p + "mlp/" + role, store, sp + "mlp." + field + ".weight",
                 (intermediate, hidden), inputs=(mlp_in,))
        _add(model, p + "mlp/down", store, sp + "mlp.down_proj.weight",
             (hidden, intermediate), inputs=(p + "mlp_product",))


def _build_vae(model, store):
    for name in list(store.weight_map):
        shape = tuple(store.describe(name).shape)
        _add(model, "vae/" + name, store, name, shape)


def build_model(base: SafetensorsSource, *, components=("dit", "text"),
                resource_overrides=None) -> Model:
    if base.path.is_file():
        raise ValueError("point --model at the pipeline root directory")
    root = base.path
    records = {}
    sources = {}
    if "dit" in components:
        index = root / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
        records["dit"] = {"config": _load_config(root / "transformer" / "config.json")}
        sources["dit"] = SafetensorsSource(index)
    if "text" in components:
        records["text"] = {"config": _load_config(root / "text_encoder" / "config.json")}
        sources["text"] = SafetensorsSource(root / "text_encoder")
    if "vae" in components:
        records["vae"] = {"config": _load_config(root / "vae" / "config.json")}
        sources["vae"] = SafetensorsSource(
            root / "vae" / "diffusion_pytorch_model.safetensors"
        )
    if not records:
        raise ValueError("select at least one of dit,encoder")

    model = Model(components=records)
    if "dit" in components:
        _build_dit(model, sources["dit"], records["dit"]["config"])
    if "text" in components:
        _build_text_encoder(model, sources["text"], records["text"]["config"])
    if "vae" in components:
        _build_vae(model, sources["vae"])
    return model


def qwen_image_fp8(model, recipe: Recipe, sources):
    recipe.assign(
        [
            "dit/layers/*/attention/query",
            "dit/layers/*/attention/key",
            "dit/layers/*/attention/value",
            "dit/layers/*/attention/output",
            "dit/layers/*/mlp/gate",
            "dit/layers/*/mlp/up",
            "dit/layers/*/mlp/down",
            "dit/txt_in/in",
            "dit/txt_in/out",
            "text/layers/*/attention/query",
            "text/layers/*/attention/key",
            "text/layers/*/attention/value",
            "text/layers/*/attention/output",
            "text/layers/*/mlp/gate",
            "text/layers/*/mlp/up",
            "text/layers/*/mlp/down",
        ],
        format=FP8,
        method=FP8_METHOD,
        activation_policy="AllowA8",
    )
    recipe.assign("text/embed_tokens", format=FP8, method=FP8_METHOD)

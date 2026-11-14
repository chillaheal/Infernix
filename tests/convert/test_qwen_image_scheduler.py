"""The Qwen-Image converter packs the scheduler config as the artifact's fourth component.

The `.infernix` artifact is the single source of truth for the diffusion scheduler: the converter
copies ``scheduler/scheduler_config.json`` verbatim into ``components["scheduler"]["config"]``,
adds no field, removes none, renames none, coerces none, and refuses a pipeline root that does not
carry the file. Nothing downstream may judge behaviour by ``_diffusers_version``; it is preserved
for provenance only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from tools.convert.qwen_image import build_model, is_qwen_image
from tools.convert.sources.safetensors import SafetensorsSource

FIXTURE = Path(__file__).parent / "fixtures" / "qwen_image_scheduler_config.json"

# The reference checkpoint's own scheduler document, copied byte-for-byte into
# tests/convert/fixtures/qwen_image_scheduler_config.json. Pinned here so a fixture that drifts
# from the shipped values is a failure rather than a silent pass.
SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.37.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# The real diffusers checkpoint that produced the live artifact, when it is beside this checkout.
REFERENCE_SCHEDULER = (
    Path(__file__).resolve().parents[2].parent
    / "image-spike"
    / "artifact"
    / "scheduler"
    / "scheduler_config.json"
)

# A pipeline root small enough to convert on the host: the geometry is the smallest the
# adapter's own declarations accept, and every tensor it reads is present.
DIT = {
    "_class_name": "QwenImage21Transformer2DModel",
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "mlp_ratio": 2,
    "in_channels": 4,
    "patch_size": 2,
    "out_channels": 4,
    "num_layers": 1,
    "context_in_dim": 8,
}

TEXT = {
    "_class_name": "QwenImage21VLTextModel",
    "text_config": {
        "hidden_size": 16,
        "vocab_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 32,
    },
}

VAE = {"_class_name": "AutoencoderKLQwenImage21"}


def _dit_shapes(cfg: dict) -> dict[str, tuple[int, ...]]:
    hidden = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    intermediate = hidden * cfg["mlp_ratio"]
    image_dim = cfg["in_channels"] * cfg["patch_size"] * cfg["patch_size"]
    out_dim = cfg["out_channels"] * cfg["patch_size"] * cfg["patch_size"]
    shapes = {
        "img_in.weight": (hidden, image_dim),
        "proj_out.weight": (out_dim, hidden),
        "modulation.1.weight": (4 * hidden, hidden),
        "norm_out.linear.weight": (hidden, hidden),
        "txt_in.in_layer.weight": (hidden, cfg["context_in_dim"]),
        "txt_in.out_layer.weight": (hidden, hidden),
        "txt_in.text_norm.weight": (cfg["context_in_dim"],),
        "time_text_embed.timestep_embedder.linear_1.weight": (hidden, 256),
        "time_text_embed.timestep_embedder.linear_2.weight": (hidden, hidden),
    }
    for i in range(cfg["num_layers"]):
        block = f"transformer_blocks.{i}."
        for name in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
                     "attn.to_out.0.weight"):
            shapes[block + name] = (hidden, hidden)
        for name in ("attn.norm_q.weight", "attn.norm_k.weight"):
            shapes[block + name] = (cfg["attention_head_dim"],)
        for name in ("img_mlp.gate_layer.weight", "img_mlp.proj.weight"):
            shapes[block + name] = (intermediate, hidden)
        shapes[block + "img_mlp.out.weight"] = (hidden, intermediate)
    return shapes


def _text_shapes(cfg: dict) -> dict[str, tuple[int, ...]]:
    text = cfg["text_config"]
    hidden = text["hidden_size"]
    head_dim = text["head_dim"]
    query = text["num_attention_heads"] * head_dim
    key_value = text["num_key_value_heads"] * head_dim
    intermediate = text["intermediate_size"]
    shapes = {
        "model.language_model.embed_tokens.weight": (text["vocab_size"], hidden),
        "model.language_model.norm.weight": (hidden,),
    }
    for i in range(text["num_hidden_layers"]):
        layer = f"model.language_model.layers.{i}."
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            shapes[layer + name] = (hidden,)
        for name, width in (("q_proj", query), ("k_proj", key_value), ("v_proj", key_value)):
            shapes[layer + f"self_attn.{name}.weight"] = (width, hidden)
        for name in ("q_norm", "k_norm"):
            shapes[layer + f"self_attn.{name}.weight"] = (head_dim,)
        shapes[layer + "self_attn.o_proj.weight"] = (hidden, query)
        for name in ("gate_proj", "up_proj"):
            shapes[layer + f"mlp.{name}.weight"] = (intermediate, hidden)
        shapes[layer + "mlp.down_proj.weight"] = (hidden, intermediate)
    return shapes


def _tensors(shapes: dict[str, tuple[int, ...]]) -> dict:
    return {name: torch.zeros(*shape) for name, shape in shapes.items()}


def pipeline_root(root: Path, *, scheduler: bool = True) -> SafetensorsSource:
    """A minimal but complete Qwen-Image pipeline root.

    ``scheduler=False`` drops the scheduler subfolder, which a fail-closed converter must refuse.
    """
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))

    transformer = root / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text(json.dumps(DIT))
    dit_shard = "diffusion_pytorch_model.safetensors"
    dit_shapes = _dit_shapes(DIT)
    save_file(_tensors(dit_shapes), transformer / dit_shard)
    (transformer / f"{dit_shard}.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict.fromkeys(dit_shapes, dit_shard)})
    )

    text_encoder = root / "text_encoder"
    text_encoder.mkdir()
    (text_encoder / "config.json").write_text(json.dumps(TEXT))
    text_shapes = _text_shapes(TEXT)
    save_file(_tensors(text_shapes), text_encoder / "model.safetensors")

    vae = root / "vae"
    vae.mkdir()
    (vae / "config.json").write_text(json.dumps(VAE))
    save_file(_tensors({"encoder.conv_in.weight": (8, 4, 3, 3), "encoder.norm.weight": (8,)}),
              vae / "diffusion_pytorch_model.safetensors")

    if scheduler:
        scheduler_dir = root / "scheduler"
        scheduler_dir.mkdir()
        (scheduler_dir / "scheduler_config.json").write_bytes(FIXTURE.read_bytes())
    return SafetensorsSource(root)


def _model(tmp_path: Path, *, scheduler: bool = True, components=("dit", "text", "vae")):
    base = pipeline_root(tmp_path / "artifact", scheduler=scheduler)
    assert is_qwen_image(base), "the fixture must be a genuine Qwen-Image pipeline root"
    try:
        return build_model(base, components=components)
    finally:
        base.close()


def test_scheduler_is_the_fourth_component_copied_verbatim(tmp_path):
    model = _model(tmp_path)
    components = model.components

    assert set(components) == {"dit", "text", "vae", "scheduler"}
    loaded = json.loads(FIXTURE.read_text())
    # The record is exactly the read document: no synthesized field beside it, no field dropped,
    # no rename, and none of the source's own fields added.
    assert components["scheduler"] == {"config": loaded}
    # Serialized equality too, so a reordered key or a coerced 1.0 -> 1 or false -> 0 fails even
    # though it compares equal as data.
    assert json.dumps(components["scheduler"]["config"]) == json.dumps(loaded)
    assert list(components["scheduler"]["config"]) == list(loaded)
    # The weight components still carry their own config and nothing else changed.
    for name in ("dit", "text", "vae"):
        assert set(components[name]) == {"config"}
    # A metadata-only component owns no weights: none were fabricated for it.
    assert not [name for name in model.parameters if name.startswith("scheduler/")]


def test_scheduler_document_holds_every_field_the_reference_checkpoint_ships(tmp_path):
    config = _model(tmp_path).components["scheduler"]["config"]

    # Exact whole-document equality: every field the reference ships, and no other.
    assert config == SCHEDULER
    # Provenance fields survive untouched; the runtime reads them as data, not as approval.
    assert config["_class_name"] == "FlowMatchEulerDiscreteScheduler"
    assert config["_diffusers_version"] == "0.37.0.dev0"
    assert config["use_dynamic_shifting"] is True
    assert config["time_shift_type"] == "exponential"
    assert config["base_image_seq_len"] == 256
    assert config["base_shift"] == 0.5
    assert config["max_image_seq_len"] == 8192
    assert config["max_shift"] == 0.9
    assert config["num_train_timesteps"] == 1000
    assert config["shift_terminal"] == 0.02
    assert config["shift"] == 1.0
    assert config["invert_sigmas"] is False
    assert config["stochastic_sampling"] is False
    assert config["use_beta_sigmas"] is False
    assert config["use_exponential_sigmas"] is False
    assert config["use_karras_sigmas"] is False


def test_committed_fixture_is_the_reference_checkpoint_file_byte_for_byte():
    if not REFERENCE_SCHEDULER.is_file():
        pytest.skip(
            f"the reference diffusers checkpoint is not beside this checkout "
            f"({REFERENCE_SCHEDULER}); the committed fixture is the byte-for-byte copy taken "
            "from it and its values are pinned by "
            "test_scheduler_document_holds_every_field_the_reference_checkpoint_ships"
        )
    assert FIXTURE.read_bytes() == REFERENCE_SCHEDULER.read_bytes()


def test_a_root_without_the_scheduler_config_refuses_conversion(tmp_path):
    with pytest.raises(FileNotFoundError) as raised:
        _model(tmp_path, scheduler=False)

    message = str(raised.value)
    # No silent omission and no synthesized default: the failure names the file it requires.
    assert str(tmp_path / "artifact" / "scheduler" / "scheduler_config.json") in message


@pytest.mark.parametrize("components", [(), ("mtp",)])
def test_the_scheduler_is_never_packed_as_the_only_component(tmp_path, components):
    # A metadata-only scheduler is packed alongside a selected component, never on its own: the
    # guard that demands at least one weight component still holds.
    base = pipeline_root(tmp_path / "artifact")
    try:
        with pytest.raises(ValueError, match="select at least one"):
            build_model(base, components=components)
    finally:
        base.close()

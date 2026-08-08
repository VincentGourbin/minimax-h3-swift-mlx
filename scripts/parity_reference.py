#!/usr/bin/env python3
"""Dump reference tensors from the diffusers MiniMax-H3 implementation for Swift parity checks.

Runs the small components on fixed inputs and writes .safetensors files that the Swift side
(`minimax-h3 parity` command, to come) reloads and compares against its own outputs.

Requires: pip install torch safetensors, plus a diffusers checkout of the `minimax-h3` branch
(PYTHONPATH=<checkout>/src). Heavy components (text encoder, transformer) are validated per-layer
with truncated loads, not full forwards.

Usage:
  python3 scripts/parity_reference.py audio-vae [--models-dir $H3_MODELS_DIR]
  python3 scripts/parity_reference.py video-vae
  python3 scripts/parity_reference.py text-layer0   # embeddings + decoder layer 0 only
  python3 scripts/parity_reference.py dit-block0    # projections + transformer block 0 only
"""

import argparse
import os
import pathlib
import sys

import torch
from safetensors.torch import save_file


def out_dir(models_dir: pathlib.Path) -> pathlib.Path:
    path = models_dir / "parity"
    path.mkdir(exist_ok=True)
    return path


def dump_audio_vae(models_dir: pathlib.Path) -> None:
    from diffusers import AutoencoderKLMiniMaxH3Audio

    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(models_dir, subfolder="audio_vae")
    vae.eval()
    torch.manual_seed(1234)
    latents = torch.randn(2, 32, 40, dtype=torch.float32)
    with torch.no_grad():
        waveform = vae.decode(latents, return_dict=False)[0]
    save_file(
        {"latents": latents, "waveform": waveform.contiguous()},
        out_dir(models_dir) / "audio_vae_decode.safetensors",
    )
    print("audio-vae:", tuple(waveform.shape), "peak", waveform.abs().max().item())


def dump_video_vae(models_dir: pathlib.Path) -> None:
    from diffusers import AutoencoderKLMiniMaxH3

    vae = AutoencoderKLMiniMaxH3.from_pretrained(models_dir, subfolder="vae")
    vae.eval()
    torch.manual_seed(1234)
    latents = torch.randn(1, 24, 7, 8, 8, dtype=torch.float32)
    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]
    save_file(
        {"latents": latents, "video": video.contiguous()},
        out_dir(models_dir) / "video_vae_decode.safetensors",
    )
    print("video-vae:", tuple(video.shape))


def dump_video_vae_encode(models_dir: pathlib.Path) -> None:
    """Keyframe encode path: _encode_clip (spatial encoder, tiled) + the full conditioning
    recipe (posterior sampled at seed 42, fp16 rounding, normalization, patchify)."""
    import sys as _sys

    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    vae = AutoencoderKLMiniMaxH3.from_pretrained(models_dir, subfolder="vae")
    vae.eval()
    torch.manual_seed(1234)
    # Synthetic keyframe in ImageNet-normalized space, 384x384 so the 256px tiling kicks in.
    pixels = torch.randn(1, 3, 1, 384, 384, dtype=torch.float32) * 0.5
    with torch.no_grad():
        moments = vae._encode_clip(pixels)

    posterior = DiagonalGaussianDistribution(moments)
    latents = posterior.sample(generator=torch.Generator().manual_seed(42))
    latents = latents.to(torch.float16).float()
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    normalized = (latents - latents_mean) / latents_std

    _sys.path.insert(0, "src")
    from diffusers.modular_pipelines.minimax_h3.packing import patchify_video_latents

    rows = patchify_video_latents(normalized, (1, 2, 2))
    save_file(
        {
            "pixels": pixels,
            "moments": moments.contiguous(),
            "condition_rows": rows.contiguous(),
        },
        out_dir(models_dir) / "video_vae_encode.safetensors",
    )
    print("video-vae-encode:", tuple(moments.shape), "->", tuple(rows.shape))


def dump_vision_tower(models_dir: pathlib.Path) -> None:
    """Vision tower alone on a processor-produced synthetic image: embeds + deepstack."""
    import json

    import numpy as np
    from safetensors import safe_open

    text_dir = models_dir / "text_encoder"
    index = json.loads((text_dir / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {k: s for k, s in index.items() if k.startswith("model.visual.")}
    weights = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(text_dir / shard, framework="pt") as handle:
            for key in handle.keys():
                if key in wanted:
                    weights[key] = handle.get_tensor(key)

    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = AutoConfig.from_pretrained(models_dir, subfolder="text_encoder")
    visual = Qwen3VLVisionModel(config.vision_config)  # ~0.4B params, plain CPU construct
    visual.load_state_dict({k[len("model.visual."):]: v for k, v in weights.items()})
    visual = visual.eval().to(torch.float32)

    processor = AutoProcessor.from_pretrained(models_dir, subfolder="processor")
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (224, 320, 3), dtype=np.uint8)
    batch = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values, grid_thw = batch["pixel_values"].to(torch.float32), batch["image_grid_thw"]
    with torch.no_grad():
        out = visual(pixel_values, grid_thw)
    save_file(
        {
            "pixel_values": pixel_values.contiguous(),
            "grid_thw": grid_thw,
            "embeds": out.pooler_output.float().contiguous(),
            "deepstack_0": out.deepstack_features[0].float().contiguous(),
            "deepstack_2": out.deepstack_features[2].float().contiguous(),
        },
        out_dir(models_dir) / "vision_tower.safetensors",
    )
    print("vision-tower:", tuple(grid_thw.tolist()[0]), tuple(out.pooler_output.shape))


def dump_text_layer0(models_dir: pathlib.Path) -> None:
    """Embeddings + first decoder layer only — enough to catch RoPE/GQA/norm mistakes cheaply."""
    import json

    from safetensors import safe_open

    text_dir = models_dir / "text_encoder"
    index = json.loads((text_dir / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {
        key: shard
        for key, shard in index.items()
        if key.startswith(("model.language_model.embed_tokens.", "model.language_model.layers.0."))
    }
    weights = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(text_dir / shard, framework="pt") as handle:
            for key in handle.keys():
                if key in wanted:
                    weights[key] = handle.get_tensor(key)

    from transformers import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextDecoderLayer,
        Qwen3VLTextRotaryEmbedding,
    )

    config = Qwen3VLTextConfig.from_pretrained(models_dir, subfolder="text_encoder")
    torch.manual_seed(0)
    layer = Qwen3VLTextDecoderLayer(config, layer_idx=0)
    prefix = "model.language_model.layers.0."
    layer.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})
    layer = layer.eval().to(torch.float32)

    embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    embed.load_state_dict({"weight": weights["model.language_model.embed_tokens.weight"]})
    embed = embed.eval().to(torch.float32)

    token_ids = torch.tensor([[3838, 374, 264, 1273, 11, 264, 2766, 5021]])
    hidden = embed(token_ids)
    rotary = Qwen3VLTextRotaryEmbedding(config=config)
    position_ids = torch.arange(token_ids.shape[1])[None].expand(3, 1, -1)
    cos, sin = rotary(hidden, position_ids)
    # A standalone decoder layer applies no causal mask on its own — build the additive 4D
    # causal mask the full model would pass, or the reference computes bidirectional attention.
    length = token_ids.shape[1]
    causal = torch.triu(torch.full((length, length), float("-inf")), diagonal=1)[None, None]
    with torch.no_grad():
        out = layer(
            hidden,
            attention_mask=causal,
            position_embeddings=(cos, sin),
            position_ids=position_ids,
        )
    out = out[0] if isinstance(out, tuple) else out
    save_file(
        {"token_ids": token_ids, "embeddings": hidden, "layer0_out": out.contiguous()},
        out_dir(models_dir) / "text_layer0.safetensors",
    )
    print("text-layer0:", tuple(out.shape), "rms", out.pow(2).mean().sqrt().item())


def dump_dit_block0(models_dir: pathlib.Path) -> None:
    """Input projections + transformer block 0 of the DiT, on a tiny synthetic packed sequence."""
    import json

    from safetensors import safe_open

    tr_dir = models_dir / "transformer"
    index = json.loads(
        (tr_dir / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    prefixes = (
        "proj_in.", "audio_proj_in.", "context_embedder.", "time_embedder.", "token_refiner.",
        "transformer_blocks.0.", "norm_out.", "proj_out.", "audio_proj_out.",
    )
    wanted = {key: shard for key, shard in index.items() if key.startswith(prefixes)}
    weights = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(tr_dir / shard, framework="pt") as handle:
            for key in handle.keys():
                if key in wanted:
                    weights[key] = handle.get_tensor(key)

    from diffusers import MiniMaxH3Transformer3DModel

    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(num_layers=1)
    model.load_state_dict(weights, strict=False, assign=True)
    # The rope inv_freq buffer is computed, not loaded — rebuild it off the meta device.
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

    model.rope = MiniMaxH3RotaryPosEmbed(
        rope_freq_dim=model.config.rope_freq_dim, rope_theta=model.config.rope_theta
    )
    model.eval()

    torch.manual_seed(7)
    num_text, latent_frames, latent_hw, audio_latents = 4, 2, 4, 10
    rows_per_frame = (latent_hw // 2) * (latent_hw // 2)
    video_rows = torch.randn(1, latent_frames * rows_per_frame, 96)
    audio_rows = torch.randn(1, audio_latents * 2, 32)
    text_embeds = torch.randn(1, num_text, 5120) * 0.02

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from diffusers.modular_pipelines.minimax_h3.packing import (
        build_packed_sequence,
        build_row_timesteps,
    )

    layout = build_packed_sequence(
        torch.ones(num_text, dtype=torch.long), latent_frames, latent_hw, latent_hw,
        audio_latents, (1, 2, 2),
    )
    timesteps, timestep_indices = build_row_timesteps(layout, 0.0, 0.0, 0.999, 1.0)
    with torch.no_grad():
        video_out, audio_out = model(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=text_embeds.to(torch.bfloat16),
            timestep=timesteps,
            timestep_indices=timestep_indices,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            return_dict=False,
        )
    save_file(
        {
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "text_embeds": text_embeds,
            "position_ids": layout.position_ids.to(torch.float32),
            "video_out": video_out.float().contiguous(),
            "audio_out": audio_out.float().contiguous(),
        },
        out_dir(models_dir) / "dit_block0.safetensors",
    )
    print("dit-block0:", tuple(video_out.shape), tuple(audio_out.shape))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["audio-vae", "video-vae", "video-vae-encode", "vision-tower", "text-layer0", "dit-block0"])
    parser.add_argument("--models-dir", default=os.environ.get("H3_MODELS_DIR", "/tmp/MiniMax-H3"))
    args = parser.parse_args()
    models_dir = pathlib.Path(args.models_dir)
    {
        "audio-vae": dump_audio_vae,
        "video-vae": dump_video_vae,
        "video-vae-encode": dump_video_vae_encode,
        "vision-tower": dump_vision_tower,
        "text-layer0": dump_text_layer0,
        "dit-block0": dump_dit_block0,
    }[args.component](models_dir)


if __name__ == "__main__":
    main()

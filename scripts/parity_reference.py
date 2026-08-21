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


def dump_keyframe_preprocess(models_dir: pathlib.Path) -> None:
    """Canvas preparation (stretch + cover-crop, PIL LANCZOS) and Qwen2-VL patchify on it."""
    import numpy as np
    from PIL import Image

    from diffusers.modular_pipelines.minimax_h3.packing import prepare_keyframe_image

    rng = np.random.default_rng(11)
    original = rng.integers(0, 255, (311, 517, 3), dtype=np.uint8)  # odd size, both axes resample
    image = Image.fromarray(original, mode="RGB")
    canvas_height, canvas_width = 256, 448
    stretched = prepare_keyframe_image(image, canvas_height, canvas_width, stretch=True)
    covered = prepare_keyframe_image(image, canvas_height, canvas_width, stretch=False)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(models_dir, subfolder="processor")
    batch = processor.image_processor(images=[stretched], return_tensors="pt")
    save_file(
        {
            "original": torch.from_numpy(original),
            "canvas_stretch": torch.from_numpy(np.asarray(stretched)),
            "canvas_cover": torch.from_numpy(np.asarray(covered)),
            "pixel_values": batch["pixel_values"].to(torch.float32).contiguous(),
            "grid_thw": batch["image_grid_thw"],
        },
        out_dir(models_dir) / "keyframe_preprocess.safetensors",
    )
    print(
        "keyframe-preprocess:", stretched.size, covered.size,
        tuple(batch["pixel_values"].shape), batch["image_grid_thw"].tolist(),
    )


def dump_text_layer0_mm(models_dir: pathlib.Path) -> None:
    """The fl2va conditioner path through real layer 0: presentation token ids, mrope 3D
    positions, vision-embed injection at the image pads, deepstack tap 0 after the layer."""
    import json

    import numpy as np
    from safetensors import safe_open

    text_dir = models_dir / "text_encoder"
    index = json.loads((text_dir / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {
        key: shard
        for key, shard in index.items()
        if key.startswith(
            ("model.language_model.embed_tokens.", "model.language_model.layers.0.", "model.visual.")
        )
    }
    weights = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(text_dir / shard, framework="pt") as handle:
            for key in handle.keys():
                if key in wanted:
                    weights[key] = handle.get_tensor(key)

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLTextDecoderLayer,
        Qwen3VLTextRotaryEmbedding,
        Qwen3VLVisionModel,
    )

    config = AutoConfig.from_pretrained(models_dir, subfolder="text_encoder")

    # The presentation, exactly as diffusers' encode_prompt builds it: label + vision block +
    # verbatim prompt. Token ids come from the pipeline tokenizer (the one the Swift side loads);
    # the processor only contributes image preprocessing and the mm token type convention.
    processor = AutoProcessor.from_pretrained(models_dir, subfolder="processor")
    tokenizer = AutoTokenizer.from_pretrained(models_dir, subfolder="tokenizer")
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (224, 320, 3), dtype=np.uint8)  # already x32: no smart-resize
    batch = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values, grid_thw = batch["pixel_values"].to(torch.float32), batch["image_grid_thw"]

    num_image_tokens = int(grid_thw[0].prod()) // processor.image_processor.merge_size**2
    token_ids = tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    token_ids += (
        [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
        + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    )
    token_ids += tokenizer("a quiet forest at dawn", add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([token_ids])
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]))

    # get_rope_index is pure tensor math on the config — instantiate the model shell weightless.
    with init_empty_weights():
        shell = Qwen3VLModel(config)
    position_ids, _ = shell.get_rope_index(input_ids, mm_token_type_ids, image_grid_thw=grid_thw)

    visual = Qwen3VLVisionModel(config.vision_config)
    visual.load_state_dict({k[len("model.visual."):]: v for k, v in weights.items() if k.startswith("model.visual.")})
    visual = visual.eval().to(torch.float32)
    with torch.no_grad():
        vision_out = visual(pixel_values, grid_thw)
    image_embeds = vision_out.pooler_output

    embed = torch.nn.Embedding(config.text_config.vocab_size, config.text_config.hidden_size)
    embed.load_state_dict({"weight": weights["model.language_model.embed_tokens.weight"]})
    embed = embed.eval().to(torch.float32)
    hidden = embed(input_ids)
    image_mask = (input_ids == tokenizer.convert_tokens_to_ids("<|image_pad|>")).unsqueeze(-1)
    hidden = hidden.masked_scatter(image_mask, image_embeds)

    layer = Qwen3VLTextDecoderLayer(config.text_config, layer_idx=0)
    prefix = "model.language_model.layers.0."
    layer.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})
    layer = layer.eval().to(torch.float32)
    rotary = Qwen3VLTextRotaryEmbedding(config=config.text_config)
    cos, sin = rotary(hidden, position_ids)
    length = input_ids.shape[1]
    causal = torch.triu(torch.full((length, length), float("-inf")), diagonal=1)[None, None]
    with torch.no_grad():
        out = layer(
            hidden,
            attention_mask=causal,
            position_embeddings=(cos, sin),
            position_ids=position_ids,
        )
    out = out[0] if isinstance(out, tuple) else out
    # Deepstack tap 0 lands after layer 0 at the image-pad positions (`_deepstack_process`).
    out = out.clone()
    pad_mask = image_mask[..., 0]
    out[pad_mask] = out[pad_mask] + vision_out.deepstack_features[0].to(out.dtype)
    save_file(
        {
            "image": torch.from_numpy(image),
            "pixel_values": pixel_values.contiguous(),
            "grid_thw": grid_thw,
            "token_ids": input_ids,
            "mm_token_types": mm_token_type_ids,
            "position_ids": position_ids[:, 0].contiguous(),
            "layer0_out": out.contiguous(),
        },
        out_dir(models_dir) / "text_layer0_mm.safetensors",
    )
    print("text-layer0-mm:", len(token_ids), "tokens,", tuple(out.shape), "rms", out.pow(2).mean().sqrt().item())


def dump_conditioner_fl2va(models_dir: pathlib.Path) -> None:
    """Full-depth fl2va conditioner: a real keyframe + prompt through the vision tower and the
    text stack to hidden_states[50], everything in the release's bf16.

    HEAVY: loads 51 decoder layers (~53 GB) on CPU — run it alone. The stack is truncated to 51
    layers because `hidden_states[50]` is the *input* of layer index 50 (output of layer 49),
    which stays unnormalized only while at least one layer follows it.
    """
    import numpy as np
    from PIL import Image

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    keyframe_path = pathlib.Path(__file__).resolve().parent.parent / "docs/examples/t2va/fox-frame.png"
    prompt = "A red fox walking through a snowy forest at dusk, soft wind, distant birdsong."
    canvas_height, canvas_width = 256, 448

    # The geometry anchor is stretched onto the canvas — this IS `prepare_keyframe_image(...,
    # stretch=True)`, inlined so the probe needs no diffusers checkout (the canvas preparation
    # itself is covered bit-exactly by the `keyframe-preprocess` probe, which does run against
    # the reference implementation).
    image = Image.open(keyframe_path).convert("RGB")
    canvas = image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

    processor = AutoProcessor.from_pretrained(models_dir, subfolder="processor")
    tokenizer = AutoTokenizer.from_pretrained(models_dir, subfolder="tokenizer")
    batch = processor.image_processor(images=[canvas], return_tensors="pt")
    pixel_values, grid_thw = batch["pixel_values"], batch["image_grid_thw"]

    num_image_tokens = int(grid_thw[0].prod()) // processor.image_processor.merge_size**2
    token_ids = tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    token_ids += (
        [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
        + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    )
    token_ids += tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([token_ids])
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]))

    config = AutoConfig.from_pretrained(models_dir, subfolder="text_encoder")
    config.text_config.num_hidden_layers = 51
    model = Qwen3VLModel.from_pretrained(
        models_dir, subfolder="text_encoder", config=config, dtype=torch.bfloat16
    ).eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values.to(torch.bfloat16),
            image_grid_thw=grid_thw,
            use_cache=False,
            output_hidden_states=True,
        )
    hidden = outputs.hidden_states[50]
    # Intermediate depths too: a bug shows up as a jump at one layer, compounding bf16 rounding
    # as a smooth exponential ramp. Cheap — the forward already computed them.
    tensors = {
        "canvas": torch.from_numpy(np.asarray(canvas)),
        "token_ids": input_ids,
        "grid_thw": grid_thw,
        "hidden_50": hidden.float().contiguous(),
    }
    for depth in (1, 2, 3, 4, 5, 10, 20, 30, 40, 42, 44, 45, 46, 47, 48, 49):
        tensors[f"hidden_{depth}"] = outputs.hidden_states[depth].float().contiguous()
    save_file(tensors, out_dir(models_dir) / "conditioner_fl2va.safetensors")
    print(
        "conditioner-fl2va:", len(token_ids), "tokens,", tuple(hidden.shape),
        "rms", hidden.float().pow(2).mean().sqrt().item(),
    )


def dump_ref_normalize(models_dir: pathlib.Path) -> None:
    """ref2va reference normalization: the frame-rate resample, the canvas rescale, the
    truncate-then-resample soundtrack contract, and the image reference's own 2048 short edge.

    The two normalizers are called as `MiniMaxH3Ref2VASetupStep` static methods with the released
    checkpoint's geometry passed explicitly, so the probe needs no pipeline and no weights. The
    image branch is inlined the same way (`image_processor.resize` on a PIL image is exactly
    `image.resize((width, height), LANCZOS)` — the `keyframe-preprocess` probe covers that
    equivalence bit for bit).
    """
    import numpy as np
    from PIL import Image

    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep

    normalize_video = MiniMaxH3Ref2VASetupStep._normalize_video_condition
    normalize_audio = MiniMaxH3Ref2VASetupStep._normalize_audio_condition
    multiple, short_edge, max_pixels = 32, 768, 768 * 1344
    num_frames = 124  # 17 * 7 + 5, the 5 s canonical count
    max_duration = num_frames / 24

    rng = np.random.default_rng(1201)

    # 1. A video reference at an odd rate and an odd shape: both passes run.
    #    10 frames at 30 fps -> floor(10 * 0.8 + 0.5) = 8 slots; 175x99 resolves to 1344x768.
    source_frames = rng.integers(0, 256, (10, 99, 175, 3), dtype=np.uint8)
    video = normalize_video(source_frames, 30.0, num_frames, multiple, short_edge, max_pixels, 24.0)

    # 2. A video already at 24 fps and already on its canvas: the parity-exact pass-through.
    passthrough_source = rng.integers(0, 256, (2, 768, 1344, 3), dtype=np.uint8)
    passthrough = normalize_video(
        passthrough_source, 24.0, num_frames, multiple, short_edge, max_pixels, 24.0
    )

    # 3. A stereo soundtrack at 44.1 kHz, longer than the generated duration: truncated at the
    #    NATIVE rate, then resampled once.
    clock = np.arange(44100 * 8) / 44100
    soundtrack_source = np.stack(
        [np.sin(2 * np.pi * 440 * clock), 0.5 * np.sin(2 * np.pi * 997 * clock)]
    ).astype(np.float32)
    soundtrack = normalize_audio(torch.from_numpy(soundtrack_source), 44100, 32000, max_duration)

    # 4. A mono audio reference at 16 kHz: upmixed by channel repeat, then upsampled.
    clock = np.arange(16000 * 2) / 16000
    mono_source = np.stack([np.sin(2 * np.pi * 220 * clock)]).astype(np.float32)
    mono = normalize_audio(torch.from_numpy(mono_source), 16000, 32000, max_duration)

    # 5. An image reference: short edge to 2048, upscaling included, no area cap.
    image_source = rng.integers(0, 256, (700, 1000, 3), dtype=np.uint8)
    pil = Image.fromarray(image_source, mode="RGB")
    width, height = pil.size
    scale = 2048 / min(width, height)
    target_height = max(multiple, round(height * scale / multiple) * multiple)
    target_width = max(multiple, round(width * scale / multiple) * multiple)
    picture = pil.resize((target_width, target_height), Image.Resampling.LANCZOS)

    save_file(
        {
            "video_source": torch.from_numpy(source_frames),
            "video": torch.from_numpy(np.ascontiguousarray(video)),
            "passthrough_source": torch.from_numpy(passthrough_source),
            # `.clone()`: an already-normalized video is returned as a *view* of the source,
            # which is the pass-through contract itself and which safetensors refuses to share.
            "passthrough": torch.from_numpy(np.ascontiguousarray(passthrough)).clone(),
            "soundtrack_source": torch.from_numpy(soundtrack_source),
            "soundtrack": soundtrack.contiguous(),
            "mono_source": torch.from_numpy(mono_source),
            "mono": mono.contiguous(),
            "image_source": torch.from_numpy(image_source),
            "image": torch.from_numpy(np.asarray(picture)),
        },
        out_dir(models_dir) / "ref_normalize.safetensors",
    )
    print(
        "ref-normalize: video", video.shape, "passthrough", passthrough.shape,
        "soundtrack", tuple(soundtrack.shape), "mono", tuple(mono.shape),
        "image", picture.size,
    )


def dump_audio_vae_encode(models_dir: pathlib.Path) -> None:
    """The DAC trunk + causal-attention `pre_block` + `mean_proj`, i.e. what ref2va encodes a
    reference soundtrack with. Stereo is two batch items of the mono VAE; MiniMax-H3 takes the
    posterior *mean* and never samples, so `mean` is the tensor that matters."""
    from diffusers import AutoencoderKLMiniMaxH3Audio

    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(models_dir, subfolder="audio_vae")
    vae.eval()
    torch.manual_seed(77)
    # 64 latents' worth of samples, plus 137 to exercise the right-pad to the 800-sample hop.
    waveform = 0.3 * torch.randn(2, 1, 800 * 64 + 137, dtype=torch.float32)
    with torch.no_grad():
        posterior = vae.encode(waveform, return_dict=False)[0]
    save_file(
        {
            "waveform": waveform.squeeze(1).contiguous(),
            "mean": posterior.mean.contiguous(),
            "logs": posterior.logs.contiguous(),
        },
        out_dir(models_dir) / "audio_vae_encode.safetensors",
    )
    print(
        "audio-vae-encode:", tuple(posterior.mean.shape),
        "rms", posterior.mean.pow(2).mean().sqrt().item(),
    )


def dump_video_condition(models_dir: pathlib.Path) -> None:
    """A ref2va video reference through Qwen3-VL's *video* processor and the vision tower.

    Two things are being pinned. First the processor: `smart_resize` over the whole clip's
    `t * h * w` budget, the tail padded to the temporal patch, and the block-major patchify whose
    two temporal slots hold two *different* frames. Second, and the reason the Swift port can run
    the tower once per merged frame pair instead of once per clip: Qwen3-VL's vision attention is
    segmented per temporal group (`cu_seqlens` splits on T) and its rotary table carries no
    temporal component, so a `grid_t = N` call must equal N independent `grid_t = 1` calls. The
    dump carries both so the Swift side can check the equality it relies on.
    """
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
    visual = Qwen3VLVisionModel(config.vision_config)
    visual.load_state_dict({k[len("model.visual."):]: v for k, v in weights.items()})
    visual = visual.eval().to(torch.float32)

    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

    processor = AutoProcessor.from_pretrained(models_dir, subfolder="processor")
    temporal_patch = processor.video_processor.temporal_patch_size
    rng = np.random.default_rng(4242)
    # 5 frames of a 96x160 clip: 3 sampled at 2 fps from a 24 fps stream needs 25 frames, so the
    # sampling is exercised on a stream long enough to produce an odd count (tail padding).
    frames = rng.integers(0, 255, (25, 96, 160, 3), dtype=np.uint8)
    sampled, timestamps = MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(
        frames, 24.0, 2.0, temporal_patch
    )
    batch = processor.video_processor(
        videos=[np.stack(sampled)], do_sample_frames=False, return_tensors="pt"
    )
    pixel_values = batch["pixel_values_videos"].to(torch.float32)
    grid_thw = batch["video_grid_thw"]
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw[0])
    with torch.no_grad():
        whole = visual(pixel_values, grid_thw)
        per_group = [
            visual(
                pixel_values[group * grid_h * grid_w : (group + 1) * grid_h * grid_w],
                torch.tensor([[1, grid_h, grid_w]]),
            )
            for group in range(grid_t)
        ]
    grouped = torch.cat([out.pooler_output for out in per_group])
    save_file(
        {
            "frames": torch.from_numpy(frames),
            "sampled_indices": torch.tensor([int(i) for i in range(len(sampled))]),
            "timestamps": torch.tensor(timestamps, dtype=torch.float32),
            "pixel_values": pixel_values.contiguous(),
            "grid_thw": grid_thw,
            "embeds": whole.pooler_output.float().contiguous(),
            "embeds_per_group": grouped.float().contiguous(),
            "deepstack_0": whole.deepstack_features[0].float().contiguous(),
        },
        out_dir(models_dir) / "video_condition.safetensors",
    )
    print(
        "video-condition:", len(sampled), "sampled ->", grid_thw.tolist(),
        "labels", [f"<{t:.1f} seconds>" for t in timestamps],
        "| grid_t=N vs N x grid_t=1 max|d|",
        (whole.pooler_output - grouped).abs().max().item(),
    )


def dump_conditioner_ref2va(models_dir: pathlib.Path) -> None:
    """Full-depth ref2va conditioner: an audio reference, a video reference with its soundtrack and
    an image reference, through the real setup + presentation steps and the text stack to
    hidden_states[50], everything in the release's bf16.

    HEAVY: loads 51 decoder layers (~53 GB) on CPU — run it alone.

    The reference list is deliberately `[audio, video+sound, image]`: it puts an `"<Audio 1>: "`
    label with no vision block first, then a soundtrack-bearing video whose own `"<Audio 2>: "`
    label precedes its `"<Video 1>: "`, then an image — so the per-modality numbering, the
    interleaving and the timestamped video blocks are all exercised in one pass.
    """
    import numpy as np
    from PIL import Image

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    prompt = (
        "subject_definitions:\n<Subject 1> is the room in <Picture 1>.\n\n"
        "summary:\n[reference generation + audio reference] A short handheld take in <Subject 1>."
    )
    num_frames = 124
    multiple, short_edge, max_pixels = 32, 768, 768 * 1344
    rng = np.random.default_rng(90210)

    # A standalone audio reference: 1.5 s of mono 16 kHz.
    clock = np.arange(int(1.5 * 16000)) / 16000
    voice = np.stack([np.sin(2 * np.pi * 180 * clock) * 0.4]).astype(np.float32)
    # A video reference with a soundtrack: 25 frames at 24 fps (3 sampled -> 2 blocks, odd count
    # so the tail padding runs), 96x160, stereo 32 kHz sound.
    clip = rng.integers(0, 255, (25, 96, 160, 3), dtype=np.uint8)
    clip_clock = np.arange(32000) / 32000
    clip_audio = np.stack(
        [np.sin(2 * np.pi * 300 * clip_clock), np.sin(2 * np.pi * 700 * clip_clock)]
    ).astype(np.float32) * 0.3
    # An image reference, deliberately small: it is upscaled to its own 2048 short edge.
    picture = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)

    references = [
        MiniMaxH3AudioReference(audio=torch.from_numpy(voice), sample_rate=16000),
        MiniMaxH3VideoReference(
            frames=clip, fps=24.0, audio=torch.from_numpy(clip_audio), sample_rate=32000
        ),
        MiniMaxH3ImageReference(image=Image.fromarray(picture, mode="RGB")),
    ]

    # Normalization, through the setup step's own arithmetic (the image branch inlined the same
    # way `ref-normalize` inlines it).
    normalize_audio = MiniMaxH3Ref2VASetupStep._normalize_audio_condition
    max_duration = num_frames / 24
    normalized = [
        MiniMaxH3AudioReference(
            audio=normalize_audio(torch.from_numpy(voice), 16000, 32000, max_duration),
            sample_rate=32000,
        ),
        MiniMaxH3VideoReference(
            frames=MiniMaxH3Ref2VASetupStep._normalize_video_condition(
                clip, 24.0, num_frames, multiple, short_edge, max_pixels, 24.0
            ),
            fps=24.0,
            audio=normalize_audio(torch.from_numpy(clip_audio), 32000, 32000, max_duration),
            sample_rate=32000,
        ),
    ]
    scale = 2048 / min(picture.shape[0], picture.shape[1])
    target_h = max(multiple, round(picture.shape[0] * scale / multiple) * multiple)
    target_w = max(multiple, round(picture.shape[1] * scale / multiple) * multiple)
    normalized.append(
        MiniMaxH3ImageReference(
            image=Image.fromarray(picture, mode="RGB").resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )
        )
    )

    processor = AutoProcessor.from_pretrained(models_dir, subfolder="processor")
    tokenizer = AutoTokenizer.from_pretrained(models_dir, subfolder="tokenizer")
    step = MiniMaxH3Ref2VATextEncoderStep()
    vision_inputs, image_counts, video_counts, video_timestamps = step._gather_vision_features(
        processor, normalized, 24.0
    )
    token_ids, token_tags = step._build_presentation(
        tokenizer, prompt, normalized, image_counts, video_counts, video_timestamps
    )
    input_ids = torch.tensor([token_ids])
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]))

    config = AutoConfig.from_pretrained(models_dir, subfolder="text_encoder")
    config.text_config.num_hidden_layers = 51
    model = Qwen3VLModel.from_pretrained(
        models_dir, subfolder="text_encoder", config=config, dtype=torch.bfloat16
    ).eval()
    vision_kwargs = {
        name: (value.to(torch.bfloat16) if name.startswith("pixel_") else value)
        for name, value in vision_inputs.items()
    }
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
            **vision_kwargs,
        )
    hidden = outputs.hidden_states[50]
    tensors = {
        "voice": torch.from_numpy(voice),
        "clip": torch.from_numpy(clip),
        "clip_audio": torch.from_numpy(clip_audio),
        "picture": torch.from_numpy(picture),
        "token_ids": input_ids,
        "token_tags": torch.tensor([token_tags]),
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": vision_inputs["image_grid_thw"],
        "video_grid_thw": vision_inputs["video_grid_thw"],
        "hidden_50": hidden.float().contiguous(),
    }
    for depth in (1, 2, 5, 10, 20, 30, 40, 45, 49):
        tensors[f"hidden_{depth}"] = outputs.hidden_states[depth].float().contiguous()
    save_file(tensors, out_dir(models_dir) / "conditioner_ref2va.safetensors")
    print(
        "conditioner-ref2va:", len(token_ids), "tokens,", tuple(hidden.shape),
        "rms", hidden.float().pow(2).mean().sqrt().item(),
        "| labels", [f"<{t:.1f} seconds>" for t in video_timestamps[0]],
    )


def dump_vision_tower_large(models_dir: pathlib.Path) -> None:
    """The vision tower at a ref2va IMAGE reference's geometry: a 2048-short-edge image, i.e. a
    128x128 patch grid and 16 384 tokens in ONE tower call.

    Every earlier tower probe ran at a few hundred patches (a 768-canvas keyframe is 16x28). A
    reference image is two orders of magnitude larger, which puts the learned 48x48 position table
    through a much wider interpolation and the tower's attention through a sequence it has never
    been checked at here. fp32 on both sides, like the other isolation probes.
    """
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
    visual = Qwen3VLVisionModel(config.vision_config)
    visual.load_state_dict({k[len("model.visual."):]: v for k, v in weights.items()})
    visual = visual.eval().to(torch.float32)

    processor = AutoProcessor.from_pretrained(models_dir, subfolder="processor")
    rng = np.random.default_rng(31337)
    image = rng.integers(0, 255, (2048, 2048, 3), dtype=np.uint8)
    batch = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values, grid_thw = batch["pixel_values"].to(torch.float32), batch["image_grid_thw"]
    with torch.no_grad():
        out = visual(pixel_values, grid_thw)
    save_file(
        {
            "image": torch.from_numpy(image),
            "grid_thw": grid_thw,
            "embeds": out.pooler_output.float().contiguous(),
            "deepstack_0": out.deepstack_features[0].float().contiguous(),
            "deepstack_2": out.deepstack_features[2].float().contiguous(),
        },
        out_dir(models_dir) / "vision_tower_large.safetensors",
    )
    print("vision-tower-large:", grid_thw.tolist(), tuple(out.pooler_output.shape))


def dump_ref2va_embeddings(models_dir: pathlib.Path) -> None:
    """`hidden_states[0]` of the ref2va presentation: the token embeddings with the vision features
    injected, and NOTHING else — no decoder layer runs.

    This is the discriminator the full-depth probe cannot be: if the port and the reference agree
    here, every later difference belongs to the text stack; if they disagree, the conditioning
    going in is already wrong and the depth profile is downstream of that. Cheap because the stack
    is truncated to a single layer — `hidden_states[0]` does not depend on how many follow.
    """
    import numpy as np
    from PIL import Image

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    # The same synthetic request as `conditioner-ref2va`, so the two probes are comparable.
    prompt = (
        "subject_definitions:\n<Subject 1> is the room in <Picture 1>.\n\n"
        "summary:\n[reference generation + audio reference] A short handheld take in <Subject 1>."
    )
    num_frames, multiple, short_edge, max_pixels = 124, 32, 768 * 1, 768 * 1344
    rng = np.random.default_rng(90210)
    clock = np.arange(int(1.5 * 16000)) / 16000
    voice = np.stack([np.sin(2 * np.pi * 180 * clock) * 0.4]).astype(np.float32)
    clip = rng.integers(0, 255, (25, 96, 160, 3), dtype=np.uint8)
    clip_clock = np.arange(32000) / 32000
    clip_audio = np.stack(
        [np.sin(2 * np.pi * 300 * clip_clock), np.sin(2 * np.pi * 700 * clip_clock)]
    ).astype(np.float32) * 0.3
    picture = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)

    normalize_audio = MiniMaxH3Ref2VASetupStep._normalize_audio_condition
    max_duration = num_frames / 24
    normalized = [
        MiniMaxH3AudioReference(
            audio=normalize_audio(torch.from_numpy(voice), 16000, 32000, max_duration),
            sample_rate=32000,
        ),
        MiniMaxH3VideoReference(
            frames=MiniMaxH3Ref2VASetupStep._normalize_video_condition(
                clip, 24.0, num_frames, multiple, short_edge, max_pixels, 24.0
            ),
            fps=24.0,
            audio=normalize_audio(torch.from_numpy(clip_audio), 32000, 32000, max_duration),
            sample_rate=32000,
        ),
    ]
    scale = 2048 / min(picture.shape[0], picture.shape[1])
    target_h = max(multiple, round(picture.shape[0] * scale / multiple) * multiple)
    target_w = max(multiple, round(picture.shape[1] * scale / multiple) * multiple)
    normalized.append(
        MiniMaxH3ImageReference(
            image=Image.fromarray(picture, mode="RGB").resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )
        )
    )

    processor = AutoProcessor.from_pretrained(models_dir, subfolder="processor")
    tokenizer = AutoTokenizer.from_pretrained(models_dir, subfolder="tokenizer")
    step = MiniMaxH3Ref2VATextEncoderStep()
    vision_inputs, image_counts, video_counts, video_timestamps = step._gather_vision_features(
        processor, normalized, 24.0
    )
    token_ids, _ = step._build_presentation(
        tokenizer, prompt, normalized, image_counts, video_counts, video_timestamps
    )
    input_ids = torch.tensor([token_ids])
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]))

    config = AutoConfig.from_pretrained(models_dir, subfolder="text_encoder")
    config.text_config.num_hidden_layers = 1  # hidden_states[0] does not depend on the rest
    model = Qwen3VLModel.from_pretrained(
        models_dir, subfolder="text_encoder", config=config, dtype=torch.bfloat16
    ).eval()
    vision_kwargs = {
        name: (value.to(torch.bfloat16) if name.startswith("pixel_") else value)
        for name, value in vision_inputs.items()
    }
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
            **vision_kwargs,
        )
    embedded = outputs.hidden_states[0]
    # The vision features on their own too, so a mismatch can be blamed on the tower or on the
    # injection rather than on "somewhere in between".
    with torch.no_grad():
        image_features = model.get_image_features(
            vision_kwargs["pixel_values"], vision_kwargs["image_grid_thw"]
        )
        video_features = model.get_video_features(
            vision_kwargs["pixel_values_videos"], vision_kwargs["video_grid_thw"]
        )

    # The same pixels through a standalone fp32 tower, so the four corners of the comparison
    # exist: ours/theirs x bf16/fp32. Two bf16 implementations of a 27-layer tower always diverge;
    # what matters is whether OURS is further from the fp32 truth than THEIRS is.
    import json as _json

    from safetensors import safe_open as _safe_open
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    _index = _json.loads(
        (models_dir / "text_encoder" / "model.safetensors.index.json").read_text()
    )["weight_map"]
    _wanted = {k: v for k, v in _index.items() if k.startswith("model.visual.")}
    _weights = {}
    for _shard in sorted(set(_wanted.values())):
        with _safe_open(models_dir / "text_encoder" / _shard, framework="pt") as _handle:
            for _key in _handle.keys():
                if _key in _wanted:
                    _weights[_key] = _handle.get_tensor(_key)
    _visual = Qwen3VLVisionModel(config.vision_config)
    _visual.load_state_dict({k[len("model.visual."):]: v for k, v in _weights.items()})
    _visual = _visual.eval().to(torch.float32)
    with torch.no_grad():
        image_features_fp32 = _visual(
            vision_inputs["pixel_values"].to(torch.float32), vision_inputs["image_grid_thw"]
        ).pooler_output
        video_features_fp32 = _visual(
            vision_inputs["pixel_values_videos"].to(torch.float32),
            vision_inputs["video_grid_thw"],
        ).pooler_output

    def first(features):
        """The merged vision tokens, whatever wrapper this transformers version returns.

        `get_image_features` has returned a bare tensor, a `ModelOutput` and a tuple of them
        across versions, so unwrap until a tensor falls out rather than assume any one shape.
        """
        seen = type(features).__name__
        for _ in range(4):
            if isinstance(features, torch.Tensor):
                return features
            for attribute in ("pooler_output", "last_hidden_state"):
                if hasattr(features, attribute):
                    features = getattr(features, attribute)
                    break
            else:
                if isinstance(features, (list, tuple)) and features:
                    features = features[0]
                else:
                    break
        raise TypeError(f"cannot find the vision tokens in a {seen}: got {type(features).__name__}")

    save_file(
        {
            # The media too, so the Swift side rebuilds the same request from this one file.
            "voice": torch.from_numpy(voice),
            "clip": torch.from_numpy(clip),
            "clip_audio": torch.from_numpy(clip_audio),
            "picture": torch.from_numpy(picture),
            "token_ids": input_ids,
            "mm_token_type_ids": mm_token_type_ids,
            "image_grid_thw": vision_inputs["image_grid_thw"],
            "video_grid_thw": vision_inputs["video_grid_thw"],
            "hidden_0": embedded.float().contiguous(),
            "image_features": first(image_features).float().contiguous(),
            "video_features": first(video_features).float().contiguous(),
            "image_features_fp32": image_features_fp32.float().contiguous(),
            "video_features_fp32": video_features_fp32.float().contiguous(),
            # The deepstack taps of the SAME bf16 call, so a replay can drive the text stack
            # entirely from the reference's own vision side and leave nothing of ours in it.
            **{
                f"image_deepstack_{level}": tap.float().contiguous()
                for level, tap in enumerate(image_features.deepstack_features)
            },
            **{
                f"video_deepstack_{level}": tap.float().contiguous()
                for level, tap in enumerate(video_features.deepstack_features)
            },
        },
        out_dir(models_dir) / "ref2va_embeddings.safetensors",
    )
    print(
        "  reference tower, its own bf16 vs fp32: image max|d|",
        (first(image_features).float() - image_features_fp32.float()).abs().max().item(),
        " video max|d|",
        (first(video_features).float() - video_features_fp32.float()).abs().max().item(),
    )
    print(
        "ref2va-embeddings:", len(token_ids), "tokens, hidden_0", tuple(embedded.shape),
        "image_features", tuple(first(image_features).shape),
        "video_features", tuple(first(video_features).shape),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["audio-vae", "video-vae", "video-vae-encode", "vision-tower", "text-layer0", "dit-block0", "keyframe-preprocess", "text-layer0-mm", "conditioner-fl2va", "ref-normalize", "audio-vae-encode", "video-condition", "conditioner-ref2va", "vision-tower-large", "ref2va-embeddings"])
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
        "keyframe-preprocess": dump_keyframe_preprocess,
        "text-layer0-mm": dump_text_layer0_mm,
        "conditioner-fl2va": dump_conditioner_fl2va,
        "ref-normalize": dump_ref_normalize,
        "audio-vae-encode": dump_audio_vae_encode,
        "video-condition": dump_video_condition,
        "conditioner-ref2va": dump_conditioner_ref2va,
        "vision-tower-large": dump_vision_tower_large,
        "ref2va-embeddings": dump_ref2va_embeddings,
    }[args.component](models_dir)


if __name__ == "__main__":
    main()

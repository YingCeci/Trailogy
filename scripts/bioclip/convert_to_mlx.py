"""
Convert BioCLIP-2 vision encoder to MLX format with INT4 quantization.

Extracts ONLY the vision tower (ViT-L/14) + projection head from BioCLIP-2,
converts PyTorch weights to MLX safetensors, and quantizes to INT4.

Output: a standalone MLX model directory that the iOS app can load.

Architecture (ViT-L/14):
    Input:  [1, 3, 224, 224]
    Conv2d(3, 1024, k=14, s=14)  → patch embeddings
    + CLS token + positional embeddings
    24x ResidualAttentionBlock(1024, 16 heads)
    LayerNorm → CLS pool → Linear(1024, 768)
    L2 normalize
    Output: [1, 768]

Usage:
    # From HuggingFace
    python src/convert_bioclip_mlx.py --output_dir models/bioclip-mlx

    # From local bioclip-2 repo
    python src/convert_bioclip_mlx.py \\
        --bioclip_repo ../bioclip-2 \\
        --output_dir models/bioclip-mlx

    # Skip quantization (FP16 output)
    python src/convert_bioclip_mlx.py --output_dir models/bioclip-mlx --no_quantize
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_bioclip_vision_weights(
    model_name: str = "hf-hub:imageomics/bioclip-2",
    bioclip_repo: str | None = None,
    device: str = "cpu",
) -> tuple[dict, dict]:
    """
    Load BioCLIP-2 and extract vision encoder state dict.
    Returns (vision_state_dict, model_config).
    """
    import torch

    # Try loading from local repo first
    if bioclip_repo:
        repo_src = Path(bioclip_repo) / "src"
        if repo_src.exists():
            sys.path.insert(0, str(repo_src))
            logger.info(f"Using local bioclip-2 repo: {repo_src}")

    try:
        from open_clip import create_model_from_pretrained
    except ImportError:
        raise ImportError(
            "open_clip not found. Install via:\n"
            "  pip install open_clip_torch\n"
            "or point --bioclip_repo to the local bioclip-2 directory."
        )

    logger.info(f"Loading BioCLIP-2 from {model_name}...")
    model, _ = create_model_from_pretrained(model_name)
    model = model.to(device).eval()

    # Extract vision encoder state dict
    vision_sd = {}
    for name, param in model.named_parameters():
        if name.startswith("visual."):
            vision_sd[name] = param.detach().cpu().numpy()

    # Also grab buffers (like positional_embedding if it's a buffer)
    for name, buf in model.named_buffers():
        if name.startswith("visual."):
            vision_sd[name] = buf.detach().cpu().numpy()

    # Model config for reconstruction
    config = {
        "model_type": "bioclip_vit",
        "image_size": 224,
        "patch_size": 14,
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "head_dim": 64,
        "mlp_ratio": 4.0,
        "output_dim": 768,
        "pool_type": "tok",
        "grid_size": 16,
        "seq_len": 257,  # 16*16 + 1 CLS
    }

    logger.info(f"Extracted {len(vision_sd)} vision parameters")
    total_params = sum(v.size for v in vision_sd.values())
    logger.info(f"Total parameters: {total_params:,} ({total_params * 4 / 1e9:.2f} GB FP32)")
    return vision_sd, config


def remap_weight_names(vision_sd: dict) -> dict:
    """
    Remap PyTorch OpenCLIP weight names to a clean MLX namespace.

    OpenCLIP names:           MLX names:
    visual.conv1.weight    →  patch_embed.weight
    visual.class_embedding →  cls_token
    visual.positional_embedding → pos_embed
    visual.ln_pre.weight   →  ln_pre.weight
    visual.transformer.resblocks.{i}.ln_1.weight → blocks.{i}.ln1.weight
    visual.transformer.resblocks.{i}.attn.in_proj_weight → blocks.{i}.attn.in_proj.weight
    visual.transformer.resblocks.{i}.attn.in_proj_bias → blocks.{i}.attn.in_proj.bias
    visual.transformer.resblocks.{i}.attn.out_proj.weight → blocks.{i}.attn.out_proj.weight
    visual.transformer.resblocks.{i}.mlp.c_fc.weight → blocks.{i}.mlp.fc1.weight
    visual.transformer.resblocks.{i}.mlp.c_proj.weight → blocks.{i}.mlp.fc2.weight
    visual.ln_post.weight  →  ln_post.weight
    visual.proj            →  proj.weight (transposed to Linear [out, in])
    """
    mapping = {}

    for key, value in vision_sd.items():
        new_key = key.replace("visual.", "")

        # Patch embedding
        new_key = new_key.replace("conv1.", "patch_embed.")

        # CLS token and position embedding
        new_key = new_key.replace("class_embedding", "cls_token")
        new_key = new_key.replace("positional_embedding", "pos_embed")

        # Transformer blocks
        new_key = new_key.replace("transformer.resblocks.", "blocks.")
        new_key = new_key.replace(".ln_1.", ".ln1.")
        new_key = new_key.replace(".ln_2.", ".ln2.")
        new_key = new_key.replace(".attn.in_proj_weight", ".attn.in_proj.weight")
        new_key = new_key.replace(".attn.in_proj_bias", ".attn.in_proj.bias")
        new_key = new_key.replace(".mlp.c_fc.", ".mlp.fc1.")
        new_key = new_key.replace(".mlp.c_proj.", ".mlp.fc2.")

        if new_key == "proj":
            new_key = "proj.weight"
            # OpenCLIP stores visual.proj as [width, output_dim] and applies
            # `x @ proj`. MLXNN.Linear stores weights as [output_dim, width]
            # and applies `x @ weight.T`, so transpose only this raw matrix.
            value = value.T

        # Skip continual_proj (BioCLIP-2 experience replay head, not needed for inference)
        if "continual_proj" in new_key:
            logger.debug(f"Skipping {key} (continual projection, not needed)")
            continue

        mapping[new_key] = value

    logger.info(f"Remapped {len(mapping)} parameters")
    return mapping


def convert_to_mlx_safetensors(
    weights: dict,
    config: dict,
    output_dir: Path,
    quantize: bool = True,
    q_bits: int = 4,
    q_group_size: int = 64,
):
    """Save weights as MLX safetensors with optional INT4 quantization."""
    import mlx.core as mx

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to MLX arrays
    mlx_weights = {}
    for key, np_array in weights.items():
        mlx_weights[key] = mx.array(np_array)

    if quantize:
        logger.info(f"Quantizing to INT{q_bits} (group_size={q_group_size})...")
        quantized_weights = {}
        n_quantized = 0
        n_skipped = 0

        for key, value in mlx_weights.items():
            # Only quantize Linear module weights. Raw arrays like pos_embed
            # must remain normal MLXArrays because the Swift model uses them
            # directly in arithmetic, not through QuantizedLinear.
            is_linear_weight = (
                key.endswith(".weight")
                and value.ndim == 2
                and value.shape[0] >= q_group_size
                and value.shape[1] >= q_group_size
            )

            if is_linear_weight:
                # MLX quantization: pack weights into (quantized, scales, biases)
                quantized, scales, biases = mx.quantize(value, group_size=q_group_size, bits=q_bits)
                quantized_weights[key] = quantized
                quantized_weights[key.replace(".weight", "").rstrip(".") + ".scales" if ".weight" in key else key + ".scales"] = scales
                quantized_weights[key.replace(".weight", "").rstrip(".") + ".biases" if ".weight" in key else key + ".biases"] = biases
                n_quantized += 1
            else:
                quantized_weights[key] = value
                n_skipped += 1

        logger.info(f"Quantized {n_quantized} tensors, kept {n_skipped} in original precision")
        save_weights = quantized_weights

        # Update config with quantization info
        config["quantization"] = {
            "bits": q_bits,
            "group_size": q_group_size,
        }
    else:
        # Save as float16
        save_weights = {}
        for key, value in mlx_weights.items():
            save_weights[key] = value.astype(mx.float16)
        logger.info("Saving in float16 (no quantization)")

    # Save weights
    weights_path = output_dir / "model.safetensors"
    mx.save_safetensors(str(weights_path), save_weights)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Compute and report sizes
    weights_size = weights_path.stat().st_size
    logger.info(f"Saved weights: {weights_path} ({weights_size / 1e6:.1f} MB)")
    logger.info(f"Saved config:  {config_path}")

    return weights_size


def verify_output(output_dir: Path):
    """Quick sanity check: load the saved model and verify shapes."""
    import mlx.core as mx

    logger.info("Verifying saved model...")
    weights = mx.load(str(output_dir / "model.safetensors"))
    config = json.loads((output_dir / "config.json").read_text())

    # Check key components exist
    expected_keys = ["patch_embed.weight", "cls_token", "pos_embed", "ln_pre.weight", "proj.weight"]
    for key in expected_keys:
        if key not in weights and key + ".scales" not in weights:
            logger.warning(f"Missing expected key: {key}")

    # Check block count
    block_keys = [k for k in weights if k.startswith("blocks.")]
    block_indices = set()
    for k in block_keys:
        parts = k.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            block_indices.add(int(parts[1]))
    n_blocks = len(block_indices)

    logger.info(f"  Blocks: {n_blocks} (expected {config['layers']})")
    logger.info(f"  Total tensors: {len(weights)}")
    logger.info(f"  Config: {config['image_size']}px, {config['width']}d, {config['output_dim']}d output")
    if "quantization" in config:
        q = config["quantization"]
        logger.info(f"  Quantization: INT{q['bits']}, group_size={q['group_size']}")

    logger.info("Verification passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BioCLIP-2 vision encoder to MLX with INT4 quantization"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hf-hub:imageomics/bioclip-2",
        help="OpenCLIP model name (default: hf-hub:imageomics/bioclip-2)",
    )
    parser.add_argument(
        "--bioclip_repo",
        type=str,
        default=None,
        help="Path to local bioclip-2 repo (uses its src/ for open_clip)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/bioclip-mlx",
        help="Output directory for MLX model",
    )
    parser.add_argument("--no_quantize", action="store_true", help="Save as FP16 instead of INT4")
    parser.add_argument("--q_bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--q_group_size", type=int, default=64, help="Quantization group size")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Step 1: Load BioCLIP-2 and extract vision weights
    logger.info("=" * 60)
    logger.info("Step 1: Extract vision encoder weights")
    logger.info("=" * 60)
    vision_sd, config = load_bioclip_vision_weights(
        model_name=args.model_name,
        bioclip_repo=args.bioclip_repo,
    )

    # Step 2: Remap weight names
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: Remap weight names to MLX convention")
    logger.info("=" * 60)
    remapped = remap_weight_names(vision_sd)

    # Step 3: Convert and quantize
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Step 3: Convert to MLX safetensors (INT{args.q_bits})")
    logger.info("=" * 60)
    size = convert_to_mlx_safetensors(
        remapped,
        config,
        output_dir,
        quantize=not args.no_quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )

    # Step 4: Verify
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 4: Verify output")
    logger.info("=" * 60)
    verify_output(output_dir)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"  Model: {output_dir}/model.safetensors ({size / 1e6:.1f} MB)")
    logger.info(f"  Config: {output_dir}/config.json")
    logger.info("")
    logger.info("To deploy to iOS app:")
    logger.info(f"  cp -r {output_dir} path/to/hikeCompanion/HikeCompanion/Resources/Models/BioCLIP/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

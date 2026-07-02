"""
Anima LoRA Test Grid Generator
===============================
Generates a comparison grid of images for different LoRA checkpoints and strengths,
using prompts from a TOML configuration file.

Usage:
    python anima_test_grid.py

Configure the lists at the top of this script before running.
"""

import argparse
import copy
import gc
import math
import os
import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import toml
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from tqdm import tqdm

from library import anima_models, anima_train_utils, anima_utils, hunyuan_image_utils, strategy_anima, strategy_base
from library.device_utils import clean_memory_on_device, synchronize_device
from library.sampling import load_prompts
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# ============================================================
# ⚙️  CONFIGURATION — modify these for your testing needs
# ============================================================

# --- Base model paths ---
DIT_PATH: str = "D:/Programs/ComfyUI-aki-v1.4/models/diffusion_models/anima_merged.safetensors"
VAE_PATH: str = "D:/Programs/ComfyUI-aki-v1.4/models/vae/qwen_image_vae.safetensors"
TEXT_ENCODER_PATH: str = "D:/Programs/ComfyUI-aki-v1.4/models/text_encoders/qwen_3_06b_base.safetensors"

# --- LoRA model paths to test ---
LORA_MODEL_PATHS: List[str] = [
    "D:/Programs/ComfyUI-aki-v1.4/models/loras/temp/Nekowuwu-000002.safetensors",
    "D:/Programs/ComfyUI-aki-v1.4/models/loras/temp/Nekowuwu-000003.safetensors",
    "D:/Programs/ComfyUI-aki-v1.4/models/loras/temp/Nekowuwu-000004.safetensors",
    "D:/Programs/ComfyUI-aki-v1.4/models/loras/temp/Nekowuwu-000005.safetensors",
    "D:/Programs/ComfyUI-aki-v1.4/models/loras/temp/Nekowuwu-000006.safetensors",
    "D:/Programs/ComfyUI-aki-v1.4/models/loras/temp/Nekowuwu-000007.safetensors",
    "D:/Programs/ComfyUI-aki-v1.4/models/loras/temp/Nekowuwu-000008.safetensors",
    "D:/Programs/ComfyUI-aki-v1.4/models/loras/temp/Nekowuwu-000009.safetensors",
]

# --- LoRA strengths (multipliers) to test ---
LORA_STRENGTHS: List[float] = [0.3, 0.5, 0.7, 0.9, 1.0]

# --- Prompt configuration file (TOML format, see configs/sample_prompts.toml) ---
PROMPT_CONFIG: str = "configs/sample_prompts.toml"

# --- Output directory ---
OUTPUT_DIR: str = "./outputs"

# --- Default inference settings (overridden by per-prompt settings in TOML) ---
DEFAULT_INFER_STEPS: int = 20
DEFAULT_GUIDANCE_SCALE: float = 5.0
DEFAULT_FLOW_SHIFT: float = 5.0
DEFAULT_SEED: int = 12345
DEFAULT_WIDTH: int = 512
DEFAULT_HEIGHT: int = 768

# --- Device ---
DEVICE: str = "cuda"  # or "cpu"

# --- Attention mode ---
ATTN_MODE: str = "torch"  # "torch", "flash", "sageattn", "xformers", "sdpa"

# --- Grid layout ---
GRID_CELL_MAX_SIZE: int = 512    # Each cell image is resized so its longer side ≤ this
GRID_LABEL_HEIGHT: int = 36      # Height of row/column label bands (pixels)
GRID_LABEL_WIDTH: int = 140      # Width of row label area (pixels)
GRID_FONT_SIZE: int = 14         # Label font size
GRID_GAP: int = 4                # Gap between cells (pixels)

# ============================================================
# End of configuration
# ============================================================


def make_base_args() -> SimpleNamespace:
    """Build a minimal args-like namespace for internal API compatibility."""
    return SimpleNamespace(
        dit=DIT_PATH,
        vae=VAE_PATH,
        text_encoder=TEXT_ENCODER_PATH,
        vae_chunk_size=None,
        vae_disable_cache=False,
        qwen_image_vae_2d=False,
        attn_mode=ATTN_MODE,
        fp8=False,
        fp8_scaled=False,
        text_encoder_cpu=False,
        device=DEVICE,
        lora_weight=None,
        lora_multiplier=None,
        lycoris=False,
        guidance_scale=DEFAULT_GUIDANCE_SCALE,
        infer_steps=DEFAULT_INFER_STEPS,
        flow_shift=DEFAULT_FLOW_SHIFT,
        seed=DEFAULT_SEED,
        image_size=[DEFAULT_HEIGHT, DEFAULT_WIDTH],
        negative_prompt="",
        output_type="images",
        no_metadata=True,
    )


# ---------------------------------------------------------------------------
#  Model loading (reused from anima_minimal_inference.py with adaptations)
# ---------------------------------------------------------------------------

def load_text_encoder(device: torch.device) -> torch.nn.Module:
    """Load Qwen3 text encoder to the given device (base model, no LoRA)."""
    text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        TEXT_ENCODER_PATH, dtype=torch.bfloat16, device=device
    )
    text_encoder.eval()
    return text_encoder


def load_text_encoder_with_lora(
    device: torch.device,
    lora_weight_path: str,
    lora_multiplier: float,
) -> torch.nn.Module:
    """Load Qwen3 text encoder, merging text-encoder LoRA weights."""
    lora_sd = load_file(lora_weight_path)
    lora_te_sd = {
        "model_" + k[len("lora_te_"):]: v
        for k, v in lora_sd.items()
        if k.startswith("lora_te_")
    }
    te_lora_weights = [lora_te_sd] if lora_te_sd else None
    te_lora_multipliers = [lora_multiplier] if lora_te_sd else None

    logger.info(
        f"  Loading Text Encoder with LoRA TE weights"
        f" ({len(lora_te_sd)} keys, multiplier={lora_multiplier})"
    )
    text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        TEXT_ENCODER_PATH,
        dtype=torch.bfloat16,
        device=device,
        lora_weights=te_lora_weights,
        lora_multipliers=te_lora_multipliers,
    )
    text_encoder.eval()
    return text_encoder


def load_dit_with_lora(
    device: torch.device,
    lora_weight_path: Optional[str],
    lora_multiplier: float,
) -> anima_models.Anima:
    """Load the Anima DiT model, optionally merging a single LoRA checkpoint."""
    lora_weights_list = None
    lora_multipliers = None

    if lora_weight_path is not None:
        logger.info(f"  Loading LoRA: {os.path.basename(lora_weight_path)}  multiplier={lora_multiplier}")
        lora_sd = load_file(lora_weight_path)
        lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
        lora_weights_list = [lora_sd]
        lora_multipliers = [lora_multiplier]

    model = anima_utils.load_anima_model(
        device=device,
        dit_path=DIT_PATH,
        attn_mode=ATTN_MODE,
        split_attn=True,
        loading_device=str(device),
        dit_weight_dtype=torch.bfloat16,
        fp8_scaled=False,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
    )
    model.to(device, dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    return model


def load_vae(device: torch.device):
    """Load the Qwen-Image VAE."""
    base_args = make_base_args()
    vae = anima_train_utils.load_qwen_image_vae(base_args, device="cpu", disable_mmap=True)
    vae.to(torch.bfloat16)
    vae.eval()
    return vae


# ---------------------------------------------------------------------------
#  Text embedding precomputation
# ---------------------------------------------------------------------------

def encode_prompt(
    text_encoder: torch.nn.Module,
    anima: anima_models.Anima,
    tokenize_strategy,
    encoding_strategy,
    prompt: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Encode a prompt into the embeddings expected by Anima.forward."""
    tokens = tokenize_strategy.tokenize(prompt)
    embed = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)

    crossattn_emb = anima._preprocess_text_embeds(
        source_hidden_states=embed[0].to(anima.device),
        target_input_ids=embed[2].to(anima.device),
        target_attention_mask=embed[3].to(anima.device),
        source_attention_mask=embed[1].to(anima.device),
    )
    crossattn_emb[~embed[3].bool()] = 0
    embed[0] = crossattn_emb
    embed[0] = embed[0].cpu()
    return {"embed": embed, "prompt": prompt}


# ---------------------------------------------------------------------------
#  Image generation
# ---------------------------------------------------------------------------

def generate_single(
    anima: anima_models.Anima,
    context: Dict[str, Any],
    context_null: Dict[str, Any],
    device: torch.device,
    height: int,
    width: int,
    infer_steps: int,
    guidance_scale: float,
    flow_shift: float,
    seed: int,
) -> torch.Tensor:
    """Run the full denoising loop and return a decoded image tensor [C, H, W] in [-1, 1]."""

    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    embed = context["embed"][0].to(device, dtype=torch.bfloat16)
    negative_embed = context_null["embed"][0].to(device, dtype=torch.bfloat16)

    num_channels_latents = anima_models.Anima.LATENT_CHANNELS
    shape = (
        1,
        num_channels_latents,
        1,  # frame dim
        height // 8,
        width // 8,
    )
    latents = randn_tensor(shape, generator=seed_g, device=device, dtype=torch.bfloat16)

    h_latent, w_latent = latents.shape[-2], latents.shape[-1]
    padding_mask = torch.zeros(1, 1, h_latent, w_latent, dtype=torch.bfloat16, device=device)

    timesteps, sigmas = hunyuan_image_utils.get_timesteps_sigmas(infer_steps, flow_shift, device)
    timesteps = timesteps / 1000.0
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    do_cfg = guidance_scale != 1.0

    for i, t in enumerate(timesteps):
        t_expand = t.expand(latents.shape[0])

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            noise_pred = anima(latents, t_expand, embed, padding_mask=padding_mask)

        if do_cfg:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                uncond_noise_pred = anima(latents, t_expand, negative_embed, padding_mask=padding_mask)
            noise_pred = uncond_noise_pred + guidance_scale * (noise_pred - uncond_noise_pred)

        latents = hunyuan_image_utils.step(latents, noise_pred, sigmas, i).to(latents.dtype)

    return latents


def decode_to_image(
    vae,
    latent: torch.Tensor,
    device: torch.device,
) -> Image.Image:
    """Decode a latent tensor [B, C, 1, H, W] into a PIL Image."""
    vae.to(device)
    with torch.no_grad():
        pixels = vae.decode_to_pixels(latent.to(device, dtype=vae.dtype))
    if pixels.ndim == 5:
        pixels = pixels.squeeze(2)  # [B, C, H, W]
    pixels = pixels.to("cpu", dtype=torch.float32)
    vae.to("cpu")

    sample = pixels[0]  # [C, H, W]
    sample = torch.clamp(sample, -1.0, 1.0)
    sample = ((sample + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    sample = sample.transpose(1, 2, 0)  # C,H,W -> H,W,C
    return Image.fromarray(sample)


# ---------------------------------------------------------------------------
#  Grid assembly
# ---------------------------------------------------------------------------

def _get_font() -> ImageFont.FreeTypeFont:
    """Try to get a reasonable TrueType font; fall back to default."""
    font_paths = [
        # Windows
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",       # Microsoft YaHei (Chinese support)
        "C:/Windows/Fonts/simhei.ttf",      # SimHei (Chinese support)
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, GRID_FONT_SIZE)
            except Exception:
                continue
    logger.warning("No suitable TrueType font found; using PIL default (labels may be tiny).")
    return ImageFont.load_default()


def make_grid(
    cell_images: List[List[Image.Image]],
    row_labels: List[str],
    col_labels: List[str],
    title: str = "",
) -> Image.Image:
    """
    Assemble a grid image with row/column labels.

    Args:
        cell_images: 2D list [row][col] of PIL Images.
        row_labels: Labels for each row (loRA paths).
        col_labels: Labels for each column (strengths).
        title: Optional overall title.

    Returns:
        Combined PIL Image.
    """
    n_rows = len(cell_images)
    n_cols = len(cell_images[0]) if n_rows > 0 else 0
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Empty grid.")

    font = _get_font()

    # Resize all cells to a consistent size
    cell_w, cell_h = GRID_CELL_MAX_SIZE, GRID_CELL_MAX_SIZE
    for r in range(n_rows):
        for c in range(n_cols):
            img = cell_images[r][c]
            # Scale so the longer side fits cell_max_size
            w, h = img.size
            scale = min(cell_w / w, cell_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            cell_images[r][c] = img.resize((new_w, new_h), Image.LANCZOS)

    # Determine actual cell dimensions (use the first cell as reference after resize)
    actual_cell_w = max(cell_images[r][c].size[0] for r in range(n_rows) for c in range(n_cols))
    actual_cell_h = max(cell_images[r][c].size[1] for r in range(n_rows) for c in range(n_cols))

    # Canvas dimensions
    title_h = GRID_LABEL_HEIGHT + 10 if title else 0
    total_w = GRID_LABEL_WIDTH + n_cols * (actual_cell_w + GRID_GAP) + GRID_GAP
    total_h = title_h + GRID_LABEL_HEIGHT + n_rows * (actual_cell_h + GRID_GAP) + GRID_GAP

    canvas = Image.new("RGB", (total_w, total_h), color=(40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    # Draw title
    if title:
        draw.text((total_w // 2, 5), title, fill=(255, 255, 255), font=font, anchor="ma")

    # Draw column labels
    for c in range(n_cols):
        x = GRID_LABEL_WIDTH + c * (actual_cell_w + GRID_GAP) + GRID_GAP + actual_cell_w // 2
        y = title_h + GRID_LABEL_HEIGHT // 2
        draw.text((x, y), col_labels[c], fill=(255, 255, 255), font=font, anchor="mm")

    # Draw row labels and cells
    for r in range(n_rows):
        # Row label
        ly = title_h + GRID_LABEL_HEIGHT + r * (actual_cell_h + GRID_GAP) + GRID_GAP + actual_cell_h // 2
        draw.text((GRID_LABEL_WIDTH // 2, ly), row_labels[r], fill=(255, 255, 255), font=font, anchor="mm")

        for c in range(n_cols):
            cx = GRID_LABEL_WIDTH + c * (actual_cell_w + GRID_GAP) + GRID_GAP
            cy = title_h + GRID_LABEL_HEIGHT + r * (actual_cell_h + GRID_GAP) + GRID_GAP

            # Center the cell image in its slot
            img = cell_images[r][c]
            offset_x = cx + (actual_cell_w - img.size[0]) // 2
            offset_y = cy + (actual_cell_h - img.size[1]) // 2

            # Draw cell background
            draw.rectangle([cx, cy, cx + actual_cell_w, cy + actual_cell_h], fill=(20, 20, 20))
            canvas.paste(img, (offset_x, offset_y))

    return canvas


# ---------------------------------------------------------------------------
#  Short label helpers
# ---------------------------------------------------------------------------

def short_lora_label(path: str, max_len: int = 30) -> str:
    """Derive a compact label from a LoRA file path."""
    name = os.path.splitext(os.path.basename(path))[0]
    if len(name) > max_len:
        name = name[:max_len - 3] + "..."
    return name


def strength_label(s: float) -> str:
    return f"strength={s:.1f}"


# ---------------------------------------------------------------------------
#  Main test procedure
# ---------------------------------------------------------------------------

def run_tests():
    # Validate inputs
    if not os.path.exists(DIT_PATH):
        logger.error(f"DiT checkpoint not found: {DIT_PATH}")
        sys.exit(1)
    if not os.path.exists(TEXT_ENCODER_PATH):
        logger.error(f"Text encoder path not found: {TEXT_ENCODER_PATH}")
        sys.exit(1)
    if not os.path.isfile(PROMPT_CONFIG):
        logger.error(f"Prompt config file not found: {PROMPT_CONFIG}")
        sys.exit(1)

    for lp in LORA_MODEL_PATHS:
        if not os.path.isfile(lp):
            logger.error(f"LoRA checkpoint not found: {lp}")
            sys.exit(1)

    if not LORA_STRENGTHS:
        logger.error("LORA_STRENGTHS list is empty.")
        sys.exit(1)

    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Set up tokenize / encoding strategies (required before any text processing)
    # ------------------------------------------------------------------
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=TEXT_ENCODER_PATH, t5_tokenizer_path=None,
        qwen3_max_length=512, t5_max_length=512,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(encoding_strategy)

    # ------------------------------------------------------------------
    # 2. Load VAE (shared across all tests)
    # ------------------------------------------------------------------
    logger.info("Loading VAE...")
    vae = load_vae(device)
    logger.info("VAE loaded.")

    # ------------------------------------------------------------------
    # 3. Load a temporary base DiT (no LoRA) for LLM Adapter preprocessing
    #    This is needed by encode_prompt() to run _preprocess_text_embeds().
    # ------------------------------------------------------------------
    logger.info("Loading temporary base DiT model for LLM Adapter preprocessing...")
    temp_anima = load_dit_with_lora(device, lora_weight_path=None, lora_multiplier=1.0)
    logger.info("Temporary DiT loaded.")

    # ------------------------------------------------------------------
    # 4. Load prompts from TOML
    # ------------------------------------------------------------------
    prompts = load_prompts(PROMPT_CONFIG)
    logger.info(f"Loaded {len(prompts)} prompt(s) from {PROMPT_CONFIG}")

    # ------------------------------------------------------------------
    # 5. Main generation loop
    #    For each prompt:
    #      For each LoRA checkpoint → reload TE with lora_te_, re-encode prompts
    #        For each strength → reload DiT with lora_unet_ at this strength, generate
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_lora = len(LORA_MODEL_PATHS)
    n_strength = len(LORA_STRENGTHS)

    for prompt_idx, pr in enumerate(prompts):
        prompt_text = pr.get("prompt", "")
        negative_text = pr.get("negative_prompt", "")
        short_prompt = prompt_text[:40].replace("/", "_").replace("\\", "_").replace(":", "_")
        prompt_dir = os.path.join(OUTPUT_DIR, f"prompt_{prompt_idx:02d}_{short_prompt}")
        os.makedirs(prompt_dir, exist_ok=True)

        height = pr.get("height", DEFAULT_HEIGHT)
        width = pr.get("width", DEFAULT_WIDTH)
        infer_steps = pr.get("sample_steps", DEFAULT_INFER_STEPS)
        guidance_scale = pr.get("scale", DEFAULT_GUIDANCE_SCALE)
        seed = pr.get("seed", DEFAULT_SEED)

        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt {prompt_idx+1}/{len(prompts)}: {prompt_text[:80]}...")
        logger.info(f"  Size: {width}x{height}, Steps: {infer_steps}, CFG: {guidance_scale}, Seed: {seed}")
        logger.info(f"{'='*60}")

        # Precompute negative-prompt embedding with base TE once (LoRA-independent).
        # We need a base TE for this — load it temporarily.
        logger.info("  Encoding negative prompt with base TE...")
        base_te = load_text_encoder(device)
        ctx_null = encode_prompt(
            base_te, temp_anima, tokenize_strategy, encoding_strategy,
            negative_text if negative_text else "", device,
        )
        del base_te
        clean_memory_on_device(device)

        # Prepare grid cell storage
        grid_cells: List[List[Image.Image]] = []
        row_labels: List[str] = []
        col_labels: List[str] = [strength_label(s) for s in LORA_STRENGTHS]

        for lora_idx, lora_path in enumerate(LORA_MODEL_PATHS):
            lora_name = short_lora_label(lora_path)
            row_labels.append(lora_name)
            row_images: List[Image.Image] = []

            # ---- 5a. Load Text Encoder with this checkpoint's lora_te_ weights ----
            logger.info(f"  [{lora_name}] Loading Text Encoder with LoRA TE weights...")
            text_encoder = load_text_encoder_with_lora(device, lora_path, lora_multiplier=1.0)
            text_encoder.to(device)

            # ---- 5b. Re-encode positive prompt with this LoRA-aware TE ----
            logger.info(f"  [{lora_name}] Encoding prompt with LoRA-aware TE...")
            ctx = encode_prompt(
                text_encoder, temp_anima, tokenize_strategy, encoding_strategy,
                prompt_text, device,
            )

            # Free TE to save VRAM before loading DiT
            del text_encoder
            clean_memory_on_device(device)

            # ---- 5c. For each strength, load DiT + LoRA and generate ----
            for strength_idx, strength in enumerate(LORA_STRENGTHS):
                logger.info(
                    f"  [{lora_name}] strength={strength:.1f}  "
                    f"(LoRA {lora_idx+1}/{n_lora}, strength {strength_idx+1}/{n_strength})"
                )

                # Load DiT with this LoRA's unet weights + multiplier
                anima = load_dit_with_lora(device, lora_weight_path=lora_path, lora_multiplier=strength)

                # Generate latent
                latent = generate_single(
                    anima, ctx, ctx_null, device,
                    height=height, width=width,
                    infer_steps=infer_steps, guidance_scale=guidance_scale,
                    flow_shift=DEFAULT_FLOW_SHIFT, seed=seed,
                )

                # Decode
                pil_img = decode_to_image(vae, latent, device)
                row_images.append(pil_img)

                # Save individual image
                fname = f"lora_{lora_idx:02d}_{lora_name}_strength_{strength:.1f}.png"
                pil_img.save(os.path.join(prompt_dir, fname))
                logger.info(f"    Saved: {fname}")

                # Free DiT model for this combination
                del anima, latent
                clean_memory_on_device(device)
                synchronize_device(device)

            grid_cells.append(row_images)

        # ------------------------------------------------------------------
        # 6. Build and save grid for this prompt
        # ------------------------------------------------------------------
        title = f"Prompt: {prompt_text[:60]}..."
        logger.info(f"  Assembling grid for prompt {prompt_idx+1}...")
        grid_img = make_grid(grid_cells, row_labels, col_labels, title=title)
        grid_path = os.path.join(prompt_dir, "_grid.png")
        grid_img.save(grid_path)
        logger.info(f"  Grid saved to: {grid_path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    del temp_anima, vae
    clean_memory_on_device(device)
    logger.info("\nAll tests complete!")
    logger.info(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    run_tests()

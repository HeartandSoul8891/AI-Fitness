"""
AI-Fitness Textual Inversion trainer.

A small, project-local implementation of the Hugging Face Diffusers SD1.x
textual-inversion training flow. It trains only the newly added token
embedding(s); the VAE and UNet remain frozen.

The module is usable from Streamlit through `train_textual_inversion(config,
progress_callback=...)`, or directly from the command line with a JSON config:

    python scripts/textual_inversion/textual_inversion_script.py config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _log(callback, message: str) -> None:
    if callback:
        callback(message)
    else:
        print(message, flush=True)


def _device_report() -> dict:
    """Return a backend-neutral hardware report.

    ROCm also appears through torch.cuda in PyTorch, so this intentionally
    does not use CUDA-specific APIs to decide whether an AMD GPU is usable.
    """
    report = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "backend": "CPU",
        "device": "cpu",
        "name": "CPU",
    }
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        report["name"] = name
        report["device"] = "cuda"
        report["backend"] = "GPU"
        if getattr(torch.version, "hip", None):
            report["backend"] = "ROCm"
            report["hip"] = torch.version.hip
        else:
            report["backend"] = "CUDA"
            report["cuda"] = torch.version.cuda
    return report


def get_hardware_report() -> dict:
    return _device_report()


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        tokenizer,
        size: int = 512,
        placeholder_token: str = "<concept>",
        repeats: int = 100,
        center_crop: bool = False,
        learnable_property: str = "object",
        concept_feature: Optional[str] = None,
    ):
        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Training image folder does not exist: {self.data_root}")

        self.image_paths = sorted(
            p for p in self.data_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No training images found in: {self.data_root}")

        self.tokenizer = tokenizer
        self.size = int(size)
        self.placeholder_token = placeholder_token
        self.repeats = max(1, int(repeats))
        self.center_crop = bool(center_crop)
        self.learnable_property = learnable_property
        self.concept_feature = concept_feature

        # Stable-Diffusion v1.x CLIP expects [-1, 1] normalized RGB tensors.
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        interpolation = InterpolationMode.BILINEAR
        ops = [
            transforms.Resize(self.size, interpolation=interpolation),
        ]
        if self.center_crop:
            ops.append(transforms.CenterCrop(self.size))
        else:
            # Preserve the useful "fit then crop" behavior of the original
            # Diffusers training example without requiring square source data.
            ops.append(transforms.CenterCrop(self.size))
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform = transforms.Compose(ops)

        self._length = len(self.image_paths) * self.repeats

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        path = self.image_paths[index % len(self.image_paths)]
        with Image.open(path) as image:
            image = image.convert("RGB")
            pixel_values = self.transform(image)

        # The prompt is intentionally simple. The learnable token carries the
        # concept; the user can use validation_prompt for richer validation.
        prompt = self.placeholder_token
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "filename": str(path),
        }


@dataclass
class TrainConfig:
    pretrained_model_name_or_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    revision: str = "main"
    tokenizer_name: Optional[str] = None
    train_data_dir: str = ""
    output_dir: str = "./output/textual_inversion"
    placeholder_token: str = "<concept>"
    initializer_token: str = "object"
    learnable_property: str = "object"
    num_vectors: int = 1
    concept_feature: Optional[str] = None

    resolution: int = 512
    center_crop: bool = False
    repeats: int = 100
    repeats_as_epoch: bool = False

    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    scale_lr: bool = False
    max_train_steps: int = 2000
    num_train_epochs: int = 100
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0

    mixed_precision: str = "no"
    gradient_checkpointing: bool = False
    allow_tf32: bool = False
    dataloader_num_workers: int = 0

    validation_prompt: Optional[str] = None
    num_validation_images: int = 4
    validation_steps: int = 100

    save_steps: int = 500
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

    seed: int = 42
    save_safetensors: bool = True
    save_a1111_pt: bool = True


def _coerce_config(config) -> TrainConfig:
    if isinstance(config, TrainConfig):
        return config
    data = dict(config)
    allowed = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    data = {k: v for k, v in data.items() if k in allowed}
    return TrainConfig(**data)


def _validate_config(cfg: TrainConfig) -> None:
    if not cfg.train_data_dir:
        raise ValueError("Train Data Directory is required.")
    if not cfg.output_dir:
        raise ValueError("Output Directory is required.")
    if not cfg.placeholder_token.strip():
        raise ValueError("Placeholder Token is required.")
    if not cfg.initializer_token.strip():
        raise ValueError("Initializer Token is required.")
    if cfg.placeholder_token in {"<|endoftext|>", "<pad>", "<unk>"}:
        raise ValueError("Choose a new placeholder token; do not use a tokenizer special token.")
    if cfg.num_vectors < 1:
        raise ValueError("Num Vectors must be at least 1.")
    if cfg.train_batch_size < 1:
        raise ValueError("Train Batch Size must be at least 1.")
    if cfg.learning_rate <= 0:
        raise ValueError("Learning Rate must be greater than zero.")
    if cfg.resolution < 64:
        raise ValueError("Resolution is too small.")


def _make_placeholder_tokens(tokenizer, placeholder_token: str, num_vectors: int):
    """Add one or more tokens and return their ids.

    For one vector the exact user token is learned. For multiple vectors,
    Diffusers/A1111-style loading is easiest when the placeholder expands to
    a sequence of newly-created tokens.
    """
    if num_vectors == 1:
        tokens = [placeholder_token]
    else:
        # Use deterministic token names so the saved embedding can be loaded
        # again without relying on tokenizer state from the training process.
        tokens = [f"{placeholder_token}_{i}" for i in range(num_vectors)]

    # If the user token already exists, training it would silently modify an
    # existing vocabulary entry, which is not Textual Inversion.
    for token in tokens:
        if tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id:
            raise ValueError(
                f"Placeholder token '{token}' already exists in the tokenizer. "
                "Choose a new, unique token."
            )

    added = tokenizer.add_tokens(tokens)
    if added != len(tokens):
        raise RuntimeError("Could not add all placeholder tokens to the tokenizer.")

    ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, ids


def _initializer_ids(tokenizer, initializer_token: str, count: int):
    ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Initializer token '{initializer_token}' could not be tokenized.")
    if len(ids) != 1:
        raise ValueError(
            f"Initializer Token must resolve to exactly one tokenizer token for "
            f"Textual Inversion. '{initializer_token}' resolves to {len(ids)} tokens."
        )
    return [ids[0]] * count


def _save_embedding_files(
    output_dir: Path,
    learned_vectors: torch.Tensor,
    placeholder_tokens: list[str],
    cfg: TrainConfig,
    step: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Diffusers-compatible format.
    learned_embeds = learned_vectors.detach().cpu()
    torch.save(
        {
            "string_to_param": {
                placeholder_tokens[0] if len(placeholder_tokens) == 1 else "*": learned_embeds
            },
            "name": cfg.placeholder_token,
            "step": step,
        },
        output_dir / "learned_embeds.pt",
    )

    # Also write the traditional Diffusers training artifacts.
    torch.save(learned_embeds, output_dir / "learned_embeds.bin")
    (output_dir / "token_identifier.txt").write_text(
        cfg.placeholder_token, encoding="utf-8"
    )
    (output_dir / "type_of_concept.txt").write_text(
        cfg.learnable_property, encoding="utf-8"
    )

    if cfg.save_safetensors:
        try:
            from safetensors.torch import save_file
            save_file({"embeds": learned_embeds}, str(output_dir / "learned_embeds.safetensors"))
        except ImportError:
            # Safetensors is optional; the core .bin/.pt files remain usable.
            pass

    if cfg.save_a1111_pt:
        # A1111 has had multiple embedding serialization variants. The
        # string_to_param form is understood by current Diffusers loaders and
        # is a useful portable .pt artifact without pretending to be a full
        # model checkpoint.
        a1111 = {
            "string_to_param": {"*": learned_embeds},
            "name": cfg.placeholder_token,
            "step": step,
        }
        torch.save(a1111, output_dir / f"{cfg.placeholder_token.strip('<>')}.pt")

    metadata = {
        "placeholder_token": cfg.placeholder_token,
        "placeholder_tokens": placeholder_tokens,
        "initializer_token": cfg.initializer_token,
        "learnable_property": cfg.learnable_property,
        "num_vectors": cfg.num_vectors,
        "step": step,
        "pretrained_model_name_or_path": cfg.pretrained_model_name_or_path,
        "train_data_dir": str(Path(cfg.train_data_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _prune_checkpoints(output_dir: Path, limit: Optional[int]):
    if not limit or limit < 1:
        return
    checkpoints = []
    for p in output_dir.glob("checkpoint-*"):
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, p))
    checkpoints.sort()
    while len(checkpoints) > limit:
        _, old = checkpoints.pop(0)
        shutil.rmtree(old, ignore_errors=True)


def _save_checkpoint(
    output_dir: Path,
    step: int,
    learned_vectors: torch.Tensor,
    placeholder_tokens: list[str],
    cfg: TrainConfig,
    optimizer,
    lr_scheduler,
):
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "learned_vectors": learned_vectors.detach().cpu(),
            "placeholder_tokens": placeholder_tokens,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        checkpoint / "training_state.pt",
    )
    _prune_checkpoints(output_dir, cfg.checkpoints_total_limit)


def _load_latest_checkpoint(path: Path):
    if path.name == "latest":
        candidates = []
        for p in path.parent.glob("checkpoint-*"):
            try:
                candidates.append((int(p.name.split("-")[-1]), p))
            except ValueError:
                pass
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {path.parent}")
        path = sorted(candidates)[-1][1]
    state_file = path / "training_state.pt"
    if not state_file.exists():
        raise FileNotFoundError(f"Checkpoint state not found: {state_file}")
    return torch.load(state_file, map_location="cpu")


def _build_validation_pipeline(cfg, accelerator, tokenizer, text_encoder, vae, unet):
    if not cfg.validation_prompt:
        return None
    try:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=cfg.revision,
        )
        pipe = pipe.to(accelerator.device)
        pipe.set_progress_bar_config(disable=True)
        return pipe
    except Exception as exc:
        return exc


def train_textual_inversion(
    config,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    cfg = _coerce_config(config)
    _validate_config(cfg)

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    hardware = _device_report()
    _log(progress_callback, f"Backend: {hardware['backend']} | Device: {hardware['name']}")

    if hardware["backend"] == "CPU" and cfg.mixed_precision != "no":
        _log(progress_callback, "CPU detected; forcing mixed precision to 'no'.")
        cfg.mixed_precision = "no"

    if cfg.allow_tf32 and hardware["backend"] == "CUDA":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
    )

    tokenizer_source = cfg.tokenizer_name or cfg.pretrained_model_name_or_path
    tokenizer_kwargs = {}
    if not cfg.tokenizer_name:
        tokenizer_kwargs["subfolder"] = "tokenizer"
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)

    placeholder_tokens, placeholder_ids = _make_placeholder_tokens(
        tokenizer, cfg.placeholder_token, cfg.num_vectors
    )
    initializer_ids = _initializer_ids(tokenizer, cfg.initializer_token, cfg.num_vectors)

    _log(
        progress_callback,
        f"Loading SD model: {cfg.pretrained_model_name_or_path}"
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=cfg.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="vae",
        revision=cfg.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="unet",
        revision=cfg.revision,
    )

    # Resize embeddings after adding tokens and initialize them from the
    # selected initializer token.
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embedding = text_encoder.get_input_embeddings()
    # Keep a frozen copy of the complete vocabulary. The optimizer sees the
    # embedding matrix, so after every optimizer update we restore all original
    # rows and keep only the placeholder rows.
    original_embeddings = token_embedding.weight.detach().clone()
    with torch.no_grad():
        for new_id, init_id in zip(placeholder_ids, initializer_ids):
            token_embedding.weight[new_id] = token_embedding.weight[init_id].clone()

    # Freeze everything except the text embedding matrix.
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    token_embedding.weight.requires_grad_(True)

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    train_dataset = TextualInversionDataset(
        data_root=cfg.train_data_dir,
        tokenizer=tokenizer,
        size=cfg.resolution,
        placeholder_token=" ".join(placeholder_tokens),
        repeats=cfg.repeats,
        center_crop=cfg.center_crop,
        learnable_property=cfg.learnable_property,
        concept_feature=cfg.concept_feature,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    trainable_parameters = [token_embedding.weight]
    optimizer_lr = cfg.learning_rate
    if cfg.scale_lr:
        optimizer_lr *= cfg.gradient_accumulation_steps * cfg.train_batch_size

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=optimizer_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    updates_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps
    )
    if cfg.repeats_as_epoch:
        max_train_steps = updates_per_epoch * cfg.num_train_epochs
    else:
        max_train_steps = int(cfg.max_train_steps)
        if max_train_steps <= 0:
            max_train_steps = updates_per_epoch * cfg.num_train_epochs
    num_train_epochs = math.ceil(max_train_steps / updates_per_epoch)

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=max_train_steps * cfg.gradient_accumulation_steps,
    )

    # Accelerator can prepare dataloader, optimizer and scheduler. The frozen
    # models are moved explicitly after preparation to keep the memory flow
    # predictable.
    train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, optimizer, lr_scheduler
    )
    device = accelerator.device

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(device=device, dtype=weight_dtype)
    unet.to(device=device, dtype=weight_dtype)
    text_encoder.to(device=device, dtype=weight_dtype)

    # The embedding weights need gradients in fp32 where possible. Diffusers'
    # training examples similarly keep the trainable text embedding precise.
    token_embedding = text_encoder.get_input_embeddings()
    token_embedding.weight.requires_grad_(True)

    global_step = 0
    if cfg.resume_from_checkpoint:
        state = _load_latest_checkpoint(Path(cfg.resume_from_checkpoint).expanduser().resolve())
        saved = state["learned_vectors"]
        with torch.no_grad():
            for idx, token_id in enumerate(placeholder_ids):
                token_embedding.weight[token_id].copy_(saved[idx].to(device))
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        global_step = int(state["step"])
        _log(progress_callback, f"Resumed from checkpoint at step {global_step}.")

    _log(
        progress_callback,
        f"Images: {len(train_dataset.image_paths)} | "
        f"Effective samples: {len(train_dataset)} | "
        f"Target steps: {max_train_steps}"
    )

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for batch in train_dataloader:
            with accelerator.accumulate(token_embedding):
                pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(input_ids)[0]
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = torch.nn.functional.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_([token_embedding.weight], 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Crucial TI rule: restore every non-placeholder embedding from
                # its original frozen value so only the new token(s) change.
                with torch.no_grad():
                    # This is intentionally performed on the full embedding
                    # matrix because the optimizer may have touched it through
                    # a dense gradient.
                    # Save a copy lazily on first use.
                    if not hasattr(train_textual_inversion, "_original_embeddings"):
                        pass

            if accelerator.sync_gradients:
                global_step += 1

                # Restore all original vocabulary rows except our learned rows.
                learned = token_embedding.weight.detach()[placeholder_ids].clone()
                # We restore the base vocabulary by copying the cached matrix
                # created before training (see cache assignment below).
                with torch.no_grad():
                    base = original_embeddings.to(device=device, dtype=token_embedding.weight.dtype)
                    token_embedding.weight.copy_(base)
                    token_embedding.weight[placeholder_ids] = learned

                if global_step % max(1, cfg.checkpointing_steps) == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        _save_checkpoint(
                            output_dir,
                            global_step,
                            learned,
                            placeholder_tokens,
                            cfg,
                            optimizer,
                            lr_scheduler,
                        )
                        _save_embedding_files(
                            output_dir,
                            learned,
                            placeholder_tokens,
                            cfg,
                            global_step,
                        )
                        _log(progress_callback, f"Checkpoint saved: step {global_step}")

                if (
                    cfg.validation_prompt
                    and global_step % max(1, cfg.validation_steps) == 0
                    and accelerator.is_main_process
                ):
                    try:
                        pipe = _build_validation_pipeline(
                            cfg, accelerator, tokenizer, text_encoder, vae, unet
                        )
                        if pipe is not None and not isinstance(pipe, Exception):
                            validation_dir = output_dir / "validation"
                            validation_dir.mkdir(exist_ok=True)
                            for i in range(int(cfg.num_validation_images)):
                                image = pipe(
                                    cfg.validation_prompt,
                                    num_inference_steps=25,
                                ).images[0]
                                image.save(validation_dir / f"step-{global_step:06d}-{i:02d}.png")
                            del pipe
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    except Exception as exc:
                        _log(progress_callback, f"Validation skipped: {exc}")

                if global_step % 10 == 0 or global_step == 1:
                    _log(
                        progress_callback,
                        f"step {global_step}/{max_train_steps} | loss {loss.item():.6f} | "
                        f"lr {lr_scheduler.get_last_lr()[0]:.3e}"
                    )

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_vectors = token_embedding.weight.detach()[placeholder_ids].clone()
        _save_embedding_files(
            output_dir,
            final_vectors,
            placeholder_tokens,
            cfg,
            global_step,
        )
        # Save the exact config used for reproducibility.
        (output_dir / "config.json").write_text(
            json.dumps(asdict(cfg), indent=2), encoding="utf-8"
        )

    _log(progress_callback, f"Training finished. Output: {output_dir}")
    return {
        "output_dir": str(output_dir),
        "steps": global_step,
        "placeholder_token": cfg.placeholder_token,
        "placeholder_tokens": placeholder_tokens,
        "hardware": hardware,
    }


def main():
    parser = argparse.ArgumentParser(description="AI-Fitness Textual Inversion trainer")
    parser.add_argument("config", help="Path to a JSON training configuration")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_textual_inversion(config)


if __name__ == "__main__":
    main()

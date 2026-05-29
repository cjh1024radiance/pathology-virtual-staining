import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.train_virtual_stain_torch import TrainStepConfig, VirtualStainTrainer
from model.virtual_stain_dataset_torch import (
    PairedVirtualStainDataset,
    VirtualStainDatasetConfig,
    build_pair_map,
    split_pairs_by_case,
)
from model.virtual_stain_resunet_torch import VirtualStainResUNetConfig
from train.virtual_stain_wsi_prep import WSIPrepareConfig, prepare_paired_dataset


CORE_METRIC_KEYS = (
    "total_loss",
    "weighted_l1",
    "lesion_region_ssim",
    "focus_reconstruction_loss",
    "clinical_consistency_loss",
    "he_aux_loss",
    "focus_weight_mean",
)


ABLATION_MODE_DESCRIPTIONS = {
    "full": "Full",
    "wo_attention_conditioned_input": "w/o attention-conditioned input",
    "wo_focus_reconstruction": "w/o focus reconstruction",
    "wo_clinical_consistency": "w/o clinical consistency",
    "wo_he_aux_head": "w/o H/E auxiliary head",
}


@dataclass
class TrainingRuntimeConfig:
    output_dir: str = "outputs/train/virtual_stain_resunet"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    epochs: int = 25
    batch_size: int = 4
    num_workers: int = 4
    seed: int = 42
    save_every: int = 5
    preview_every: int = 5
    device: str = "auto"
    ablation_mode: str = "full"


@dataclass
class FullTrainingConfig:
    prepare: WSIPrepareConfig
    dataset: VirtualStainDatasetConfig
    model: VirtualStainResUNetConfig
    train: TrainStepConfig
    runtime: TrainingRuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-command virtual stain training runner.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "training_config.json"))
    parser.add_argument("--ablation-mode", default=None, choices=sorted(ABLATION_MODE_DESCRIPTIONS.keys()))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_training_config(args.config)
    if args.ablation_mode:
        config.runtime.ablation_mode = args.ablation_mode
    apply_ablation_mode(config)
    set_seed(config.runtime.seed)

    print_stage("Loading config")
    print(
        json.dumps(
            {
                "config_path": args.config,
                "runtime": asdict(config.runtime),
                "ablation_description": ABLATION_MODE_DESCRIPTIONS[config.runtime.ablation_mode],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print_stage("Preparing paired training patches")
    prepared = prepare_paired_dataset(config.prepare)
    print(
        f"prepared_dir={prepared['prepared_dir']}\n"
        f"train_input_dir={prepared['train_input_dir']}\n"
        f"train_target_dir={prepared['train_target_dir']}\n"
        f"train_attention_dir={prepared['train_attention_dir']}\n"
        f"reused_existing={prepared['reused_existing']}"
    )
    output_dir = Path(config.runtime.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    pairs = build_pair_map(prepared["train_input_dir"], prepared["train_target_dir"], prepared["train_attention_dir"])
    train_pairs, val_pairs, test_pairs = split_pairs_by_case(
        pairs,
        train_ratio=config.runtime.train_ratio,
        val_ratio=config.runtime.val_ratio,
        seed=config.runtime.seed,
    )
    train_loader, val_loader, test_loader = build_dataloaders(train_pairs, val_pairs, test_pairs, config)
    if len(train_pairs) == 0:
        raise RuntimeError("No training pairs available after case-level split. Check prepared patches and split ratios.")
    print_stage("Dataset summary")
    print(
        f"total_pairs={len(pairs)}\n"
        f"train_pairs={len(train_pairs)}\n"
        f"val_pairs={len(val_pairs)}\n"
        f"test_pairs={len(test_pairs)}\n"
        f"train_batches={len(train_loader)}\n"
        f"val_batches={len(val_loader)}\n"
        f"test_batches={len(test_loader)}"
    )
    if len(pairs) < 100:
        print("warning=prepared patch pairs are quite small; check metadata.json and preparation thresholds if training still feels too fast.")
    if len(train_loader) < 10:
        print("warning=train_batches is very small; consider increasing top_k_per_slide or lowering pair_quality_threshold.")

    device = resolve_device(config.runtime.device)
    trainer = VirtualStainTrainer(model_config=config.model, train_config=config.train).to(device)
    optimizer = trainer.configure_optimizer()
    
    print_stage("Training setup")
    print(
        f"ablation_mode={config.runtime.ablation_mode}\n"
        f"ablation_description={ABLATION_MODE_DESCRIPTIONS[config.runtime.ablation_mode]}\n"
        f"device={device}\n"
        f"epochs={config.runtime.epochs}\n"
        f"batch_size={config.runtime.batch_size}\n"
        f"learning_rate={config.train.learning_rate}\n"
        f"patch_size={config.dataset.patch_size}\n"
        f"use_attention_conditioned_input={config.model.use_attention_conditioned_input}\n"
        f"focus_reconstruction_weight={config.model.focus_reconstruction_weight}\n"
        f"clinical_loss_weight={config.model.clinical_loss_weight}\n"
        f"use_he_aux_head={config.model.use_he_aux_head}\n"
        f"he_aux_weight={config.model.he_aux_weight}"
    )

    history_csv = output_dir / "history.csv"
    save_run_config(output_dir / "run_config.json", config, prepared, device)

    best_val = None
    best_epoch = 0
    for epoch in range(1, config.runtime.epochs + 1):
        print_stage(f"Epoch {epoch}/{config.runtime.epochs}")
        epoch_start = time.time()
        train_metrics = run_epoch(trainer, train_loader, optimizer, device, training=True, epoch=epoch, total_epochs=config.runtime.epochs)
        val_metrics = run_epoch(trainer, val_loader, optimizer, device, training=False, epoch=epoch, total_epochs=config.runtime.epochs)
        row = {
            "epoch": epoch,
            **prefix_metrics("train", select_core_metrics(train_metrics)),
            **prefix_metrics("val", select_core_metrics(val_metrics)),
        }
        append_history(history_csv, row)

        if config.runtime.save_every > 0 and epoch % config.runtime.save_every == 0:
            try_save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", trainer, optimizer, epoch, row)
        if best_val is None or val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            best_epoch = epoch
            try_save_checkpoint(ckpt_dir / "best.pt", trainer, optimizer, epoch, row)

        if config.runtime.preview_every > 0 and epoch % config.runtime.preview_every == 0:
            save_preview(preview_dir / f"epoch_{epoch:03d}.png", trainer, val_loader, device)
        elapsed = time.time() - epoch_start
        print(
            f"epoch={epoch} "
            f"train_total_loss={train_metrics['total_loss']:.6f} "
            f"val_total_loss={val_metrics['total_loss']:.6f} "
            f"train_l1={train_metrics.get('weighted_l1', 0.0):.6f} "
            f"train_focus_loss={train_metrics.get('focus_reconstruction_loss', 0.0):.6f} "
            f"val_clinical={val_metrics.get('clinical_consistency_loss', 0.0):.6f} "
            f"elapsed_sec={elapsed:.1f}"
        )

    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        trainer.load_state_dict(state["model_state_dict"])
    if len(test_loader) > 0:
        print_stage("Final test evaluation")
        test_metrics = run_epoch(trainer, test_loader, optimizer, device, training=False, epoch=best_epoch or config.runtime.epochs, total_epochs=config.runtime.epochs)
        with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
            json.dump({"best_epoch": best_epoch, "test_metrics": select_core_metrics(test_metrics)}, file, ensure_ascii=False, indent=2)
        print("test_metrics=" + json.dumps(select_core_metrics(test_metrics), ensure_ascii=False))


def load_training_config(path: str) -> FullTrainingConfig:
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    config = FullTrainingConfig(
        prepare=WSIPrepareConfig(**raw.get("prepare", {})),
        dataset=VirtualStainDatasetConfig(**raw.get("dataset", {})),
        model=VirtualStainResUNetConfig(**raw.get("model", {})),
        train=TrainStepConfig(**raw.get("train", {})),
        runtime=TrainingRuntimeConfig(**raw.get("runtime", {})),
    )
    config.prepare.dataset_dir = resolve_workspace_path(config.prepare.dataset_dir)
    config.prepare.prepared_dir = resolve_workspace_path(config.prepare.prepared_dir)
    config.runtime.output_dir = resolve_workspace_path(config.runtime.output_dir)
    return config


def apply_ablation_mode(config: FullTrainingConfig) -> None:
    mode = str(config.runtime.ablation_mode or "full").strip()
    if mode not in ABLATION_MODE_DESCRIPTIONS:
        raise ValueError(f"Unsupported ablation_mode: {mode}")
    config.runtime.ablation_mode = mode

    if mode == "full":
        return
    if mode == "wo_attention_conditioned_input":
        config.model.use_attention_conditioned_input = False
        return
    if mode == "wo_focus_reconstruction":
        config.model.focus_reconstruction_weight = 0.0
        config.model.background_reconstruction_weight = 1.0
        return
    if mode == "wo_clinical_consistency":
        config.model.clinical_loss_weight = 0.0
        return
    if mode == "wo_he_aux_head":
        config.model.use_he_aux_head = False
        config.model.he_aux_weight = 0.0
        return


def resolve_workspace_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((WORKSPACE_ROOT / path).resolve())


def build_dataloaders(
    train_pairs: List[Dict[str, str | None]],
    val_pairs: List[Dict[str, str | None]],
    test_pairs: List[Dict[str, str | None]],
    config: FullTrainingConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = PairedVirtualStainDataset(
        train_pairs,
        VirtualStainDatasetConfig(
            patch_size=config.dataset.patch_size,
            random_crop=True,
            horizontal_flip=config.dataset.horizontal_flip,
            vertical_flip=config.dataset.vertical_flip,
            normalize_to_unit=config.dataset.normalize_to_unit,
            use_generated_attention_if_missing=config.dataset.use_generated_attention_if_missing,
        ),
    )
    val_dataset = PairedVirtualStainDataset(
        val_pairs,
        VirtualStainDatasetConfig(
            patch_size=config.dataset.patch_size,
            random_crop=False,
            horizontal_flip=False,
            vertical_flip=False,
            normalize_to_unit=config.dataset.normalize_to_unit,
            use_generated_attention_if_missing=config.dataset.use_generated_attention_if_missing,
        ),
    )
    test_dataset = PairedVirtualStainDataset(
        test_pairs,
        VirtualStainDatasetConfig(
            patch_size=config.dataset.patch_size,
            random_crop=False,
            horizontal_flip=False,
            vertical_flip=False,
            normalize_to_unit=config.dataset.normalize_to_unit,
            use_generated_attention_if_missing=config.dataset.use_generated_attention_if_missing,
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.runtime.batch_size,
        shuffle=True,
        num_workers=config.runtime.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
    )
    return train_loader, val_loader, test_loader


def run_epoch(
    trainer: VirtualStainTrainer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: bool,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    metrics_list: List[Dict[str, float]] = []
    phase = "train" if training else "val"
    total_steps = max(1, len(loader))
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        if training:
            metrics = trainer.training_step(batch, optimizer)
        else:
            metrics = trainer.validation_step(batch)
        metrics_list.append(metrics)
        render_progress(
            phase=phase,
            epoch=epoch,
            total_epochs=total_epochs,
            step=step,
            total_steps=total_steps,
            metrics=metrics,
        )
    sys.stdout.write("\n")
    sys.stdout.flush()
    return average_metrics(metrics_list)


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def average_metrics(metrics_list: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(metrics_list)
    if not items:
        return {"total_loss": 0.0}
    keys = items[0].keys()
    return {key: float(sum(item[key] for item in items) / len(items)) for key in keys}


def prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def append_history(csv_path: Path, row: Dict[str, float | int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def select_core_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {key: float(metrics[key]) for key in CORE_METRIC_KEYS if key in metrics}


def save_checkpoint(
    path: Path,
    trainer: VirtualStainTrainer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float | int],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": trainer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_config": asdict(trainer.model_config),
            "train_config": asdict(trainer.train_config),
        },
        path,
    )


def try_save_checkpoint(
    path: Path,
    trainer: VirtualStainTrainer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float | int],
) -> None:
    try:
        save_checkpoint(path, trainer, optimizer, epoch, metrics)
    except Exception as exc:
        print(f"warning=checkpoint_save_failed path={path} error={exc}")


@torch.no_grad()
def save_preview(path: Path, trainer: VirtualStainTrainer, loader: DataLoader, device: torch.device) -> None:
    trainer.eval()
    try:
        sample = next(iter(loader))
    except StopIteration:
        return
    sample = move_batch_to_device(sample, device)
    outputs, _ = trainer.forward(sample)
    input_rgb = tensor_to_image(sample["input_rgb"][0])
    target_rgb = tensor_to_image(sample["target_rgb"][0])
    attention = tensor_to_gray(sample["attention_map"][0])
    pred_rgb = tensor_to_image(outputs["rgb"][0])
    hematoxylin = tensor_to_gray(outputs["hematoxylin"][0])
    eosin = tensor_to_gray(outputs["eosin"][0])

    row1 = np.concatenate([input_rgb, pred_rgb, target_rgb], axis=1)
    row2 = np.concatenate([attention, hematoxylin, eosin], axis=1)
    preview = np.concatenate([row1, row2], axis=0)
    Image.fromarray(preview).save(path)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().clamp(0.0, 1.0).numpy()
    array = (array.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return array


def tensor_to_gray(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().clamp(0.0, 1.0).numpy()
    if array.ndim == 3:
        array = array[0]
    gray = (array * 255.0).round().astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def save_run_config(path: Path, config: FullTrainingConfig, prepared: Dict[str, object], device: torch.device) -> None:
    content = {
        "prepare": asdict(config.prepare),
        "dataset": asdict(config.dataset),
        "model": asdict(config.model),
        "train": asdict(config.train),
        "runtime": asdict(config.runtime),
        "prepared": prepared,
        "device": str(device),
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def resolve_device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_stage(title: str) -> None:
    line = "=" * 18
    print(f"\n{line} {title} {line}")


def render_progress(
    phase: str,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    metrics: Dict[str, float],
    bar_width: int = 28,
) -> None:
    ratio = step / float(max(1, total_steps))
    filled = int(round(ratio * bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)
    total_loss = metrics.get("total_loss", 0.0)
    focus = metrics.get("focus_reconstruction_loss", 0.0)
    clinical = metrics.get("clinical_consistency_loss", 0.0)
    text = (
        f"\r[{phase}] epoch {epoch}/{total_epochs} "
        f"[{bar}] {step}/{total_steps} "
        f"loss={total_loss:.4f} "
        f"focus={focus:.4f} "
        f"clinical={clinical:.4f}"
    )
    sys.stdout.write(text)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

"""Fine-tune EfficientNet-S on the BirdVision species list and export to ONNX.

Data layout expected (ImageFolder format):
    <data_dir>/
        american_robin/
            img001.jpg
            img002.jpg
        dark_eyed_junco/
            ...

Folder names are normalized species names: lowercase, spaces → underscores,
apostrophes and hyphens removed.  Run --list-species to print expected folder
names for all species in the species list.

Typical workflow:
    1. Download images (iNaturalist export, see README)
    2. Organize into the folder layout above
    3. uv run scripts/train_efficientnet.py --data-dir ./train_data --output-dir ./pi/models
    4. Verify: onnxruntime inference on a held-out image

The script does two training phases:
    Phase 1 (head only):  backbone frozen, train the new classifier head
    Phase 2 (full):       unfreeze all layers, fine-tune end-to-end at low LR
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

IMG_SIZE = 224
BATCH_SIZE = 64
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 15
PHASE1_LR = 1e-3
PHASE2_LR = 5e-5
VAL_FRACTION = 0.15
LABEL_SMOOTHING = 0.1
NUM_WORKERS = min(8, os.cpu_count() or 4)


def normalize_species_name(name: str) -> str:
    """'American Robin' → 'american_robin'  (matches expected folder names)."""
    name = name.strip().lower()
    name = re.sub(r"['\u2019]", "", name)   # remove apostrophes
    name = re.sub(r"[\s\-]+", "_", name)    # spaces + hyphens → underscore
    name = re.sub(r"[^a-z0-9_]", "", name) # strip anything else
    return name


def load_species_list(path: Path) -> list[str]:
    species = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            species.append(line)
    return species


def build_folder_to_species_map(species_list: list[str]) -> dict[str, str]:
    """Return {normalized_folder_name: display_name}."""
    mapping: dict[str, str] = {}
    for s in species_list:
        key = normalize_species_name(s)
        if key in mapping:
            logger.warning("Duplicate normalized name: %s → %s (collision with %s)", s, key, mapping[key])
        mapping[key] = s
    return mapping


VALID_EXTS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}


class ImageFolderSkipEmpty(ImageFolderSkipEmpty):
    """ImageFolder that silently skips class directories with no valid images."""

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        classes, _ = super().find_classes(directory)
        non_empty = []
        for cls in classes:
            cls_dir = Path(directory) / cls
            if any(f.suffix.lower() in VALID_EXTS for f in cls_dir.iterdir()):
                non_empty.append(cls)
            else:
                logger.warning("Skipping empty class directory: %s", cls)
        if len(non_empty) < len(classes):
            logger.warning("Skipped %d empty class(es), training on %d",
                           len(classes) - len(non_empty), len(non_empty))
        class_to_idx = {cls: i for i, cls in enumerate(non_empty)}
        return non_empty, class_to_idx


def make_transforms(augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    bar = tqdm(loader, desc=f"epoch {epoch}", unit="batch", leave=False)
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
        bar.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{correct / total:.3f}")
    return running_loss / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    for images, labels in tqdm(loader, desc="val", unit="batch", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, top5 = outputs.topk(5, dim=1)
        correct_top1 += (top5[:, 0] == labels).sum().item()
        correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += images.size(0)
    return correct_top1 / total, correct_top5 / total


def export_onnx(model: nn.Module, output_path: Path, device: torch.device) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info("ONNX model written to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-S for BirdVision species classification")
    parser.add_argument("--data-dir", type=Path, help="ImageFolder-style training data directory")
    parser.add_argument("--species-list", type=Path, default=Path("data/species_lists/north_america_common.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("pi/models"))
    parser.add_argument("--checkpoint", type=Path, help="Resume from a .pt checkpoint")
    parser.add_argument("--phase1-epochs", type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--phase2-epochs", type=int, default=PHASE2_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list-species", action="store_true",
                        help="Print expected folder names and exit")
    args = parser.parse_args()

    species_list = load_species_list(args.species_list)
    folder_map = build_folder_to_species_map(species_list)

    if args.list_species:
        for folder, display in sorted(folder_map.items()):
            print(f"{folder:45s}  {display}")
        return

    if args.data_dir is None:
        parser.error("--data-dir is required unless --list-species is used")

    device = torch.device(args.device)
    logger.info("Device: %s  |  Species: %d  |  Data: %s", device, len(species_list), args.data_dir)

    # Dataset — train split uses augmentation, val split does not
    full_dataset = ImageFolderSkipEmpty(str(args.data_dir), transform=make_transforms(augment=True))
    val_size = max(1, int(len(full_dataset) * VAL_FRACTION))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    # Apply non-augmenting transform to val split
    val_ds.dataset = ImageFolderSkipEmpty(str(args.data_dir), transform=make_transforms(augment=False))

    # Map dataset folder-name classes → ordered species list
    # dataset.classes are folder names (sorted); build label→species mapping
    dataset_classes = full_dataset.classes  # folder names as found on disk
    folder_to_idx = full_dataset.class_to_idx

    # Build species_labels in dataset class index order (0..N-1)
    # We only include classes that are in our species list; warn about extras
    idx_to_species: dict[int, str] = {}
    unrecognized: list[str] = []
    for folder_name, idx in folder_to_idx.items():
        if folder_name in folder_map:
            idx_to_species[idx] = folder_map[folder_name]
        else:
            unrecognized.append(folder_name)

    if unrecognized:
        logger.warning("%d folder(s) not in species list (will be trained as-is): %s",
                       len(unrecognized), unrecognized[:10])
        # Fall back: use folder name as species display name
        for folder_name, idx in folder_to_idx.items():
            if idx not in idx_to_species:
                idx_to_species[idx] = folder_name

    num_classes = len(dataset_classes)
    species_labels = [idx_to_species[i] for i in range(num_classes)]
    logger.info("Classes in dataset: %d  |  Recognized species: %d",
                num_classes, sum(1 for f in dataset_classes if f in folder_map))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(num_classes).to(device)

    if args.checkpoint:
        logger.info("Resuming from checkpoint: %s", args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: head only
    if args.phase1_epochs > 0 and not args.checkpoint:
        logger.info("=== Phase 1: head-only training (%d epochs) ===", args.phase1_epochs)
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=PHASE1_LR, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase1_epochs)
        epoch_bar = tqdm(range(1, args.phase1_epochs + 1), desc="Phase 1", unit="epoch")
        for epoch in epoch_bar:
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
            top1, top5 = evaluate(model, val_loader, device)
            scheduler.step()
            elapsed = time.time() - t0
            epoch_bar.set_postfix(loss=f"{loss:.4f}", top1=f"{top1:.3f}", top5=f"{top5:.3f}", s=f"{elapsed:.0f}s")
            logger.info("Phase1 epoch %d/%d  loss=%.4f  val_top1=%.3f  val_top5=%.3f  %.1fs",
                        epoch, args.phase1_epochs, loss, top1, top5, elapsed)
        ckpt = args.output_dir / "efficientnet_s_birds_phase1.pt"
        torch.save(model.state_dict(), ckpt)
        logger.info("Phase 1 checkpoint saved: %s", ckpt)

    # Phase 2: full fine-tune
    if args.phase2_epochs > 0:
        logger.info("=== Phase 2: full fine-tune (%d epochs) ===", args.phase2_epochs)
        unfreeze_all(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)
        best_top1 = 0.0
        best_ckpt = args.output_dir / "efficientnet_s_birds_best.pt"
        epoch_bar = tqdm(range(1, args.phase2_epochs + 1), desc="Phase 2", unit="epoch")
        for epoch in epoch_bar:
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
            top1, top5 = evaluate(model, val_loader, device)
            scheduler.step()
            elapsed = time.time() - t0
            epoch_bar.set_postfix(loss=f"{loss:.4f}", top1=f"{top1:.3f}", top5=f"{top5:.3f}",
                                  best=f"{best_top1:.3f}", s=f"{elapsed:.0f}s")
            logger.info("Phase2 epoch %d/%d  loss=%.4f  val_top1=%.3f  val_top5=%.3f  %.1fs",
                        epoch, args.phase2_epochs, loss, top1, top5, elapsed)
            if top1 > best_top1:
                best_top1 = top1
                torch.save(model.state_dict(), best_ckpt)
                logger.info("  → new best val_top1=%.3f, checkpoint saved", best_top1)

        logger.info("Loading best checkpoint (val_top1=%.3f)", best_top1)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Final eval
    top1, top5 = evaluate(model, val_loader, device)
    logger.info("Final val_top1=%.3f  val_top5=%.3f", top1, top5)

    # Export ONNX
    onnx_path = args.output_dir / "efficientnet_s_birds.onnx"
    export_onnx(model, onnx_path, device)

    # Save species labels JSON (index order matches model output)
    labels_path = args.output_dir / "species_labels.json"
    labels_path.write_text(json.dumps(species_labels, indent=2))
    logger.info("Species labels written to %s (%d classes)", labels_path, len(species_labels))

    # Verify ONNX with onnxruntime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        import numpy as np
        dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        out = sess.run(None, {"input": dummy})
        assert out[0].shape == (1, num_classes), f"Unexpected output shape: {out[0].shape}"
        logger.info("ONNX verification passed — output shape %s", out[0].shape)
    except ImportError:
        logger.info("onnxruntime not installed, skipping ONNX verification")


if __name__ == "__main__":
    main()
